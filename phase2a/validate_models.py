"""
Quick validation script for trained depth-2 and depth-3 models.
Tests that each model achieves high accuracy on its respective task.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import SymbolicOntologyTokenizer
from model import InterpretableTransformer, TransformerConfig
from torch.utils.data import Dataset


# ============================================================================
# INLINE DATASET CLASS (works with list of samples, not file)
# ============================================================================

class SimpleDataset(Dataset):
    """Simple dataset that takes a list of samples directly."""

    def __init__(self, samples, tokenizer, max_input_len=512, max_target_len=32):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = self.tokenizer.encode(sample["input"], add_special_tokens=True)
        target_ids = self.tokenizer.encode(sample["target"], add_special_tokens=False)
        target_ids = target_ids + [self.tokenizer.eos_token_id]

        answer_token_id = self.tokenizer.token_to_id[self.tokenizer.ANSWER_TOKEN]
        newline_id = self.tokenizer.token_to_id["\n"]

        if input_ids[-1] == self.tokenizer.eos_token_id:
            input_ids = input_ids[:-1]

        max_input_tokens = self.max_input_len - 2
        if len(input_ids) > max_input_tokens:
            input_ids = input_ids[:max_input_tokens]
        if len(target_ids) > self.max_target_len:
            target_ids = target_ids[:self.max_target_len]

        full_ids = input_ids + [answer_token_id, newline_id] + target_ids
        max_len = self.max_input_len + self.max_target_len

        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]

        target_start = len(input_ids) + 2
        actual_content_len = len(full_ids)  # Length before padding
        attention_mask = [1] * actual_content_len
        padding_len = max_len - actual_content_len
        full_ids = full_ids + [self.tokenizer.pad_token_id] * padding_len
        attention_mask = attention_mask + [0] * padding_len

        labels = full_ids.copy()
        labels[:target_start] = [-100] * target_start
        # BUG FIX: Also mask padding at end so we don't compare against PAD tokens
        for i in range(actual_content_len, max_len):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

# Import data generation (copy the essential parts inline to avoid dependency issues)
import random
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque

# ============================================================================
# DATA GENERATOR (same as training notebook)
# ============================================================================

@dataclass
class OntologyNode:
    concept_id: int
    depth: int
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    properties: List[int] = field(default_factory=list)
    members: List[int] = field(default_factory=list)


class SymbolicOntologyGenerator:
    def __init__(self, depth: int = 3, branching_factor: int = 2,
                 num_properties: int = 5, property_assignment_prob: float = 0.4,
                 members_per_leaf: int = 2, seed: Optional[int] = None):
        self.depth = depth
        self.branching_factor = branching_factor
        self.num_properties = num_properties
        self.property_assignment_prob = property_assignment_prob
        self.members_per_leaf = members_per_leaf

        if seed is not None:
            random.seed(seed)

        self.nodes: Dict[int, OntologyNode] = {}
        self.num_concepts = 0
        self.num_members = 0
        self._generate_structure()

    def _generate_structure(self):
        root = OntologyNode(concept_id=0, depth=0)
        self.nodes[0] = root
        self.num_concepts = 1
        queue = deque([0])

        while queue:
            curr_id = queue.popleft()
            curr_node = self.nodes[curr_id]

            if random.random() < self.property_assignment_prob:
                used_props = self._get_ancestor_properties(curr_id)
                available_props = [p for p in range(1, self.num_properties + 1)
                                   if p not in used_props]
                if available_props:
                    prop_id = random.choice(available_props)
                    curr_node.properties.append(prop_id)

            if curr_node.depth < self.depth - 1:
                for _ in range(self.branching_factor):
                    child_id = self.num_concepts
                    self.num_concepts += 1
                    child = OntologyNode(
                        concept_id=child_id,
                        depth=curr_node.depth + 1,
                        parent_id=curr_id
                    )
                    curr_node.children_ids.append(child_id)
                    self.nodes[child_id] = child
                    queue.append(child_id)
            else:
                for _ in range(self.members_per_leaf):
                    member_id = self.num_members
                    curr_node.members.append(member_id)
                    self.num_members += 1

    def _get_ancestor_properties(self, concept_id: int) -> Set[int]:
        props = set()
        curr_id = concept_id
        while curr_id is not None:
            node = self.nodes[curr_id]
            props.update(node.properties)
            curr_id = node.parent_id
        return props

    def _get_all_descendants(self, concept_id: int) -> List[int]:
        descendants = [concept_id]
        queue = deque([concept_id])
        while queue:
            curr = queue.popleft()
            for child_id in self.nodes[curr].children_ids:
                descendants.append(child_id)
                queue.append(child_id)
        return descendants

    def _get_all_descendant_members(self, concept_id: int) -> List[Tuple[int, int]]:
        members = []
        descendants = self._get_all_descendants(concept_id)
        for desc_id in descendants:
            for member_id in self.nodes[desc_id].members:
                members.append((member_id, desc_id))
        return members


def generate_inductive_sample(gen: SymbolicOntologyGenerator) -> Optional[Dict]:
    candidates = []
    for concept_id, node in gen.nodes.items():
        if node.properties:
            descendant_members = gen._get_all_descendant_members(concept_id)
            if len(descendant_members) >= 2:
                for prop_id in node.properties:
                    candidates.append((concept_id, prop_id, descendant_members))

    if not candidates:
        return None

    target_c, target_p, all_members = random.choice(candidates)
    target_node = gen.nodes[target_c]

    num_observations = min(3, len(all_members))
    if len(all_members) <= num_observations:
        selected = all_members
    else:
        selected = random.sample(all_members, num_observations)

    observations = []
    for entity_id, leaf_id in selected:
        observations.append(f"c{leaf_id}(e{entity_id})")
        observations.append(f"p{target_p}(e{entity_id})")

    gt_hypothesis = f"∀x: c{target_c}(x) -> p{target_p}(x)"

    world_model = []
    for node in gen.nodes.values():
        if node.parent_id is not None:
            world_model.append(f"∀x: c{node.concept_id}(x) -> c{node.parent_id}(x)")
        for prop_id in node.properties:
            if not (node.concept_id == target_c and prop_id == target_p):
                world_model.append(f"∀x: c{node.concept_id}(x) -> p{prop_id}(x)")
        for member_id in node.members:
            world_model.append(f"c{node.concept_id}(e{member_id})")

    random.shuffle(world_model)
    random.shuffle(observations)

    prompt = "[WORLD_MODEL]\n" + "\n".join(world_model)
    prompt += "\n[OBSERVATIONS]\n" + "\n".join(observations)
    prompt += "\n[TASK]\nInfer the most general rule that explains all observations."

    return {
        "input": prompt,
        "target": gt_hypothesis,
        "task_type": "inductive",
        "metadata": {
            "target_concept": f"c{target_c}",
            "target_property": f"p{target_p}",
            "depth_of_truth": target_node.depth,
            "tree_depth": gen.depth,
        }
    }


def get_vocab_requirements(depth: int, branching: int = 2, members_per_leaf: int = 2):
    """Calculate vocabulary requirements for a given tree depth."""
    num_concepts = sum(branching**i for i in range(depth))
    num_leaves = branching ** (depth - 1)
    num_entities = num_leaves * members_per_leaf
    return num_concepts, num_entities


def generate_test_data(tree_depth: int, num_samples: int = 200, seed: int = 777) -> List[Dict]:
    """Generate test data filtered to depth_of_truth=0 (root-level targets)."""
    random.seed(seed)
    samples = []
    attempts = 0
    max_attempts = num_samples * 50  # Increased for deeper trees

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        # IMPORTANT: Use branching=2 to match training!
        branch = 2  # Fixed, not random.randint(2, 3)

        gen = SymbolicOntologyGenerator(
            depth=tree_depth,
            branching_factor=branch,
            num_properties=5,
            property_assignment_prob=0.4
        )

        sample = generate_inductive_sample(gen)
        if sample and sample['metadata']['depth_of_truth'] == 0:  # Filter to root-level
            sample["id"] = len(samples)
            samples.append(sample)

    return samples


def validate_data_generation(samples: List[Dict], tokenizer, depth: int) -> Dict:
    """Validate generated data for potential issues."""
    issues = []
    stats = {
        "total_samples": len(samples),
        "oov_tokens": 0,
        "max_concept_id": 0,
        "max_entity_id": 0,
        "max_property_id": 0,
    }

    import re
    concept_pattern = re.compile(r'c(\d+)')
    entity_pattern = re.compile(r'e(\d+)')
    property_pattern = re.compile(r'p(\d+)')

    for sample in samples:
        text = sample['input'] + sample['target']

        # Find all concept, entity, property IDs
        concepts = [int(m) for m in concept_pattern.findall(text)]
        entities = [int(m) for m in entity_pattern.findall(text)]
        properties = [int(m) for m in property_pattern.findall(text)]

        if concepts:
            stats["max_concept_id"] = max(stats["max_concept_id"], max(concepts))
        if entities:
            stats["max_entity_id"] = max(stats["max_entity_id"], max(entities))
        if properties:
            stats["max_property_id"] = max(stats["max_property_id"], max(properties))

        # Check if all tokens are in vocabulary
        try:
            input_ids = tokenizer.encode(sample['input'], add_special_tokens=True)
            target_ids = tokenizer.encode(sample['target'], add_special_tokens=False)

            # Check for UNK tokens
            unk_id = tokenizer.unk_token_id
            if unk_id in input_ids or unk_id in target_ids:
                stats["oov_tokens"] += 1
                issues.append(f"Sample {sample.get('id', '?')} has OOV tokens")
        except Exception as e:
            issues.append(f"Sample {sample.get('id', '?')} tokenization error: {e}")

    # Check vocab coverage
    if stats["max_concept_id"] >= tokenizer.max_concepts:
        issues.append(f"max_concept_id ({stats['max_concept_id']}) >= tokenizer.max_concepts ({tokenizer.max_concepts})")
    if stats["max_entity_id"] >= tokenizer.max_entities:
        issues.append(f"max_entity_id ({stats['max_entity_id']}) >= tokenizer.max_entities ({tokenizer.max_entities})")
    if stats["max_property_id"] > tokenizer.max_properties:
        issues.append(f"max_property_id ({stats['max_property_id']}) > tokenizer.max_properties ({tokenizer.max_properties})")

    # Expected values for this depth
    expected_concepts, expected_entities = get_vocab_requirements(depth, branching=2)
    stats["expected_concepts"] = expected_concepts
    stats["expected_entities"] = expected_entities

    return {"stats": stats, "issues": issues}


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, dataloader, device, tokenizer=None, debug_samples=0):
    """Compute sequence accuracy."""
    model.eval()
    correct = 0
    total = 0
    debug_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            predictions = outputs["logits"].argmax(dim=-1)

            shift_preds = predictions[:, :-1]
            shift_labels = labels[:, 1:]
            target_mask = (shift_labels != -100)

            for i in range(shift_labels.size(0)):
                mask_i = target_mask[i]
                if mask_i.sum() > 0:
                    is_correct = (shift_preds[i][mask_i] == shift_labels[i][mask_i]).all().item()
                    if is_correct:
                        correct += 1

                    # Debug: print first few predictions
                    if debug_count < debug_samples and tokenizer is not None:
                        pred_ids = shift_preds[i][mask_i].tolist()
                        label_ids = shift_labels[i][mask_i].tolist()
                        pred_str = tokenizer.decode(pred_ids, skip_special_tokens=False)
                        label_str = tokenizer.decode(label_ids, skip_special_tokens=False)
                        status = "✓" if is_correct else "✗"
                        print(f"  [{status}] Predicted: {pred_str}")
                        print(f"       Expected:  {label_str}")
                        debug_count += 1

                total += 1

    return correct / total if total > 0 else 0.0


def load_model(model_path: str, device: str, vocab_size: int = 89) -> InterpretableTransformer:
    """Load a trained model."""
    # Try to infer vocab size from checkpoint first
    checkpoint = torch.load(model_path, map_location=device)

    # Print checkpoint metadata for debugging
    print(f"\n  --- Checkpoint Metadata ---")
    if isinstance(checkpoint, dict):
        # Print all non-tensor keys (metadata)
        metadata_keys = ['tree_depth', 'best_val_acc', 'hops_required', 'data_format',
                         'epoch', 'step', 'optimizer', 'train_config', 'branching_factor',
                         'depth_of_truth_filter', 'training_samples', 'val_samples']
        found_metadata = False
        for key in checkpoint.keys():
            if key not in ['model_state_dict', 'optimizer_state_dict', 'config']:
                val = checkpoint[key]
                # Don't print large tensors or state dicts
                if not isinstance(val, (torch.Tensor, dict)):
                    print(f"    {key}: {val}")
                    found_metadata = True
                elif key == 'config':
                    print(f"    {key}: <config object>")

        # Also check for config object
        if 'config' in checkpoint:
            cfg = checkpoint['config']
            if hasattr(cfg, '__dict__'):
                print(f"    config.vocab_size: {getattr(cfg, 'vocab_size', 'N/A')}")

        if not found_metadata:
            print("    (No metadata found - raw state dict or minimal checkpoint)")

        # Print state dict keys count
        if 'model_state_dict' in checkpoint:
            print(f"    model layers: {len(checkpoint['model_state_dict'])} keys")
    else:
        print("    (Raw state dict format - no metadata)")
    print(f"  ----------------------------\n")

    # Check if checkpoint has config info
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        # Use saved config
        saved_config = checkpoint['config']
        if hasattr(saved_config, 'vocab_size'):
            vocab_size = saved_config.vocab_size
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Infer from embedding size
        state_dict = checkpoint['model_state_dict']
        if 'token_embedding.weight' in state_dict:
            vocab_size = state_dict['token_embedding.weight'].shape[0]
    else:
        # Raw state dict - infer from embedding
        if 'token_embedding.weight' in checkpoint:
            vocab_size = checkpoint['token_embedding.weight'].shape[0]

    # Create config with correct vocab size
    config = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=544,
        n_layers=4,
        n_heads=1,
        d_model=64,
        d_ff=256,
        pad_token_id=0,
        concat_pos_emb=True,
        pre_ln=True,
    )

    model = InterpretableTransformer(config)

    # Load state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, vocab_size


def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("=" * 60)

    # Model paths - check current directory and parent
    model_paths = {
        2: ['best_model_depth2.pt', 'depth2_trained.pt', '../depth2_trained.pt'],
        3: ['best_model_depth3.pt', 'depth3_trained.pt', '../depth3_trained.pt'],
        4: ['best_model_depth4.pt', 'depth4_trained.pt', '../depth4_trained.pt'],
        5: ['best_model_depth5.pt', 'depth5_trained.pt', '../depth5_trained.pt'],
        6: ['best_model_depth6.pt', 'depth6_trained.pt', '../depth6_trained.pt'],
    }

    results = {}

    # Test all depths that have models available
    for depth in [2, 3, 4, 5, 6]:
        hops = depth - 1
        print(f"\n{'='*60}")
        print(f"DEPTH-{depth} MODEL ({hops}-hop reasoning)")
        print(f"{'='*60}")

        # Find model file
        model_path = None
        for path in model_paths[depth]:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            print(f"ERROR: No model found for depth-{depth}")
            print(f"Looked for: {model_paths[depth]}")
            continue

        print(f"Loading model from: {model_path}")

        try:
            model, vocab_size = load_model(model_path, device)
            print(f"Model loaded successfully (vocab_size={vocab_size})")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            continue

        # Calculate vocab requirements for this depth
        required_concepts, required_entities = get_vocab_requirements(depth, branching=2)
        vocab_concepts = max(30, required_concepts + 5)
        vocab_entities = max(30, required_entities + 5)

        # Create tokenizer - need to match model's vocab_size
        # vocab_size = 14 (fixed tokens) + concepts + 15 (properties) + entities
        # So: concepts + entities = vocab_size - 29
        # We'll try to create tokenizer that matches model, or at least covers test data
        tokenizer = SymbolicOntologyTokenizer(
            max_concepts=vocab_concepts,
            max_entities=vocab_entities,
            max_seq_len=544
        )

        # Check if tokenizer vocab matches model vocab
        if tokenizer.vocab_size != vocab_size:
            print(f"WARNING: Tokenizer vocab ({tokenizer.vocab_size}) != model vocab ({vocab_size})")
            # Try to adjust tokenizer to match model
            # vocab_size = 14 + concepts + 15 + entities = 29 + concepts + entities
            target_sum = vocab_size - 29
            # Distribute evenly, but ensure we have enough for the depth
            adjusted_concepts = max(vocab_concepts, (target_sum + 1) // 2)
            adjusted_entities = target_sum - adjusted_concepts + vocab_concepts
            if adjusted_entities < vocab_entities:
                adjusted_entities = vocab_entities
                adjusted_concepts = target_sum - adjusted_entities + vocab_entities
            tokenizer = SymbolicOntologyTokenizer(
                max_concepts=adjusted_concepts,
                max_entities=adjusted_entities,
                max_seq_len=544
            )
            print(f"Adjusted tokenizer: {adjusted_concepts} concepts, {adjusted_entities} entities (vocab={tokenizer.vocab_size})")

        print(f"Tokenizer: {tokenizer.max_concepts} concepts, {tokenizer.max_entities} entities (vocab={tokenizer.vocab_size})")

        # Generate test data
        print(f"Generating {200} test samples (depth={depth}, depth_of_truth=0)...")
        test_samples = generate_test_data(tree_depth=depth, num_samples=200, seed=777)
        print(f"Generated {len(test_samples)} samples")

        if len(test_samples) == 0:
            print("ERROR: Could not generate test samples")
            continue

        # Validate data generation
        validation = validate_data_generation(test_samples, tokenizer, depth)
        stats = validation["stats"]
        issues = validation["issues"]

        print(f"\n  --- Data Generation Validation (depth {depth}) ---")
        print(f"  Samples generated: {stats['total_samples']}")
        print(f"  Max concept ID: c{stats['max_concept_id']} (expected max: c{stats['expected_concepts']-1}, tokenizer supports: c0-c{tokenizer.max_concepts-1})")
        print(f"  Max entity ID: e{stats['max_entity_id']} (expected max: e{stats['expected_entities']-1}, tokenizer supports: e0-e{tokenizer.max_entities-1})")
        print(f"  Max property ID: p{stats['max_property_id']} (tokenizer supports: p1-p{tokenizer.max_properties})")
        print(f"  OOV tokens: {stats['oov_tokens']}")

        if issues:
            print(f"  ⚠️ ISSUES FOUND:")
            for issue in issues[:5]:  # Show first 5 issues
                print(f"    - {issue}")
            if len(issues) > 5:
                print(f"    ... and {len(issues) - 5} more issues")
        else:
            print(f"  ✓ No issues found")

        # Debug: Show first sample format for depth >= 4
        if depth >= 4:
            sample = test_samples[0]
            print(f"\n  Sample format preview:")
            print(f"  Input (first 300 chars): {sample['input'][:300]}...")
            print(f"  Target: {sample['target']}")
        print(f"  -----------------------------------------------\n")

        # Create dataset and dataloader
        test_dataset = SimpleDataset(
            test_samples, tokenizer,
            max_input_len=512, max_target_len=32
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Evaluate (show debug output for depth >= 4 to diagnose issues)
        debug_samples = 5 if depth >= 4 else 0
        print(f"Evaluating... (showing {debug_samples} sample predictions)")
        accuracy = evaluate_model(model, test_loader, device, tokenizer=tokenizer, debug_samples=debug_samples)
        results[depth] = accuracy

        print(f"\n>>> DEPTH-{depth} ACCURACY: {accuracy:.1%} <<<")

        if accuracy > 0.9:
            print("    Status: PASSED (>90%)")
        elif accuracy > 0.5:
            print("    Status: PARTIAL (50-90%)")
        else:
            print("    Status: FAILED (<50%)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for depth, acc in results.items():
        hops = depth - 1
        status = "PASS" if acc > 0.9 else "PARTIAL" if acc > 0.5 else "FAIL"
        print(f"Depth-{depth} ({hops}-hop): {acc:.1%} [{status}]")

    print("\nInterpretation:")
    if len(results) >= 2:
        # Find the boundary where accuracy drops
        passing = [d for d, acc in results.items() if acc > 0.9]
        failing = [d for d, acc in results.items() if acc <= 0.9]

        if failing:
            min_fail = min(failing)
            max_pass = max(passing) if passing else 0
            print(f"  BOUNDARY FOUND: {max_pass}-hop works, {min_fail - 1}-hop fails!")
            print(f"  Architectural limit appears to be at {min_fail - 1}+ hops.")
        elif len(passing) == len(results):
            max_tested = max(results.keys())
            print(f"  All tested depths pass ({max_tested}-hop max)!")
            if max_tested < 5:
                print(f"  Consider testing depth-{max_tested + 1} ({max_tested}-hop) to find the limit.")
            else:
                print("  Model handles deep multi-hop reasoning well.")
        else:
            print("  Mixed results - investigate further.")


if __name__ == "__main__":
    main()
