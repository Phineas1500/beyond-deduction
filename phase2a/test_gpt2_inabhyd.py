"""
Test LLMs on INABHYD-style natural language inductive reasoning.

This script tests whether LLMs show similar multi-hop reasoning
limitations as our custom transformer on symbolic ontology tasks.

Supported models:
- GPT-2 family (gpt2, gpt2-medium, gpt2-large) - 1024 context
- GPT-Neo family (gpt-neo-125m, gpt-neo-1.3b) - 2048 context
- Pythia family (pythia-160m, pythia-410m, pythia-1b) - 2048 context
- StableLM (stablelm-3b) - 4096 context
- Qwen family (qwen-0.6b, qwen-1.7b) - 32k context

Usage:
    python test_gpt2_inabhyd.py --model gpt2 --num-samples 50
    python test_gpt2_inabhyd.py --model qwen-0.6b --num-samples 100

Requirements:
    pip install transformers torch
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import deque
import torch
import argparse

# Import transformers components
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Please install/reinstall transformers: pip install transformers --upgrade")
    exit(1)


# =============================================================================
# INABHYD-style vocabulary (from their morphology.py)
# =============================================================================

CONCEPT_NAMES = [
    "wumpus", "yumpus", "zumpus", "dumpus", "rompus",
    "numpus", "tumpus", "vumpus", "impus", "jompus",
    "lorpus", "borpus", "corpus", "dorpus", "forpus",  # Extended
]

PROPERTY_NAMES = [
    "red", "blue", "small", "large", "cold", "hot",
    "bright", "dark", "sweet", "bitter", "loud", "quiet",
    "fast", "slow", "happy", "sad"
]

ENTITY_NAMES = [
    "Alex", "Sam", "Max", "Pat", "Rex", "Wex", "Zex", "Tex",
    "Lex", "Dex", "Jex", "Kex", "Nex", "Bex", "Fex", "Gex",
]


# =============================================================================
# Natural Language Ontology Generator
# =============================================================================

@dataclass
class NLOntologyNode:
    """A concept node in the ontology tree with natural language names."""
    concept_name: str
    concept_id: int
    depth: int
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    members: List[str] = field(default_factory=list)


class NLOntologyGenerator:
    """
    Generates ontology trees with natural language names (INABHYD-style).
    """

    def __init__(self, depth: int = 3, branching_factor: int = 2,
                 num_properties: int = 5, property_prob: float = 0.4,
                 members_per_leaf: int = 2, seed: Optional[int] = None):
        self.depth = depth
        self.branching_factor = branching_factor
        self.num_properties = num_properties
        self.property_prob = property_prob
        self.members_per_leaf = members_per_leaf

        if seed is not None:
            random.seed(seed)

        # Shuffle vocabularies for variety
        self.concept_vocab = CONCEPT_NAMES.copy()
        self.property_vocab = PROPERTY_NAMES[:num_properties]
        self.entity_vocab = ENTITY_NAMES.copy()
        random.shuffle(self.concept_vocab)
        random.shuffle(self.property_vocab)
        random.shuffle(self.entity_vocab)

        self.nodes: Dict[int, NLOntologyNode] = {}
        self.num_concepts = 0
        self.num_members = 0
        self._generate_structure()

    def _generate_structure(self):
        """Build the ontology tree with BFS traversal."""
        root_name = self.concept_vocab[0]
        root = NLOntologyNode(concept_name=root_name, concept_id=0, depth=0)
        self.nodes[0] = root
        self.num_concepts = 1

        queue = deque([0])

        while queue:
            curr_id = queue.popleft()
            curr_node = self.nodes[curr_id]

            # Maybe assign a property
            if random.random() < self.property_prob:
                used_props = self._get_ancestor_properties(curr_id)
                available = [p for p in self.property_vocab if p not in used_props]
                if available:
                    prop = random.choice(available)
                    curr_node.properties.append(prop)

            # Add children if not at max depth
            if curr_node.depth < self.depth - 1:
                for _ in range(self.branching_factor):
                    child_id = self.num_concepts
                    child_name = self.concept_vocab[child_id % len(self.concept_vocab)]
                    self.num_concepts += 1

                    child = NLOntologyNode(
                        concept_name=child_name,
                        concept_id=child_id,
                        depth=curr_node.depth + 1,
                        parent_id=curr_id
                    )
                    curr_node.children_ids.append(child_id)
                    self.nodes[child_id] = child
                    queue.append(child_id)
            else:
                # Leaf node: assign members
                for _ in range(self.members_per_leaf):
                    entity_name = self.entity_vocab[self.num_members % len(self.entity_vocab)]
                    curr_node.members.append(entity_name)
                    self.num_members += 1

    def _get_ancestor_properties(self, concept_id: int) -> Set[str]:
        """Get all properties from ancestors."""
        props = set()
        curr_id = concept_id
        while curr_id is not None:
            node = self.nodes[curr_id]
            props.update(node.properties)
            curr_id = node.parent_id
        return props

    def _get_all_descendant_members(self, concept_id: int) -> List[Tuple[str, int]]:
        """Get all entity members under a concept."""
        members = []
        queue = deque([concept_id])
        while queue:
            curr = queue.popleft()
            node = self.nodes[curr]
            for member in node.members:
                members.append((member, curr))
            for child_id in node.children_ids:
                queue.append(child_id)
        return members


def generate_nl_inductive_sample(gen: NLOntologyGenerator) -> Optional[Dict]:
    """
    Generate a natural language inductive reasoning task.

    Returns a dict with:
    - theories: The world model in natural language
    - observations: What we observe
    - hypothesis: The rule to induce (target)
    - metadata: Extra info
    """
    # Find concepts with properties that have descendant members
    candidates = []
    for concept_id, node in gen.nodes.items():
        if node.properties:
            members = gen._get_all_descendant_members(concept_id)
            if len(members) >= 2:
                for prop in node.properties:
                    candidates.append((concept_id, prop, members))

    if not candidates:
        return None

    target_id, target_prop, all_members = random.choice(candidates)
    target_node = gen.nodes[target_id]

    # Select entities for observations
    num_obs = min(3, len(all_members))
    selected = random.sample(all_members, num_obs) if len(all_members) > num_obs else all_members

    # Build theories (world model) in natural language
    theories = []
    for node in gen.nodes.values():
        # Subtype relations: "Every yumpus is a wumpus"
        if node.parent_id is not None:
            parent = gen.nodes[node.parent_id]
            theories.append(f"Every {node.concept_name} is a {parent.concept_name}")

        # Property rules (except the hidden target)
        for prop in node.properties:
            if not (node.concept_id == target_id and prop == target_prop):
                theories.append(f"Every {node.concept_name} is {prop}")

        # Membership: "Alex is a yumpus"
        for member in node.members:
            theories.append(f"{member} is a {node.concept_name}")

    random.shuffle(theories)

    # Build observations
    observations = []
    for entity, leaf_id in selected:
        leaf_node = gen.nodes[leaf_id]
        observations.append(f"{entity} is a {leaf_node.concept_name}")
        observations.append(f"{entity} is {target_prop}")

    random.shuffle(observations)

    # Ground truth hypothesis
    hypothesis = f"Every {target_node.concept_name} is {target_prop}"

    return {
        "theories": ". ".join(theories) + ".",
        "observations": ". ".join(observations) + ".",
        "hypothesis": hypothesis,
        "metadata": {
            "target_concept": target_node.concept_name,
            "target_property": target_prop,
            "depth_of_truth": target_node.depth,
            "tree_depth": gen.depth,
            "hops_required": gen.depth - 1 - target_node.depth,
        }
    }


def generate_nl_dataset(num_samples: int, tree_depth: int, seed: int = 42) -> List[Dict]:
    """Generate dataset filtered to depth_of_truth=0 (root-level targets)."""
    random.seed(seed)
    samples = []
    attempts = 0
    max_attempts = num_samples * 50

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1

        gen = NLOntologyGenerator(
            depth=tree_depth,
            branching_factor=2,
            num_properties=5,
            property_prob=0.4,
            seed=None  # Use global seed
        )

        sample = generate_nl_inductive_sample(gen)
        if sample and sample['metadata']['depth_of_truth'] == 0:
            sample['id'] = len(samples)
            samples.append(sample)

    return samples


# =============================================================================
# Model Configuration
# =============================================================================

MODEL_CONFIGS = {
    # GPT-2 family (1024 context)
    "gpt2": {"hf_name": "gpt2", "context_len": 1024},
    "gpt2-medium": {"hf_name": "gpt2-medium", "context_len": 1024},
    "gpt2-large": {"hf_name": "gpt2-large", "context_len": 1024},
    # GPT-Neo family (2048 context)
    "gpt-neo-125m": {"hf_name": "EleutherAI/gpt-neo-125M", "context_len": 2048},
    "gpt-neo-1.3b": {"hf_name": "EleutherAI/gpt-neo-1.3B", "context_len": 2048},
    # Pythia family (2048 context)
    "pythia-160m": {"hf_name": "EleutherAI/pythia-160m", "context_len": 2048},
    "pythia-410m": {"hf_name": "EleutherAI/pythia-410m", "context_len": 2048},
    "pythia-1b": {"hf_name": "EleutherAI/pythia-1b", "context_len": 2048},
    # StableLM (4096 context)
    "stablelm-3b": {"hf_name": "stabilityai/stablelm-3b-4e1t", "context_len": 4096},
    # Qwen family (32k context - great for deep trees!)
    "qwen-0.6b": {"hf_name": "Qwen/Qwen3-0.6B", "context_len": 32768},
    "qwen-1.7b": {"hf_name": "Qwen/Qwen3-1.7B", "context_len": 32768},
    # Phi family (2048 context, but very capable)
    "phi-2": {"hf_name": "microsoft/phi-2", "context_len": 2048},
}


# =============================================================================
# LLM Evaluation
# =============================================================================

def create_few_shot_prompt(test_sample: Dict, few_shot_examples: List[Dict]) -> str:
    """Create a few-shot prompt for GPT-2."""
    prompt = "Task: Given facts about a world, induce the general rule that explains observations.\n\n"

    # Add few-shot examples
    for i, ex in enumerate(few_shot_examples):
        prompt += f"Example {i+1}:\n"
        prompt += f"Facts: {ex['theories']}\n"
        prompt += f"Observations: {ex['observations']}\n"
        prompt += f"Rule: {ex['hypothesis']}\n\n"

    # Add test example
    prompt += "Now solve this:\n"
    prompt += f"Facts: {test_sample['theories']}\n"
    prompt += f"Observations: {test_sample['observations']}\n"
    prompt += "Rule:"

    return prompt


def generate_text(model, tokenizer, prompt: str, device: str, max_context: int = 1024, max_new_tokens: int = 30) -> str:
    """Generate text using greedy decoding."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Check if prompt is too long - truncate from beginning, leaving room for generation
    max_input = max_context - max_new_tokens - 10  # Leave buffer
    if input_ids.shape[1] > max_input:
        print(f"  [WARNING] Input truncated from {input_ids.shape[1]} to {max_input} tokens")
        input_ids = input_ids[:, -max_input:]

    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop at newline or EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            # Check for newline (handle different tokenizers)
            decoded = tokenizer.decode([next_token.item()])
            if '\n' in decoded:
                break

    # Return only the generated part
    generated_ids = input_ids[0, prompt_len:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated


def evaluate_nl(model, tokenizer, test_samples: List[Dict],
                few_shot_examples: List[Dict], device: str,
                max_context: int = 1024, debug: bool = False) -> Dict:
    """Evaluate LLM on natural language test samples."""
    correct = 0
    total = 0
    results = []

    for i, sample in enumerate(test_samples):
        prompt = create_few_shot_prompt(sample, few_shot_examples)

        # Generate using manual loop - now returns only generated text
        generated_answer = generate_text(model, tokenizer, prompt, device, max_context=max_context)
        generated_answer = generated_answer.strip()

        # Debug: show first few raw outputs
        if debug and i < 3:
            print(f"  DEBUG raw output: '{generated_answer[:100]}'")

        # Check if correct (flexible matching)
        expected = sample['hypothesis'].lower()
        generated_lower = generated_answer.lower()

        # Check for exact match or key components
        is_correct = (
            expected in generated_lower or
            (sample['metadata']['target_concept'].lower() in generated_lower and
             sample['metadata']['target_property'].lower() in generated_lower and
             "every" in generated_lower)
        )

        if is_correct:
            correct += 1
        total += 1

        results.append({
            'expected': sample['hypothesis'],
            'generated': generated_answer[:100],  # Truncate for display
            'correct': is_correct,
            'metadata': sample['metadata']
        })

    return {
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description="Test LLMs on INABHYD-style natural language tasks")
    parser.add_argument("--model", default="gpt2", choices=list(MODEL_CONFIGS.keys()),
                        help="Model to test (see MODEL_CONFIGS for options)")
    parser.add_argument("--num-samples", type=int, default=50, help="Samples per depth")
    parser.add_argument("--num-few-shot", type=int, default=3, help="Few-shot examples")
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Get model config
    model_config = MODEL_CONFIGS[args.model]
    hf_name = model_config["hf_name"]
    max_context = model_config["context_len"]

    # Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer using AutoModel/AutoTokenizer
    print(f"\nLoading {args.model} ({hf_name})...")
    print(f"Context length: {max_context} tokens")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForCausalLM.from_pretrained(hf_name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")

    # Results storage
    all_results = {}

    print("\n" + "="*60)
    print(f"TESTING {args.model.upper()} ON INABHYD-STYLE NATURAL LANGUAGE REASONING")
    print("="*60)

    for depth in range(args.min_depth, args.max_depth + 1):
        hops = depth - 1
        print(f"\n{'='*60}")
        print(f"DEPTH-{depth} ({hops}-hop reasoning)")
        print(f"{'='*60}")

        # Generate test data
        print(f"Generating {args.num_samples} test samples...")
        test_samples = generate_nl_dataset(args.num_samples, depth, seed=args.seed + depth)

        if len(test_samples) < args.num_samples:
            print(f"WARNING: Only generated {len(test_samples)} samples")

        if len(test_samples) == 0:
            print("ERROR: Could not generate samples for this depth")
            continue

        # Generate few-shot examples from simpler depth
        few_shot_depth = min(depth, 2)  # Use depth-2 examples for few-shot
        few_shot_samples = generate_nl_dataset(args.num_few_shot * 2, few_shot_depth, seed=args.seed + 100)
        few_shot_examples = few_shot_samples[:args.num_few_shot]

        # Show sample
        sample = test_samples[0]
        print(f"\nSample (truncated):")
        print(f"  Facts: {sample['theories'][:150]}...")
        print(f"  Observations: {sample['observations']}")
        print(f"  Expected: {sample['hypothesis']}")

        # Evaluate (debug=True for first depth to see raw output)
        print(f"\nEvaluating {len(test_samples)} samples...")
        result = evaluate_nl(model, tokenizer, test_samples, few_shot_examples, device,
                             max_context=max_context, debug=(depth == args.min_depth))
        all_results[depth] = result

        # Show some predictions
        print(f"\nSample predictions:")
        for r in result['results'][:3]:
            status = "✓" if r['correct'] else "✗"
            print(f"  [{status}] Expected: {r['expected']}")
            print(f"       Got: {r['generated'][:60]}...")

        print(f"\n>>> DEPTH-{depth} ACCURACY: {result['accuracy']:.1%} ({result['correct']}/{result['total']}) <<<")

    # Summary
    print("\n" + "="*60)
    print(f"SUMMARY: {args.model.upper()} ON NATURAL LANGUAGE FORMAT")
    print("="*60)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params/1e6:.1f}M params, context: {max_context})")
    print()
    for depth, result in all_results.items():
        hops = depth - 1
        acc = result['accuracy']
        status = "PASS" if acc > 0.8 else "PARTIAL" if acc > 0.4 else "FAIL"
        print(f"Depth-{depth} ({hops}-hop): {acc:.1%} [{status}]")

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print("\nCustom Transformer (1.9M params, trained on symbolic):")
    print("  Depth-2: 100.0%  |  Depth-3: 100.0%  |  Depth-4: 100.0%  |  Depth-5: 82.5%")

    print(f"\nGPT-2 on Natural Language (few-shot, 124M params):")
    print("  Depth-2: 42.0%   |  Depth-3: 16.0%   |  Depth-4: 6.0%    |  Depth-5: 8.0%")

    print(f"\nQwen3-0.6B on Symbolic (few-shot, 596M params):")
    print("  Depth-2: 93.0%   |  Depth-3: 91.0%   |  Depth-4: 93.0%   |  Depth-5: 98.0%   |  Depth-6: 95.4%")

    print(f"\n{args.model} on Natural Language (few-shot, {n_params/1e6:.1f}M params):")
    summary = "  "
    for depth in range(args.min_depth, args.max_depth + 1):
        if depth in all_results:
            summary += f"Depth-{depth}: {all_results[depth]['accuracy']:.1%}   |  "
    print(summary.rstrip("  |  "))


if __name__ == "__main__":
    main()
