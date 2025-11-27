#!/usr/bin/env python3
"""
generate_symbolic_ontology.py

Phase 1: Symbolic Ontology Dataset Generator for Mechanistic Interpretability Research

This script generates synthetic inductive/abductive reasoning datasets using:
- Symbolic notation (c1, c2, p1, e1) instead of natural language
- Controlled complexity via depth (L) and branching factor (B)
- Parsimony traps to test Occam's Razor reasoning
- MI-ready metadata for linear probing experiments

Based on INABHYD (Sun & Saparov, 2025) ontology structure but adapted for
pure logical reasoning analysis without semantic interference.

Usage:
    python generate_symbolic_ontology.py --samples 500 --depth 3 --branch 2 --output train.jsonl
    python generate_symbolic_ontology.py --samples 500 --task inductive --depth 4 --output inductive_train.jsonl
    python generate_symbolic_ontology.py --samples 500 --task abductive --depth 3 --output abductive_train.jsonl
"""

import random
import json
import argparse
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict


@dataclass
class OntologyNode:
    """Represents a concept node in the ontology tree."""
    concept_id: int
    depth: int
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    properties: List[int] = field(default_factory=list)  # Property IDs assigned to this concept
    members: List[int] = field(default_factory=list)      # Entity IDs that are members of this concept


class SymbolicOntologyGenerator:
    """
    Generates symbolic ontology trees with controlled complexity for MI research.
    
    Key design decisions:
    1. Properties distributed throughout tree (not just root) to test hierarchical inference
    2. Parsimony traps: observations from multiple branches to force common ancestor inference
    3. Rich metadata for probing: tracks alternative hypotheses and depth of truth
    """
    
    def __init__(self, depth: int = 3, branching_factor: int = 2, 
                 num_properties: int = 5, property_assignment_prob: float = 0.4,
                 members_per_leaf: int = 2, seed: Optional[int] = None):
        """
        Args:
            depth: Maximum depth of the ontology tree (L)
            branching_factor: Number of children per non-leaf node (B)
            num_properties: Pool of available properties
            property_assignment_prob: Probability of assigning a property to each node
            members_per_leaf: Number of entity members at each leaf
            seed: Random seed for reproducibility
        """
        self.depth = depth
        self.branching_factor = branching_factor
        self.num_properties = num_properties
        self.property_assignment_prob = property_assignment_prob
        self.members_per_leaf = members_per_leaf
        
        if seed is not None:
            random.seed(seed)
        
        # Storage
        self.nodes: Dict[int, OntologyNode] = {}
        self.num_concepts = 0
        self.num_members = 0
        
        self._generate_structure()
    
    def _generate_structure(self):
        """Build the ontology tree with BFS traversal."""
        # Create root
        root = OntologyNode(concept_id=0, depth=0)
        self.nodes[0] = root
        self.num_concepts = 1
        
        # BFS to build tree
        queue = deque([0])  # Queue of concept IDs
        
        while queue:
            curr_id = queue.popleft()
            curr_node = self.nodes[curr_id]
            
            # Assign properties randomly (distributed, not just root)
            # This allows testing if model can find correct level of abstraction
            if random.random() < self.property_assignment_prob:
                # Avoid assigning same property to ancestor/descendant
                used_props = self._get_ancestor_properties(curr_id)
                available_props = [p for p in range(1, self.num_properties + 1) 
                                   if p not in used_props]
                if available_props:
                    prop_id = random.choice(available_props)
                    curr_node.properties.append(prop_id)
            
            # Add children if not at max depth
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
                # Leaf node: assign members (entities)
                for _ in range(self.members_per_leaf):
                    member_id = self.num_members
                    curr_node.members.append(member_id)
                    self.num_members += 1
    
    def _get_ancestor_properties(self, concept_id: int) -> Set[int]:
        """Get all properties from ancestors to avoid conflicts."""
        props = set()
        curr_id = concept_id
        while curr_id is not None:
            node = self.nodes[curr_id]
            props.update(node.properties)
            curr_id = node.parent_id
        return props
    
    def _get_all_descendants(self, concept_id: int) -> List[int]:
        """Get all descendant concept IDs (including self)."""
        descendants = [concept_id]
        queue = deque([concept_id])
        while queue:
            curr = queue.popleft()
            for child_id in self.nodes[curr].children_ids:
                descendants.append(child_id)
                queue.append(child_id)
        return descendants
    
    def _get_all_descendant_members(self, concept_id: int) -> List[Tuple[int, int]]:
        """
        Get all entity members that belong to a concept (including from subtypes).
        Returns list of (entity_id, immediate_leaf_concept_id) tuples.
        """
        members = []
        descendants = self._get_all_descendants(concept_id)
        for desc_id in descendants:
            for member_id in self.nodes[desc_id].members:
                members.append((member_id, desc_id))
        return members
    
    def _concept_has_property(self, concept_id: int, property_id: int) -> bool:
        """Check if concept or any ancestor has the property (inheritance)."""
        curr_id = concept_id
        while curr_id is not None:
            if property_id in self.nodes[curr_id].properties:
                return True
            curr_id = self.nodes[curr_id].parent_id
        return False
    
    def _get_depth_to_root(self, concept_id: int) -> int:
        """Calculate depth from concept to root."""
        return self.nodes[concept_id].depth
    
    def to_axioms(self, include_hidden: bool = True) -> List[str]:
        """
        Generate logical axioms from the ontology.
        
        Returns axioms in symbolic FOL format:
        - Subtype: ∀x: c{child}(x) -> c{parent}(x)
        - Property: ∀x: c{concept}(x) -> p{prop}(x)
        - Membership: c{concept}(e{entity})
        """
        axioms = []
        
        for node in self.nodes.values():
            # Subtype relations (child implies parent)
            if node.parent_id is not None:
                axioms.append(f"∀x: c{node.concept_id}(x) -> c{node.parent_id}(x)")
            
            # Property axioms
            for prop_id in node.properties:
                axioms.append(f"∀x: c{node.concept_id}(x) -> p{prop_id}(x)")
            
            # Membership axioms
            for member_id in node.members:
                axioms.append(f"c{node.concept_id}(e{member_id})")
        
        return axioms
    
    def generate_inductive_task(self) -> Optional[Dict]:
        """
        Generate an Inductive Reasoning task (Infer Property).
        
        Task: Given observations that multiple entities have a property,
        infer the most general rule (find the common ancestor).
        
        Parsimony test: We select entities from DIFFERENT branches so the
        model must identify the common ancestor, not just repeat observations.
        """
        # Find concepts with properties that have descendants with members
        candidates = []
        for concept_id, node in self.nodes.items():
            if node.properties:  # Has at least one property
                descendant_members = self._get_all_descendant_members(concept_id)
                if len(descendant_members) >= 2:  # Need multiple entities
                    for prop_id in node.properties:
                        candidates.append((concept_id, prop_id, descendant_members))
        
        if not candidates:
            return None
        
        # Select a target concept and property
        target_c, target_p, all_members = random.choice(candidates)
        target_node = self.nodes[target_c]
        
        # Select entities from different subtrees for parsimony testing
        # This forces the model to find the common ancestor
        num_observations = min(3, len(all_members))
        if len(all_members) <= num_observations:
            selected = all_members
        else:
            # Try to select from different child branches
            selected = random.sample(all_members, num_observations)
        
        # Generate observations
        observations = []
        entity_concept_map = {}  # Track which concept each entity belongs to
        
        for entity_id, leaf_id in selected:
            # Observation 1: Entity is of type (leaf concept)
            observations.append(f"c{leaf_id}(e{entity_id})")
            # Observation 2: Entity has the property
            observations.append(f"p{target_p}(e{entity_id})")
            entity_concept_map[entity_id] = leaf_id
        
        # Ground truth hypothesis (parsimonious)
        gt_hypothesis = f"∀x: c{target_c}(x) -> p{target_p}(x)"
        
        # Generate less parsimonious alternatives (distractors for probing)
        distractors = []
        
        # Alternative 1: Specific to child concepts
        for child_id in target_node.children_ids:
            if self._get_all_descendant_members(child_id):
                distractors.append(f"∀x: c{child_id}(x) -> p{target_p}(x)")
        
        # Alternative 2: Specific to leaf concepts
        for entity_id, leaf_id in selected:
            distractor = f"∀x: c{leaf_id}(x) -> p{target_p}(x)"
            if distractor not in distractors:
                distractors.append(distractor)
        
        # Build world model (without the hidden hypothesis)
        world_model = []
        for node in self.nodes.values():
            # Subtype relations
            if node.parent_id is not None:
                world_model.append(f"∀x: c{node.concept_id}(x) -> c{node.parent_id}(x)")
            
            # Only include properties that are NOT the target (hidden) property for this concept
            for prop_id in node.properties:
                if not (node.concept_id == target_c and prop_id == target_p):
                    world_model.append(f"∀x: c{node.concept_id}(x) -> p{prop_id}(x)")
            
            # Membership
            for member_id in node.members:
                world_model.append(f"c{node.concept_id}(e{member_id})")
        
        random.shuffle(world_model)
        random.shuffle(observations)
        
        # Format for training
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
                "tree_depth": self.depth,
                "branching_factor": self.branching_factor,
                "num_observations": len(selected),
                "is_parsimony_test": len(set(leaf for _, leaf in selected)) > 1,
                "valid_but_less_parsimonious": distractors[:5],  # Keep top 5
                "observed_entities": [f"e{eid}" for eid, _ in selected],
                "observed_leaf_concepts": list(set(f"c{lid}" for _, lid in selected))
            }
        }
    
    def generate_abductive_task(self) -> Optional[Dict]:
        """
        Generate an Abductive Reasoning task (Infer Membership).
        
        Task: Given that an entity has certain properties, infer which
        concept(s) it must belong to.
        """
        # Find leaf concepts with inherited properties
        leaf_concepts = [cid for cid, node in self.nodes.items() 
                        if not node.children_ids and node.members]
        
        if not leaf_concepts:
            return None
        
        target_leaf = random.choice(leaf_concepts)
        target_node = self.nodes[target_leaf]
        
        if not target_node.members:
            return None
        
        # Pick a "mystery" entity to hide
        mystery_entity = random.choice(target_node.members)
        
        # Collect all properties this entity would have (via inheritance)
        inherited_props = []
        curr_id = target_leaf
        while curr_id is not None:
            node = self.nodes[curr_id]
            inherited_props.extend([(p, curr_id) for p in node.properties])
            curr_id = node.parent_id
        
        if not inherited_props:
            return None
        
        # Sample some properties as evidence
        num_props_to_show = min(3, len(inherited_props))
        shown_props = random.sample(inherited_props, num_props_to_show)
        
        # Generate observations (entity has properties, but we hide its concept membership)
        observations = []
        for prop_id, _ in shown_props:
            observations.append(f"p{prop_id}(e{mystery_entity})")
        
        # Ground truth hypothesis
        gt_hypothesis = f"c{target_leaf}(e{mystery_entity})"
        
        # Alternative hypotheses (could also explain the observations)
        # These are ancestor concepts that have some of the shown properties
        alternatives = []
        curr_id = target_node.parent_id
        while curr_id is not None:
            alternatives.append(f"c{curr_id}(e{mystery_entity})")
            curr_id = self.nodes[curr_id].parent_id
        
        # Build world model (without the mystery entity's membership)
        world_model = []
        for node in self.nodes.values():
            if node.parent_id is not None:
                world_model.append(f"∀x: c{node.concept_id}(x) -> c{node.parent_id}(x)")
            
            for prop_id in node.properties:
                world_model.append(f"∀x: c{node.concept_id}(x) -> p{prop_id}(x)")
            
            for member_id in node.members:
                if not (node.concept_id == target_leaf and member_id == mystery_entity):
                    world_model.append(f"c{node.concept_id}(e{member_id})")
        
        random.shuffle(world_model)
        random.shuffle(observations)
        
        prompt = "[WORLD_MODEL]\n" + "\n".join(world_model)
        prompt += "\n[OBSERVATIONS]\n" + "\n".join(observations)
        prompt += "\n[TASK]\nInfer the concept membership of entity e{}.".format(mystery_entity)
        
        return {
            "input": prompt,
            "target": gt_hypothesis,
            "task_type": "abductive",
            "metadata": {
                "mystery_entity": f"e{mystery_entity}",
                "target_concept": f"c{target_leaf}",
                "depth_of_truth": target_node.depth,
                "tree_depth": self.depth,
                "branching_factor": self.branching_factor,
                "num_property_observations": len(shown_props),
                "shown_properties": [f"p{p}" for p, _ in shown_props],
                "valid_but_less_specific": alternatives[:3],
                "is_parsimony_test": len(alternatives) > 0
            }
        }
    
    def generate_subtype_task(self) -> Optional[Dict]:
        """
        Generate a Subtype Inference task (both inductive and abductive).
        
        Task: Given that entities of a subconcept all belong to a superconcept,
        infer the subtype relation.
        """
        # Find non-root concepts with parent
        candidates = [(cid, node) for cid, node in self.nodes.items() 
                      if node.parent_id is not None and self._get_all_descendant_members(cid)]
        
        if not candidates:
            return None
        
        target_c, target_node = random.choice(candidates)
        parent_c = target_node.parent_id
        
        # Get descendant members
        desc_members = self._get_all_descendant_members(target_c)
        if len(desc_members) < 2:
            return None
        
        num_obs = min(3, len(desc_members))
        selected = random.sample(desc_members, num_obs)
        
        # Observations: entities of child type are also of parent type
        observations = []
        for entity_id, leaf_id in selected:
            observations.append(f"c{leaf_id}(e{entity_id})")
            observations.append(f"c{parent_c}(e{entity_id})")
        
        # Ground truth: the subtype relation
        gt_hypothesis = f"∀x: c{target_c}(x) -> c{parent_c}(x)"
        
        # Build world model (without the target subtype relation)
        world_model = []
        for node in self.nodes.values():
            if node.parent_id is not None:
                if not (node.concept_id == target_c):
                    world_model.append(f"∀x: c{node.concept_id}(x) -> c{node.parent_id}(x)")
            
            for prop_id in node.properties:
                world_model.append(f"∀x: c{node.concept_id}(x) -> p{prop_id}(x)")
            
            for member_id in node.members:
                world_model.append(f"c{node.concept_id}(e{member_id})")
        
        random.shuffle(world_model)
        random.shuffle(observations)
        
        prompt = "[WORLD_MODEL]\n" + "\n".join(world_model)
        prompt += "\n[OBSERVATIONS]\n" + "\n".join(observations)
        prompt += "\n[TASK]\nInfer the subtype relation."
        
        return {
            "input": prompt,
            "target": gt_hypothesis,
            "task_type": "subtype",
            "metadata": {
                "child_concept": f"c{target_c}",
                "parent_concept": f"c{parent_c}",
                "depth_of_truth": target_node.depth,
                "tree_depth": self.depth,
                "branching_factor": self.branching_factor,
                "num_observations": len(selected),
                "observed_entities": [f"e{eid}" for eid, _ in selected]
            }
        }


def generate_dataset(num_samples: int, 
                     task_type: str = "mixed",
                     depth_range: Tuple[int, int] = (3, 4),
                     branch_range: Tuple[int, int] = (2, 3),
                     num_properties: int = 5,
                     seed: Optional[int] = None) -> List[Dict]:
    """
    Generate a dataset of reasoning examples.
    
    Args:
        num_samples: Number of examples to generate
        task_type: "inductive", "abductive", "subtype", or "mixed"
        depth_range: (min_depth, max_depth) for ontology trees
        branch_range: (min_branch, max_branch) for branching factor
        num_properties: Number of available properties in pool
        seed: Random seed
    """
    if seed is not None:
        random.seed(seed)
    
    dataset = []
    attempts = 0
    max_attempts = num_samples * 10
    
    task_generators = {
        "inductive": lambda gen: gen.generate_inductive_task(),
        "abductive": lambda gen: gen.generate_abductive_task(),
        "subtype": lambda gen: gen.generate_subtype_task()
    }
    
    while len(dataset) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Vary complexity
        depth = random.randint(depth_range[0], depth_range[1])
        branch = random.randint(branch_range[0], branch_range[1])
        
        gen = SymbolicOntologyGenerator(
            depth=depth,
            branching_factor=branch,
            num_properties=num_properties
        )
        
        # Select task type
        if task_type == "mixed":
            selected_task = random.choice(["inductive", "abductive", "subtype"])
        else:
            selected_task = task_type
        
        generator_func = task_generators.get(selected_task)
        if generator_func:
            sample = generator_func(gen)
            if sample:
                sample["id"] = len(dataset)
                dataset.append(sample)
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate symbolic ontology datasets for MI research on non-deductive reasoning."
    )
    parser.add_argument("--samples", "-n", type=int, default=500,
                        help="Number of samples to generate")
    parser.add_argument("--task", "-t", type=str, default="mixed",
                        choices=["inductive", "abductive", "subtype", "mixed"],
                        help="Task type to generate")
    parser.add_argument("--min-depth", type=int, default=3,
                        help="Minimum ontology tree depth")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="Maximum ontology tree depth")
    parser.add_argument("--min-branch", type=int, default=2,
                        help="Minimum branching factor")
    parser.add_argument("--max-branch", type=int, default=3,
                        help="Maximum branching factor")
    parser.add_argument("--num-properties", type=int, default=5,
                        help="Number of available properties")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", "-o", type=str, default="symbolic_ontology_train.jsonl",
                        help="Output file path")
    parser.add_argument("--preview", action="store_true",
                        help="Print a sample entry before saving")
    
    args = parser.parse_args()
    
    print(f"Generating {args.samples} {args.task} reasoning examples...")
    print(f"  Depth range: {args.min_depth}-{args.max_depth}")
    print(f"  Branch range: {args.min_branch}-{args.max_branch}")
    print(f"  Properties: {args.num_properties}")
    print(f"  Seed: {args.seed}")
    
    dataset = generate_dataset(
        num_samples=args.samples,
        task_type=args.task,
        depth_range=(args.min_depth, args.max_depth),
        branch_range=(args.min_branch, args.max_branch),
        num_properties=args.num_properties,
        seed=args.seed
    )
    
    # Save dataset
    with open(args.output, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    
    print(f"\nGenerated {len(dataset)} samples. Saved to {args.output}")
    
    # Task type distribution
    task_counts = defaultdict(int)
    for entry in dataset:
        task_counts[entry["task_type"]] += 1
    print("\nTask distribution:")
    for task, count in task_counts.items():
        print(f"  {task}: {count}")
    
    # Parsimony test distribution
    parsimony_count = sum(1 for e in dataset if e.get("metadata", {}).get("is_parsimony_test", False))
    print(f"\nParsimony tests: {parsimony_count}/{len(dataset)} ({100*parsimony_count/len(dataset):.1f}%)")
    
    # Preview
    if args.preview and dataset:
        print("\n" + "="*60)
        print("SAMPLE ENTRY")
        print("="*60)
        sample = dataset[0]
        print("\n[INPUT]")
        print(sample["input"])
        print("\n[TARGET]")
        print(sample["target"])
        print("\n[METADATA]")
        for k, v in sample["metadata"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
