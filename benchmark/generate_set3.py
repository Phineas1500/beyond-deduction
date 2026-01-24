#!/usr/bin/env python3
"""
Generate Set 3: Occam's Razor Test

Set 3 tests whether models apply Occam's razor correctly by using:
- SAME ontology for H1 and H2
- DIFFERENT observations pointing to different levels

Structure:
- Ontology includes properties at both child and parent levels
- H1: Observe child property → child-level answer is most parsimonious
- H2: Observe parent property → parent-level answer is most parsimonious

If the model reasons correctly:
- H1 accuracy should be HIGH (output child when child property observed)
- H2 accuracy should be HIGH (output parent when parent property observed)
- H1 + H2 could be >> 100%

If the model has a fixed bias:
- H1 + H2 ≈ 100% (same output regardless of which is correct)
"""

import random
import numpy as np
from random import shuffle
from morphology import Morphology
import pickle
import argparse


def generate_set3_pair(seed: int):
    """
    Generate a Set 3 pair with SAME ontology, DIFFERENT observations.

    Properties are pre-assigned to levels:
    - Child concept has property A
    - Parent concept has property B

    H1: Observe property A → child-level answer is most parsimonious
    H2: Observe property B → parent-level answer is most parsimonious

    Returns:
        tuple: (h1_example, h2_example)
    """
    random.seed(seed)
    np.random.seed(seed)

    morph = Morphology()

    # Get concepts
    child_concept = morph.next_concept
    parent_concept = morph.next_concept

    # Get TWO different properties (from different families)
    prop_child = morph.next_property([])
    prop_parent = morph.next_property([prop_child.family])

    # Get the property names and handle negation
    child_prop_name = prop_child.name
    parent_prop_name = prop_parent.name

    if prop_child.is_negated:
        child_prop_text = f"not {child_prop_name}"
    else:
        child_prop_text = child_prop_name

    if prop_parent.is_negated:
        parent_prop_text = f"not {parent_prop_name}"
    else:
        parent_prop_text = parent_prop_name

    # Get entities (same entities for both H1 and H2)
    entities = [morph.next_entity for _ in range(3)]

    # === SHARED ONTOLOGY ===
    # Includes:
    # - Entity memberships: "Amy is a [child]"
    # - Subsumption: "Every [child] is a [parent]"
    # - Child property rule: "[Child]s are [property A]"
    # - Parent property rule: "[Parent]s are [property B]"

    membership_statements = [f"{e} is a {child_concept}" for e in entities]
    ontology_statement = f"Every {child_concept} is a {parent_concept}"
    child_property_rule = f"{child_concept.capitalize()}s are {child_prop_text}"
    parent_property_rule = f"{parent_concept.capitalize()}s are {parent_prop_text}"

    # Shuffle the theory statements
    theory_parts = membership_statements + [ontology_statement, child_property_rule, parent_property_rule]
    shuffle(theory_parts)
    theories_nl = ". ".join(theory_parts) + "."

    # === H1: Observe CHILD property ===
    # When we observe the child property, the most parsimonious explanation
    # is that the entities are members of the child concept.
    h1_observations = [f"{e} is {child_prop_text}" for e in entities]
    shuffle(h1_observations)
    h1_observations_nl = ". ".join(h1_observations) + "."
    h1_gt = f"They are {child_concept}s"

    # === H2: Observe PARENT property ===
    # When we observe the parent property, the most parsimonious explanation
    # is that the entities are members of the parent concept.
    # Saying they are [child]s would over-specify.
    h2_observations = [f"{e} is {parent_prop_text}" for e in entities]
    shuffle(h2_observations)
    h2_observations_nl = ". ".join(h2_observations) + "."
    h2_gt = f"They are {parent_concept}s"

    h1_example = {
        'theories_nl': theories_nl,
        'observations_nl': h1_observations_nl,
        'gt_hypothesis_nl': h1_gt,
        'child_concept': child_concept,
        'parent_concept': parent_concept,
        'child_property': child_prop_text,
        'parent_property': parent_prop_text,
        'entities': entities,
        'depth': 1,
        'seed': seed,
        'set_type': 'occam',
    }

    h2_example = {
        'theories_nl': theories_nl,  # SAME ontology
        'observations_nl': h2_observations_nl,  # DIFFERENT observation
        'gt_hypothesis_nl': h2_gt,
        'child_concept': child_concept,
        'parent_concept': parent_concept,
        'child_property': child_prop_text,
        'parent_property': parent_prop_text,
        'entities': entities,
        'depth': 2,
        'seed': seed,
        'set_type': 'occam',
    }

    return h1_example, h2_example


def generate_set3(n_pairs: int = 100, base_seed: int = 42):
    """Generate n Set 3 pairs."""
    pairs = []
    for i in range(n_pairs):
        seed = base_seed + i
        h1, h2 = generate_set3_pair(seed)
        pairs.append((h1, h2))
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Generate Set 3 (Occam\'s Razor) pairs')
    parser.add_argument('-n', '--n-pairs', type=int, default=100,
                        help='Number of pairs to generate (default: 100)')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='Base seed for reproducibility (default: 42)')
    parser.add_argument('--output', type=str, default='matched_pairs_set3_occam.pkl',
                        help='Output pickle file (default: matched_pairs_set3_occam.pkl)')
    parser.add_argument('--preview', type=int, default=2,
                        help='Number of examples to preview (default: 2)')
    args = parser.parse_args()

    print(f"Generating {args.n_pairs} Set 3 pairs...")
    pairs = generate_set3(args.n_pairs, args.base_seed)

    # Preview
    if args.preview > 0:
        print("\n" + "=" * 70)
        print("PREVIEW")
        print("=" * 70)

        for i in range(min(args.preview, len(pairs))):
            h1, h2 = pairs[i]
            print(f"\n--- PAIR {i} ---")
            print(f"\nShared Ontology:")
            print(f"  {h1['theories_nl']}")
            print(f"\nH1 (child-level correct):")
            print(f"  Observations: {h1['observations_nl']}")
            print(f"  Ground truth: {h1['gt_hypothesis_nl']}")
            print(f"\nH2 (parent-level correct):")
            print(f"  Observations: {h2['observations_nl']}")
            print(f"  Ground truth: {h2['gt_hypothesis_nl']}")

    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(pairs, f)

    print(f"\nSaved {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
