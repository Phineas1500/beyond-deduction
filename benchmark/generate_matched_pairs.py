#!/usr/bin/env python3
"""
Generate matched minimal pairs for MI analysis.

Each pair has:
- H1 version: 1-hop inference (child_concept → property)
- H2 version: 2-hop inference (root_concept → property)

Both versions share:
- Same entities (e.g., Amy, Bob, Carol)
- Same observations (e.g., "Amy is rainy")
- Same visible tree structure (child_concept → root_concept)

Only the hidden rule differs, isolating inference depth as the sole variable.
"""

import random
import numpy as np
from random import shuffle
from morphology import Morphology, Prop
from fol import FOL, FOL_Entity, FOL_Concept, FOL_Property
import pickle
import argparse


def generate_matched_pair(seed: int, include_negated: bool = True):
    """
    Generate a matched H1/H2 pair.

    Returns:
        tuple: (h1_example, h2_example) where each is a dict with:
            - theories_nl: Natural language theories
            - observations_nl: Natural language observations
            - gt_hypothesis_nl: Natural language ground truth
            - theories_fol: FOL format theories
            - observations_fol: FOL format observations
            - gt_hypothesis_fol: FOL format ground truth
            - depth: 1 or 2
            - seed: The seed used
            - entities: List of entity names
            - child_concept: The child concept name
            - root_concept: The root concept name
            - property: The property (Prop object)
    """
    # Seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Create morphology with seeded random state
    morph = Morphology()

    # Get names - order matters for reproducibility
    child_concept = morph.next_concept  # e.g., "dalpist"
    root_concept = morph.next_concept   # e.g., "rompus"

    # Get property - morphology may give us positive or negated
    # (it shuffles both positive and negated versions together)
    prop = morph.next_property([])  # e.g., Prop("weather", "rainy", negated=False or True)

    # If user doesn't want negated and we got one, get another
    # (This is a simple approach - just skip negated properties)
    if not include_negated and prop.is_negated:
        # Try to get a non-negated property
        for _ in range(10):  # Try up to 10 times
            prop = morph.next_property([prop.family])  # Exclude current family
            if not prop.is_negated:
                break

    # Get 3 entities
    entities = [morph.next_entity for _ in range(3)]  # e.g., ["Amy", "Bob", "Carol"]

    # Build FOL objects for reuse
    fol_entities = [FOL_Entity(e) for e in entities]
    fol_child = FOL_Concept(child_concept)
    fol_root = FOL_Concept(root_concept)
    fol_prop = FOL_Property(prop)

    # === SHARED COMPONENTS ===

    # Membership statements: "Amy is a dalpist" (always visible)
    membership_fols = [FOL(fe, fol_child) for fe in fol_entities]

    # Ontology statement: "All dalpists are rompuses" (always visible)
    ontology_fol = FOL(fol_child, fol_root)

    # Observations: "Amy is rainy" or "Amy is not rainy"
    observation_fols = [FOL(fe, fol_prop) for fe in fol_entities]

    # === H1 VERSION: Hide child_concept → property ===
    # Hidden rule: "All dalpists are rainy"
    # Model must infer: dalpist → rainy (1 hop from membership)

    h1_hidden_fol = FOL(fol_child, fol_prop)

    # H1 theories = memberships + ontology (but NOT the property rule)
    h1_theory_fols = membership_fols + [ontology_fol]

    # === H2 VERSION: Hide root_concept → property ===
    # Hidden rule: "All rompuses are rainy"
    # Model must infer: rompus → rainy (2 hops: membership → ontology → property)

    h2_hidden_fol = FOL(fol_root, fol_prop)

    # H2 theories = same as H1 (memberships + ontology)
    h2_theory_fols = membership_fols + [ontology_fol]

    # === FORMAT OUTPUTS ===
    # IMPORTANT: For matched pairs, we need IDENTICAL token sequences
    # So we shuffle ONCE and use the same order for both H1 and H2

    def format_nl(fol_list):
        """Convert list of FOL objects to NL string (no shuffle - order preserved)."""
        return ". ".join([str(f).capitalize() for f in fol_list]) + "."

    def format_fol(fol_list):
        """Convert list of FOL objects to FOL string (no shuffle - order preserved)."""
        return ". ".join([f.to_fol() for f in fol_list]) + "."

    # Shuffle theories and observations ONCE, then use same order for both
    # This ensures token alignment between H1 and H2
    shared_theory_fols = list(h1_theory_fols)  # H1 and H2 have same theories
    shuffle(shared_theory_fols)

    shared_observation_fols = list(observation_fols)
    shuffle(shared_observation_fols)

    # Now format with the shared order
    theories_nl = format_nl(shared_theory_fols)
    theories_fol = format_fol(shared_theory_fols)

    observations_nl = format_nl(shared_observation_fols)
    observations_fol = format_fol(shared_observation_fols)

    h1_example = {
        'theories_nl': theories_nl,  # SAME as H2
        'observations_nl': observations_nl,  # SAME as H2
        'gt_hypothesis_nl': str(h1_hidden_fol).capitalize(),
        'theories_fol': theories_fol,  # SAME as H2
        'observations_fol': observations_fol,  # SAME as H2
        'gt_hypothesis_fol': h1_hidden_fol.to_fol(),
        'depth': 1,
        'seed': seed,
        'entities': entities,
        'child_concept': child_concept,
        'root_concept': root_concept,
        'property': prop,
        'is_negated': prop.is_negated,  # Track actual negation status
    }

    h2_example = {
        'theories_nl': theories_nl,  # SAME as H1
        'observations_nl': observations_nl,  # SAME as H1
        'gt_hypothesis_nl': str(h2_hidden_fol).capitalize(),
        'theories_fol': theories_fol,  # SAME as H1
        'observations_fol': observations_fol,  # SAME as H1
        'gt_hypothesis_fol': h2_hidden_fol.to_fol(),
        'depth': 2,
        'seed': seed,
        'entities': entities,
        'child_concept': child_concept,
        'root_concept': root_concept,
        'property': prop,
        'is_negated': prop.is_negated,  # Track actual negation status
    }

    return h1_example, h2_example


def generate_matched_pairs(n_pairs: int = 50, base_seed: int = 42, include_negated: bool = True):
    """
    Generate n matched H1/H2 pairs.

    Args:
        n_pairs: Number of pairs to generate
        base_seed: Starting seed (each pair uses base_seed + i)
        include_negated: Whether to include negated property cases (~50%)

    Returns:
        list: List of (h1_example, h2_example) tuples
    """
    pairs = []
    for i in range(n_pairs):
        seed = base_seed + i
        h1, h2 = generate_matched_pair(seed, include_negated)
        pairs.append((h1, h2))

    return pairs


def generate_salience_pair(seed: int, include_negated: bool = True):
    """
    Generate a Set 2 (Salience Test) pair.

    Same structure as Set 1, but adds 3 "distractor" entities that are
    direct members of the parent concept. These distractors do NOT appear
    in observations, making them logically irrelevant.

    This tests whether frequency/salience alone recovers performance.
    Parent concept appears ~4x instead of 1x.

    Example:
        Theories: "Barbara is a lerpant. Each lerpant is a timple. Pamela is a lerpant.
                   Carol is a lerpant. Dave is a timple. Eric is a timple. Frank is a timple."
        Observations: "Barbara is not salty. Carol is not salty. Pamela is not salty."
        GT: "Timples are not salty"
    """
    # Seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    morph = Morphology()

    # Get concepts
    child_concept = morph.next_concept
    root_concept = morph.next_concept

    # Get property
    prop = morph.next_property([])
    if not include_negated and prop.is_negated:
        for _ in range(10):
            prop = morph.next_property([prop.family])
            if not prop.is_negated:
                break

    # Get 3 main entities (will appear in observations)
    main_entities = [morph.next_entity for _ in range(3)]

    # Get 3 distractor entities (direct members of parent, NOT in observations)
    distractor_entities = [morph.next_entity for _ in range(3)]

    # Build FOL objects
    fol_main = [FOL_Entity(e) for e in main_entities]
    fol_distractor = [FOL_Entity(e) for e in distractor_entities]
    fol_child = FOL_Concept(child_concept)
    fol_root = FOL_Concept(root_concept)
    fol_prop = FOL_Property(prop)

    # Main entity memberships (child concept)
    membership_fols = [FOL(fe, fol_child) for fe in fol_main]

    # Distractor memberships (DIRECT parent concept members)
    distractor_fols = [FOL(fe, fol_root) for fe in fol_distractor]

    # Ontology: child → parent
    ontology_fol = FOL(fol_child, fol_root)

    # Observations: only main entities (distractors are irrelevant)
    observation_fols = [FOL(fe, fol_prop) for fe in fol_main]

    # H2 hidden rule: parent → property
    h2_hidden_fol = FOL(fol_root, fol_prop)

    # Combine theories: memberships + ontology + distractors
    theory_fols = membership_fols + [ontology_fol] + distractor_fols

    # Format outputs
    def format_nl(fol_list):
        return ". ".join([str(f).capitalize() for f in fol_list]) + "."

    def format_fol(fol_list):
        return ". ".join([f.to_fol() for f in fol_list]) + "."

    # Shuffle once
    shuffle(theory_fols)
    shuffle(observation_fols)

    theories_nl = format_nl(theory_fols)
    theories_fol = format_fol(theory_fols)
    observations_nl = format_nl(observation_fols)
    observations_fol = format_fol(observation_fols)

    # Count parent mentions for verification
    parent_mentions = theories_nl.lower().count(root_concept.lower())

    example = {
        'theories_nl': theories_nl,
        'observations_nl': observations_nl,
        'gt_hypothesis_nl': str(h2_hidden_fol).capitalize(),
        'theories_fol': theories_fol,
        'observations_fol': observations_fol,
        'gt_hypothesis_fol': h2_hidden_fol.to_fol(),
        'depth': 2,
        'seed': seed,
        'main_entities': main_entities,
        'distractor_entities': distractor_entities,
        'child_concept': child_concept,
        'root_concept': root_concept,
        'property': prop,
        'is_negated': prop.is_negated,
        'set_type': 'salience',
        'parent_mentions': parent_mentions,
    }

    return example


def generate_evidential_path_pair(seed: int, include_negated: bool = True):
    """
    Generate a Set 5 (Evidential Path) example.

    Same structure as Set 2 (Salience), but the distractor entities that are
    direct members of the parent concept DO appear in observations with the property.

    This provides a 1-hop shortcut:
        "Dave is a timple + Dave is not salty → Timples are not salty"

    The key test: Does the model use this shortcut even when the 2-hop path
    (Barbara → lerpant → timple) is also available?

    Example:
        Theories: "Barbara is a lerpant. Each lerpant is a timple. Pamela is a lerpant.
                   Carol is a lerpant. Dave is a timple. Eric is a timple. Frank is a timple."
        Observations: "Barbara is not salty. Carol is not salty. Pamela is not salty.
                       Dave is not salty. Eric is not salty. Frank is not salty."
        GT: "Timples are not salty"

    Prediction: Set 5 ≈ Set 4 (~44%) because direct evidential path is available.
    """
    # Seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    morph = Morphology()

    # Get concepts
    child_concept = morph.next_concept
    root_concept = morph.next_concept

    # Get property
    prop = morph.next_property([])
    if not include_negated and prop.is_negated:
        for _ in range(10):
            prop = morph.next_property([prop.family])
            if not prop.is_negated:
                break

    # Get 3 main entities (go through child concept - the "hard" path)
    main_entities = [morph.next_entity for _ in range(3)]

    # Get 3 distractor entities (direct members of parent - the "easy" path)
    distractor_entities = [morph.next_entity for _ in range(3)]

    # Build FOL objects
    fol_main = [FOL_Entity(e) for e in main_entities]
    fol_distractor = [FOL_Entity(e) for e in distractor_entities]
    fol_child = FOL_Concept(child_concept)
    fol_root = FOL_Concept(root_concept)
    fol_prop = FOL_Property(prop)

    # Main entity memberships (child concept)
    membership_fols = [FOL(fe, fol_child) for fe in fol_main]

    # Distractor memberships (DIRECT parent concept members)
    distractor_membership_fols = [FOL(fe, fol_root) for fe in fol_distractor]

    # Ontology: child → parent
    ontology_fol = FOL(fol_child, fol_root)

    # KEY DIFFERENCE FROM SET 2:
    # Observations include BOTH main entities AND distractors with the property
    # This provides the direct evidential path
    all_entities = fol_main + fol_distractor
    observation_fols = [FOL(fe, fol_prop) for fe in all_entities]

    # H2 hidden rule: parent → property
    h2_hidden_fol = FOL(fol_root, fol_prop)

    # Combine theories: memberships + ontology + distractor memberships
    theory_fols = membership_fols + [ontology_fol] + distractor_membership_fols

    # Format outputs
    def format_nl(fol_list):
        return ". ".join([str(f).capitalize() for f in fol_list]) + "."

    def format_fol(fol_list):
        return ". ".join([f.to_fol() for f in fol_list]) + "."

    # Shuffle once
    shuffle(theory_fols)
    shuffle(observation_fols)

    theories_nl = format_nl(theory_fols)
    theories_fol = format_fol(theory_fols)
    observations_nl = format_nl(observation_fols)
    observations_fol = format_fol(observation_fols)

    # Count parent mentions for verification
    parent_mentions = theories_nl.lower().count(root_concept.lower())

    # Track which entities are in observations for analysis
    entities_in_observations = main_entities + distractor_entities

    example = {
        'theories_nl': theories_nl,
        'observations_nl': observations_nl,
        'gt_hypothesis_nl': str(h2_hidden_fol).capitalize(),
        'theories_fol': theories_fol,
        'observations_fol': observations_fol,
        'gt_hypothesis_fol': h2_hidden_fol.to_fol(),
        'depth': 2,
        'seed': seed,
        'main_entities': main_entities,
        'distractor_entities': distractor_entities,
        'entities_in_observations': entities_in_observations,
        'child_concept': child_concept,
        'root_concept': root_concept,
        'property': prop,
        'is_negated': prop.is_negated,
        'set_type': 'evidential_path',
        'parent_mentions': parent_mentions,
        'has_direct_evidential_path': True,  # Distractors are in observations
    }

    return example


def generate_inabhyd_style_pair(seed: int, include_negated: bool = True):
    """
    Generate a Set 4 (INABHYD-style) pair.

    Multiple child concepts converging to same parent (3 different children).
    One entity is a direct member of the parent.
    Parent concept appears multiple times (~4x).

    This should approximate INABHYD H2's ~51% accuracy.

    Example:
        Theories: "Barbara is a lerpant. Each lerpant is a timple. Amy is a pergit.
                   Each pergit is a timple. Carol is a zumpus. Each zumpus is a timple.
                   Jerry is a timple."
        Observations: "Barbara is not salty. Carol is not salty. Amy is not salty. Jerry is not salty."
        GT: "Timples are not salty"
    """
    # Seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    morph = Morphology()

    # Get parent concept (the one to be inferred)
    root_concept = morph.next_concept

    # Get 3 different child concepts
    child_concepts = [morph.next_concept for _ in range(3)]

    # Get property
    prop = morph.next_property([])
    if not include_negated and prop.is_negated:
        for _ in range(10):
            prop = morph.next_property([prop.family])
            if not prop.is_negated:
                break

    # Get entities: one per child concept + one direct parent member
    child_entities = [morph.next_entity for _ in range(3)]  # One per child
    direct_parent_entity = morph.next_entity  # Direct member of parent

    # Build FOL objects
    fol_root = FOL_Concept(root_concept)
    fol_children = [FOL_Concept(c) for c in child_concepts]
    fol_child_entities = [FOL_Entity(e) for e in child_entities]
    fol_direct = FOL_Entity(direct_parent_entity)
    fol_prop = FOL_Property(prop)

    # Build theories
    theory_fols = []

    # Entity memberships in child concepts
    for fe, fc in zip(fol_child_entities, fol_children):
        theory_fols.append(FOL(fe, fc))

    # Child → Parent relationships (3 different paths)
    for fc in fol_children:
        theory_fols.append(FOL(fc, fol_root))

    # Direct parent member
    theory_fols.append(FOL(fol_direct, fol_root))

    # Observations: all 4 entities have the property
    all_entities = fol_child_entities + [fol_direct]
    observation_fols = [FOL(fe, fol_prop) for fe in all_entities]

    # H2 hidden rule: parent → property
    h2_hidden_fol = FOL(fol_root, fol_prop)

    # Format outputs
    def format_nl(fol_list):
        return ". ".join([str(f).capitalize() for f in fol_list]) + "."

    def format_fol(fol_list):
        return ". ".join([f.to_fol() for f in fol_list]) + "."

    # Shuffle once
    shuffle(theory_fols)
    shuffle(observation_fols)

    theories_nl = format_nl(theory_fols)
    theories_fol = format_fol(theory_fols)
    observations_nl = format_nl(observation_fols)
    observations_fol = format_fol(observation_fols)

    # Count parent mentions for verification
    parent_mentions = theories_nl.lower().count(root_concept.lower())

    example = {
        'theories_nl': theories_nl,
        'observations_nl': observations_nl,
        'gt_hypothesis_nl': str(h2_hidden_fol).capitalize(),
        'theories_fol': theories_fol,
        'observations_fol': observations_fol,
        'gt_hypothesis_fol': h2_hidden_fol.to_fol(),
        'depth': 2,
        'seed': seed,
        'child_entities': child_entities,
        'direct_parent_entity': direct_parent_entity,
        'child_concepts': child_concepts,
        'root_concept': root_concept,
        'property': prop,
        'is_negated': prop.is_negated,
        'set_type': 'inabhyd_style',
        'parent_mentions': parent_mentions,
    }

    return example


def generate_conflict_example(seed: int, correct_is_negated: bool = True):
    """
    Generate a Set 6 (Conflict Test) example.

    Creates a conflict between 2-hop logical reasoning and 1-hop shortcut:
    - Child entity → child concept → parent concept, child entity has property P
      → Logically correct answer: parent concept has P
    - Bridge entity is direct parent member, bridge entity has OPPOSITE of P
      → Shortcut answer: parent concept has opposite of P

    If model uses conjunction detection (shortcut), it will output the WRONG answer.
    If model uses compositional reasoning, it will output the CORRECT answer.

    Args:
        seed: Random seed for reproducibility
        correct_is_negated: If True, correct answer is negated property (e.g., "not salty")
                           If False, correct answer is positive property (e.g., "salty")

    Example (correct_is_negated=True):
        Theories: "Barbara is a lerpant. Each lerpant is a timple. Dave is a timple."
        Observations: "Barbara is not salty. Dave is salty."
        Ground truth: "Timples are not salty" (via 2-hop: Barbara → lerpant → timple)
        Shortcut answer: "Timples are salty" (via 1-hop: Dave is timple ∧ Dave is salty)
    """
    random.seed(seed)
    np.random.seed(seed)

    morph = Morphology()

    # Get concepts
    child_concept = morph.next_concept  # e.g., "lerpant"
    parent_concept = morph.next_concept  # e.g., "timple"

    # Get a base property from a family (we'll create both versions manually)
    base_prop = morph.next_property([])

    # Create the correct property (what 2-hop reasoning should derive)
    correct_prop = Prop(base_prop.family, base_prop.name, negated=correct_is_negated)

    # Create the shortcut property (opposite of correct - what conjunction detection gives)
    shortcut_prop = Prop(base_prop.family, base_prop.name, negated=not correct_is_negated)

    # Get entities
    child_entity = morph.next_entity  # Goes through child concept (2-hop path)
    bridge_entity = morph.next_entity  # Direct parent member (1-hop shortcut)

    # Build FOL objects
    fol_child_entity = FOL_Entity(child_entity)
    fol_bridge_entity = FOL_Entity(bridge_entity)
    fol_child_concept = FOL_Concept(child_concept)
    fol_parent_concept = FOL_Concept(parent_concept)
    fol_correct_prop = FOL_Property(correct_prop)
    fol_shortcut_prop = FOL_Property(shortcut_prop)

    # === THEORIES ===
    # Child entity membership: "Barbara is a lerpant"
    child_membership_fol = FOL(fol_child_entity, fol_child_concept)

    # Ontology: "Each lerpant is a timple"
    ontology_fol = FOL(fol_child_concept, fol_parent_concept)

    # Bridge entity direct membership: "Dave is a timple"
    bridge_membership_fol = FOL(fol_bridge_entity, fol_parent_concept)

    theory_fols = [child_membership_fol, ontology_fol, bridge_membership_fol]

    # === OBSERVATIONS ===
    # Child entity has the CORRECT property: "Barbara is not salty"
    child_observation_fol = FOL(fol_child_entity, fol_correct_prop)

    # Bridge entity has the SHORTCUT (opposite) property: "Dave is salty"
    bridge_observation_fol = FOL(fol_bridge_entity, fol_shortcut_prop)

    observation_fols = [child_observation_fol, bridge_observation_fol]

    # === ANSWERS ===
    # Ground truth (via 2-hop reasoning): "Timples are not salty"
    ground_truth_fol = FOL(fol_parent_concept, fol_correct_prop)

    # Shortcut answer (via conjunction detection): "Timples are salty"
    shortcut_answer_fol = FOL(fol_parent_concept, fol_shortcut_prop)

    # Format outputs
    def format_nl(fol_list):
        return ". ".join([str(f).capitalize() for f in fol_list]) + "."

    def format_fol(fol_list):
        return ". ".join([f.to_fol() for f in fol_list]) + "."

    # Shuffle for natural presentation
    shuffle(theory_fols)
    shuffle(observation_fols)

    theories_nl = format_nl(theory_fols)
    theories_fol = format_fol(theory_fols)
    observations_nl = format_nl(observation_fols)
    observations_fol = format_fol(observation_fols)

    example = {
        # Input
        'theories_nl': theories_nl,
        'observations_nl': observations_nl,
        'theories_fol': theories_fol,
        'observations_fol': observations_fol,

        # Answers for evaluation
        'ground_truth_nl': str(ground_truth_fol).capitalize(),
        'ground_truth_fol': ground_truth_fol.to_fol(),
        'shortcut_answer_nl': str(shortcut_answer_fol).capitalize(),
        'shortcut_answer_fol': shortcut_answer_fol.to_fol(),

        # For backwards compatibility with existing evaluation code
        'gt_hypothesis_nl': str(ground_truth_fol).capitalize(),
        'gt_hypothesis_fol': ground_truth_fol.to_fol(),

        # Metadata for analysis
        'seed': seed,
        'child_entity': child_entity,
        'bridge_entity': bridge_entity,
        'child_concept': child_concept,
        'parent_concept': parent_concept,
        'correct_property': correct_prop,
        'shortcut_property': shortcut_prop,
        'correct_is_negated': correct_is_negated,
        'set_type': 'conflict',

        # Helper fields for three-way classification
        'correct_property_name': correct_prop.name if not correct_prop.is_negated else f"not {correct_prop.name}",
        'shortcut_property_name': shortcut_prop.name if not shortcut_prop.is_negated else f"not {shortcut_prop.name}",
        'parent_concept_lower': parent_concept.lower(),
    }

    return example


def generate_conflict_set(n_examples: int = 50, base_seed: int = 42):
    """
    Generate Set 6 (Conflict Test) examples.

    Balances 50/50 between:
    - Correct answer is negated property (correct_is_negated=True)
    - Correct answer is positive property (correct_is_negated=False)

    Args:
        n_examples: Number of examples to generate
        base_seed: Starting seed

    Returns:
        list: List of conflict test examples
    """
    examples = []
    for i in range(n_examples):
        seed = base_seed + i
        # Alternate: even indices get negated correct, odd get positive correct
        correct_is_negated = (i % 2 == 0)
        example = generate_conflict_example(seed, correct_is_negated)
        examples.append(example)
    return examples


def generate_salience_set(n_examples: int = 50, base_seed: int = 42, include_negated: bool = True):
    """Generate Set 2 (Salience Test) examples."""
    examples = []
    for i in range(n_examples):
        seed = base_seed + i
        example = generate_salience_pair(seed, include_negated)
        examples.append(example)
    return examples


def generate_inabhyd_style_set(n_examples: int = 50, base_seed: int = 42, include_negated: bool = True):
    """Generate Set 4 (INABHYD-style) examples."""
    examples = []
    for i in range(n_examples):
        seed = base_seed + i
        example = generate_inabhyd_style_pair(seed, include_negated)
        examples.append(example)
    return examples


def generate_evidential_path_set(n_examples: int = 50, base_seed: int = 42, include_negated: bool = True):
    """Generate Set 5 (Evidential Path) examples."""
    examples = []
    for i in range(n_examples):
        seed = base_seed + i
        example = generate_evidential_path_pair(seed, include_negated)
        examples.append(example)
    return examples


def print_pair_example(h1, h2):
    """Pretty print a matched pair for inspection."""
    print("=" * 70)
    print(f"MATCHED PAIR (seed={h1['seed']}, negated={h1['is_negated']})")
    print("=" * 70)

    print(f"\nShared components:")
    print(f"  Entities: {h1['entities']}")
    print(f"  Child concept: {h1['child_concept']}")
    print(f"  Root concept: {h1['root_concept']}")
    print(f"  Property: {h1['property']}")

    print(f"\n--- H1 (depth=1) ---")
    print(f"Theories (NL): {h1['theories_nl']}")
    print(f"Observations (NL): {h1['observations_nl']}")
    print(f"GT Hypothesis (NL): {h1['gt_hypothesis_nl']}")
    print(f"GT Hypothesis (FOL): {h1['gt_hypothesis_fol']}")

    print(f"\n--- H2 (depth=2) ---")
    print(f"Theories (NL): {h2['theories_nl']}")
    print(f"Observations (NL): {h2['observations_nl']}")
    print(f"GT Hypothesis (NL): {h2['gt_hypothesis_nl']}")
    print(f"GT Hypothesis (FOL): {h2['gt_hypothesis_fol']}")

    print()


def print_set_example(example, set_name):
    """Pretty print a single set example."""
    print("=" * 70)
    print(f"{set_name} (seed={example['seed']}, negated={example.get('is_negated', 'N/A')})")
    print("=" * 70)

    print(f"\nTheories (NL): {example['theories_nl']}")
    print(f"Observations (NL): {example['observations_nl']}")
    print(f"GT Hypothesis (NL): {example['gt_hypothesis_nl']}")
    if 'parent_mentions' in example:
        print(f"Parent mentions: {example['parent_mentions']}")
    print()


def print_conflict_example(example):
    """Pretty print a Set 6 (Conflict Test) example."""
    print("=" * 70)
    print(f"Set 6 CONFLICT TEST (seed={example['seed']})")
    print(f"  Correct answer uses: {'negated' if example['correct_is_negated'] else 'positive'} property")
    print("=" * 70)

    print(f"\nStructure:")
    print(f"  Child entity: {example['child_entity']} → {example['child_concept']} → {example['parent_concept']}")
    print(f"  Bridge entity: {example['bridge_entity']} → {example['parent_concept']} (direct)")

    print(f"\nTheories (NL): {example['theories_nl']}")
    print(f"Observations (NL): {example['observations_nl']}")

    print(f"\n--- ANSWERS ---")
    print(f"  CORRECT (2-hop logic): {example['ground_truth_nl']}")
    print(f"  SHORTCUT (1-hop):      {example['shortcut_answer_nl']}")

    print(f"\nReasoning paths:")
    print(f"  2-hop: {example['child_entity']} is {example['child_concept']} → "
          f"{example['child_concept']} is {example['parent_concept']} → "
          f"{example['child_entity']} has {example['correct_property_name']} → "
          f"{example['parent_concept']}s have {example['correct_property_name']}")
    print(f"  1-hop: {example['bridge_entity']} is {example['parent_concept']} ∧ "
          f"{example['bridge_entity']} has {example['shortcut_property_name']} → "
          f"{example['parent_concept']}s have {example['shortcut_property_name']}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Generate matched minimal pairs for MI analysis')
    parser.add_argument('--n-pairs', '-n', type=int, default=50, help='Number of pairs/examples to generate')
    parser.add_argument('--base-seed', '-s', type=int, default=42, help='Base seed for reproducibility')
    parser.add_argument('--no-negated', action='store_true', help='Exclude negated property cases')
    parser.add_argument('--output', '-o', type=str, default='matched_pairs.pkl', help='Output pickle file')
    parser.add_argument('--preview', '-p', type=int, default=3, help='Number of examples to preview (0 to skip)')
    parser.add_argument('--set', type=str, default='pure',
                        choices=['pure', 'salience', 'inabhyd', 'evidential', 'conflict', '6', 'all'],
                        help='Which set to generate: pure (Set 1), salience (Set 2), inabhyd (Set 4), evidential (Set 5), conflict/6 (Set 6), or all')
    args = parser.parse_args()

    include_negated = not args.no_negated

    if args.set == 'all':
        # Generate all sets
        print("=" * 70)
        print("GENERATING ALL EXPERIMENTAL SETS")
        print("=" * 70)

        # Set 1: Pure matched pairs
        print(f"\n--- Set 1 (Pure Multi-hop) ---")
        pairs = generate_matched_pairs(args.n_pairs, args.base_seed, include_negated)
        with open('matched_pairs_set1_pure.pkl', 'wb') as f:
            pickle.dump(pairs, f)
        print(f"Saved {len(pairs)} pairs to matched_pairs_set1_pure.pkl")

        # Set 2: Salience test
        print(f"\n--- Set 2 (Salience Test) ---")
        salience_examples = generate_salience_set(args.n_pairs, args.base_seed, include_negated)
        avg_mentions = sum(e['parent_mentions'] for e in salience_examples) / len(salience_examples)
        print(f"Average parent mentions: {avg_mentions:.1f}")
        with open('matched_pairs_set2_salience.pkl', 'wb') as f:
            pickle.dump(salience_examples, f)
        print(f"Saved {len(salience_examples)} examples to matched_pairs_set2_salience.pkl")

        # Set 4: INABHYD-style
        print(f"\n--- Set 4 (INABHYD-style) ---")
        inabhyd_examples = generate_inabhyd_style_set(args.n_pairs, args.base_seed, include_negated)
        avg_mentions = sum(e['parent_mentions'] for e in inabhyd_examples) / len(inabhyd_examples)
        print(f"Average parent mentions: {avg_mentions:.1f}")
        with open('matched_pairs_set4_inabhyd.pkl', 'wb') as f:
            pickle.dump(inabhyd_examples, f)
        print(f"Saved {len(inabhyd_examples)} examples to matched_pairs_set4_inabhyd.pkl")

        # Set 5: Evidential Path
        print(f"\n--- Set 5 (Evidential Path) ---")
        evidential_examples = generate_evidential_path_set(args.n_pairs, args.base_seed, include_negated)
        avg_mentions = sum(e['parent_mentions'] for e in evidential_examples) / len(evidential_examples)
        print(f"Average parent mentions: {avg_mentions:.1f}")
        with open('matched_pairs_set5_evidential.pkl', 'wb') as f:
            pickle.dump(evidential_examples, f)
        print(f"Saved {len(evidential_examples)} examples to matched_pairs_set5_evidential.pkl")

        # Set 6: Conflict Test
        print(f"\n--- Set 6 (Conflict Test) ---")
        conflict_examples = generate_conflict_set(args.n_pairs, args.base_seed)
        n_negated_correct = sum(1 for e in conflict_examples if e['correct_is_negated'])
        print(f"Balance: {n_negated_correct} negated correct, {len(conflict_examples) - n_negated_correct} positive correct")
        with open('matched_pairs_set6_conflict.pkl', 'wb') as f:
            pickle.dump(conflict_examples, f)
        print(f"Saved {len(conflict_examples)} examples to matched_pairs_set6_conflict.pkl")

        # Preview
        if args.preview > 0:
            print(f"\n{'='*70}")
            print(f"PREVIEW")
            print(f"{'='*70}")

            print("\n--- Set 1 (Pure) Example ---")
            print_pair_example(pairs[0][0], pairs[0][1])

            print("\n--- Set 2 (Salience) Example ---")
            print_set_example(salience_examples[0], "Set 2 (Salience)")

            print("\n--- Set 4 (INABHYD-style) Example ---")
            print_set_example(inabhyd_examples[0], "Set 4 (INABHYD-style)")

            print("\n--- Set 5 (Evidential Path) Example ---")
            print_set_example(evidential_examples[0], "Set 5 (Evidential Path)")

            print("\n--- Set 6 (Conflict Test) Example ---")
            print_conflict_example(conflict_examples[0])

    elif args.set == 'pure':
        print(f"Generating {args.n_pairs} pure matched pairs (Set 1)...")
        pairs = generate_matched_pairs(args.n_pairs, args.base_seed, include_negated)

        n_negated = sum(1 for h1, h2 in pairs if h1['is_negated'])
        print(f"\nGenerated {len(pairs)} pairs:")
        print(f"  Positive: {len(pairs) - n_negated}, Negated: {n_negated}")

        if args.preview > 0:
            for i, (h1, h2) in enumerate(pairs[:args.preview]):
                print_pair_example(h1, h2)

        with open(args.output, 'wb') as f:
            pickle.dump(pairs, f)
        print(f"\nSaved to {args.output}")

    elif args.set == 'salience':
        print(f"Generating {args.n_pairs} salience test examples (Set 2)...")
        examples = generate_salience_set(args.n_pairs, args.base_seed, include_negated)

        n_negated = sum(1 for e in examples if e['is_negated'])
        avg_mentions = sum(e['parent_mentions'] for e in examples) / len(examples)
        print(f"\nGenerated {len(examples)} examples:")
        print(f"  Positive: {len(examples) - n_negated}, Negated: {n_negated}")
        print(f"  Average parent mentions: {avg_mentions:.1f}")

        if args.preview > 0:
            for e in examples[:args.preview]:
                print_set_example(e, "Set 2 (Salience)")

        output = args.output.replace('.pkl', '_salience.pkl') if 'salience' not in args.output else args.output
        with open(output, 'wb') as f:
            pickle.dump(examples, f)
        print(f"\nSaved to {output}")

    elif args.set == 'inabhyd':
        print(f"Generating {args.n_pairs} INABHYD-style examples (Set 4)...")
        examples = generate_inabhyd_style_set(args.n_pairs, args.base_seed, include_negated)

        n_negated = sum(1 for e in examples if e['is_negated'])
        avg_mentions = sum(e['parent_mentions'] for e in examples) / len(examples)
        print(f"\nGenerated {len(examples)} examples:")
        print(f"  Positive: {len(examples) - n_negated}, Negated: {n_negated}")
        print(f"  Average parent mentions: {avg_mentions:.1f}")

        if args.preview > 0:
            for e in examples[:args.preview]:
                print_set_example(e, "Set 4 (INABHYD-style)")

        output = args.output.replace('.pkl', '_inabhyd.pkl') if 'inabhyd' not in args.output else args.output
        with open(output, 'wb') as f:
            pickle.dump(examples, f)
        print(f"\nSaved to {output}")

    elif args.set == 'evidential':
        print(f"Generating {args.n_pairs} evidential path examples (Set 5)...")
        examples = generate_evidential_path_set(args.n_pairs, args.base_seed, include_negated)

        n_negated = sum(1 for e in examples if e['is_negated'])
        avg_mentions = sum(e['parent_mentions'] for e in examples) / len(examples)
        print(f"\nGenerated {len(examples)} examples:")
        print(f"  Positive: {len(examples) - n_negated}, Negated: {n_negated}")
        print(f"  Average parent mentions: {avg_mentions:.1f}")

        if args.preview > 0:
            for e in examples[:args.preview]:
                print_set_example(e, "Set 5 (Evidential Path)")

        output = args.output.replace('.pkl', '_evidential.pkl') if 'evidential' not in args.output else args.output
        with open(output, 'wb') as f:
            pickle.dump(examples, f)
        print(f"\nSaved to {output}")

    elif args.set in ['conflict', '6']:
        print(f"Generating {args.n_pairs} conflict test examples (Set 6)...")
        examples = generate_conflict_set(args.n_pairs, args.base_seed)

        n_negated_correct = sum(1 for e in examples if e['correct_is_negated'])
        print(f"\nGenerated {len(examples)} examples:")
        print(f"  Negated correct: {n_negated_correct}, Positive correct: {len(examples) - n_negated_correct}")

        if args.preview > 0:
            for e in examples[:args.preview]:
                print_conflict_example(e)

        output = args.output.replace('.pkl', '_conflict.pkl') if 'conflict' not in args.output else args.output
        with open(output, 'wb') as f:
            pickle.dump(examples, f)
        print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
