"""
Symbolic FOL renderer for INABHYD ontologies.

This module provides isomorphic symbolic representations of INABHYD data,
mapping natural language concepts/entities/properties to symbolic identifiers
while preserving the exact logical structure.

Symbolic Format:
    - Concepts: c0, c1, c2, ...
    - Entities: e0, e1, e2, ...
    - Properties: p1, p2, p3, ... (negated: ~p1, ~p2, ...)

    - Subtype: forall x: c1(x) -> c0(x)
    - Property: forall x: c0(x) -> p1(x)
    - Membership: c0(e0)
    - Entity property: p1(e0)
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class SymbolicMapping:
    """Tracks the mapping between NL names and symbolic IDs."""
    concepts: Dict[str, int] = field(default_factory=dict)
    entities: Dict[str, int] = field(default_factory=dict)
    properties: Dict[str, int] = field(default_factory=dict)

    # Reverse mappings for debugging
    concept_names: Dict[int, str] = field(default_factory=dict)
    entity_names: Dict[int, str] = field(default_factory=dict)
    property_names: Dict[int, str] = field(default_factory=dict)

    _next_concept_id: int = 0
    _next_entity_id: int = 0
    _next_property_id: int = 1  # Start at 1 to match p1, p2, ... convention

    def get_concept_id(self, name: str) -> int:
        """Get or create a symbolic ID for a concept."""
        if name not in self.concepts:
            self.concepts[name] = self._next_concept_id
            self.concept_names[self._next_concept_id] = name
            self._next_concept_id += 1
        return self.concepts[name]

    def get_entity_id(self, name: str) -> int:
        """Get or create a symbolic ID for an entity."""
        # Handle the "family::name" format from INABHYD
        clean_name = name.split("::")[-1] if "::" in name else name
        if clean_name not in self.entities:
            self.entities[clean_name] = self._next_entity_id
            self.entity_names[self._next_entity_id] = clean_name
            self._next_entity_id += 1
        return self.entities[clean_name]

    def get_property_id(self, prop) -> Tuple[int, bool]:
        """
        Get or create a symbolic ID for a property.

        Returns:
            Tuple of (property_id, is_negated)
        """
        # Property objects have name and is_negated attributes
        prop_name = prop.name if hasattr(prop, 'name') else str(prop)
        is_negated = prop.is_negated if hasattr(prop, 'is_negated') else False

        if prop_name not in self.properties:
            self.properties[prop_name] = self._next_property_id
            self.property_names[self._next_property_id] = prop_name
            self._next_property_id += 1

        return self.properties[prop_name], is_negated

    def to_dict(self) -> Dict[str, Any]:
        """Export mapping as a dictionary."""
        return {
            "concepts": {name: f"c{id}" for name, id in self.concepts.items()},
            "entities": {name: f"e{id}" for name, id in self.entities.items()},
            "properties": {name: f"p{id}" for name, id in self.properties.items()},
        }


class SymbolicFOL:
    """Represents a single FOL statement in symbolic form."""

    def __init__(self, subject_type: str, subject_id: int,
                 predicate_type: str, predicate_id: int,
                 subject_negated: bool = False, predicate_negated: bool = False):
        """
        Create a symbolic FOL statement.

        Args:
            subject_type: 'concept' or 'entity'
            subject_id: Numeric ID of subject
            predicate_type: 'concept' or 'property'
            predicate_id: Numeric ID of predicate
            subject_negated: Whether subject is negated (rare)
            predicate_negated: Whether predicate is negated
        """
        self.subject_type = subject_type
        self.subject_id = subject_id
        self.predicate_type = predicate_type
        self.predicate_id = predicate_id
        self.subject_negated = subject_negated
        self.predicate_negated = predicate_negated

    def __str__(self) -> str:
        """Render as symbolic FOL string."""
        # Format predicate
        if self.predicate_type == 'concept':
            pred = f"c{self.predicate_id}"
        else:  # property
            pred = f"p{self.predicate_id}"

        if self.predicate_negated:
            pred = f"~{pred}"

        # Format based on subject type
        if self.subject_type == 'entity':
            # Entity assertion: c0(e0) or p1(e0)
            subj = f"e{self.subject_id}"
            return f"{pred}({subj})"
        else:
            # Universal statement: forall x: c1(x) -> c0(x)
            subj = f"c{self.subject_id}"
            return f"forall x: {subj}(x) -> {pred}(x)"

    def __repr__(self) -> str:
        return f"SymbolicFOL({self})"


class SymbolicRenderer:
    """
    Renders INABHYD ontologies to symbolic FOL format.

    Usage:
        renderer = SymbolicRenderer()
        result = renderer.render(ontology)

        # result contains:
        # - theories: List of symbolic FOL strings (world model)
        # - observations: List of symbolic FOL strings (evidence)
        # - hypotheses: List of symbolic FOL strings (ground truth)
        # - mapping: SymbolicMapping object
    """

    def __init__(self):
        self.mapping = SymbolicMapping()

    def render(self, ontology) -> Dict[str, Any]:
        """
        Render an INABHYD ontology to symbolic FOL.

        Args:
            ontology: An Ontology object from inabhyd/ontology.py

        Returns:
            Dictionary with theories, observations, hypotheses, and mapping
        """
        # Reset mapping for each ontology
        self.mapping = SymbolicMapping()

        # First pass: assign IDs to all concepts in tree order (BFS)
        # This ensures consistent ordering: root=c0, children=c1,c2,...
        self._assign_concept_ids(ontology)

        # Render each component
        theories = self._render_theories(ontology)
        observations = self._render_observations(ontology)
        hypotheses = self._render_hypotheses(ontology)

        return {
            "theories": theories,
            "observations": observations,
            "hypotheses": hypotheses,
            "mapping": self.mapping.to_dict(),
        }

    def _assign_concept_ids(self, ontology):
        """Assign concept IDs in BFS order from root."""
        # Handle pseudo_root if present
        if ontology.pseudo_root:
            self.mapping.get_concept_id(ontology.pseudo_root.name)

        # BFS through tree
        for level_nodes in ontology.nodes:
            for node in level_nodes:
                self.mapping.get_concept_id(node.name)

    def _render_theories(self, ontology) -> List[str]:
        """Render visible axioms as symbolic FOL theories."""
        theories = []

        for level_nodes in ontology.nodes:
            for node in level_nodes:
                node_id = self.mapping.get_concept_id(node.name)

                # Visible memberships: c0(e0)
                for member in node.members(visible=True):
                    entity_id = self.mapping.get_entity_id(member)
                    fol = SymbolicFOL('entity', entity_id, 'concept', node_id)
                    theories.append(str(fol))

                # Visible properties: forall x: c0(x) -> p1(x)
                for prop in node.properties(visible=True):
                    prop_id, is_negated = self.mapping.get_property_id(prop)
                    fol = SymbolicFOL('concept', node_id, 'property', prop_id,
                                     predicate_negated=is_negated)
                    theories.append(str(fol))

                # Visible parent relations: forall x: c1(x) -> c0(x)
                for parent in node.parents(visible=True):
                    parent_id = self.mapping.get_concept_id(parent.name)
                    fol = SymbolicFOL('concept', node_id, 'concept', parent_id)
                    theories.append(str(fol))

        return theories

    def _render_observations(self, ontology) -> List[str]:
        """Render observations (evidence for hidden axioms) as symbolic FOL."""
        observations = []

        for level_nodes in ontology.nodes:
            for node in level_nodes:
                # Observations for hidden parent (INFER_SUBTYPE task)
                # When parent is hidden, we observe entities belonging to that parent
                if node.parents(visible=False):
                    hidden_parent = node.parents(visible=False)[0]
                    parent_id = self.mapping.get_concept_id(hidden_parent.name)

                    for (chosen_node, member) in node.associated_members_for_recover_ontolog:
                        entity_id = self.mapping.get_entity_id(member)
                        fol = SymbolicFOL('entity', entity_id, 'concept', parent_id)
                        observations.append(str(fol))

                # Observations for hidden properties (INFER_PROPERTY task)
                # When property is hidden, we observe entities having that property
                for prop in node.properties(visible=False):
                    prop_id, is_negated = self.mapping.get_property_id(prop)

                    for (chosen_node, member) in node.associated_members_for_recover_properties[prop]:
                        entity_id = self.mapping.get_entity_id(member)
                        fol = SymbolicFOL('entity', entity_id, 'property', prop_id,
                                         predicate_negated=is_negated)
                        observations.append(str(fol))

                # Observations for hidden memberships (INFER_MEMBERSHIP task)
                # When membership is hidden, we observe entity having properties
                for member in node.members(visible=False):
                    for (chosen_node, prop) in node.associated_properties_for_recover_memberships:
                        prop_id, is_negated = self.mapping.get_property_id(prop)
                        entity_id = self.mapping.get_entity_id(member)
                        fol = SymbolicFOL('entity', entity_id, 'property', prop_id,
                                         predicate_negated=is_negated)
                        observations.append(str(fol))

        return observations

    def _render_hypotheses(self, ontology) -> List[str]:
        """Render hidden axioms as symbolic FOL hypotheses (ground truth)."""
        hypotheses = []

        for level_nodes in ontology.nodes:
            for node in level_nodes:
                node_id = self.mapping.get_concept_id(node.name)

                # Hidden parent relations: forall x: c1(x) -> c0(x)
                for parent in node.parents(visible=False):
                    parent_id = self.mapping.get_concept_id(parent.name)
                    fol = SymbolicFOL('concept', node_id, 'concept', parent_id)
                    hypotheses.append(str(fol))

                # Hidden properties: forall x: c0(x) -> p1(x)
                for prop in node.properties(visible=False):
                    prop_id, is_negated = self.mapping.get_property_id(prop)
                    fol = SymbolicFOL('concept', node_id, 'property', prop_id,
                                     predicate_negated=is_negated)
                    hypotheses.append(str(fol))

                # Hidden memberships: c0(e0)
                for member in node.members(visible=False):
                    entity_id = self.mapping.get_entity_id(member)
                    fol = SymbolicFOL('entity', entity_id, 'concept', node_id)
                    hypotheses.append(str(fol))

        return hypotheses

    def render_cot(self, ontology) -> List[str]:
        """
        Render chain-of-thought reasoning in symbolic FOL.

        This mirrors ontology.CoT but in symbolic form.
        """
        cot = []

        for level_nodes in ontology.nodes:
            for node in level_nodes:
                node_id = self.mapping.get_concept_id(node.name)

                # CoT for hidden memberships
                if node.members(visible=False):
                    for hidden_member in node.members(visible=False):
                        entity_id = self.mapping.get_entity_id(hidden_member)

                        for (chosen_node, prop) in node.associated_properties_for_recover_memberships:
                            chosen_id = self.mapping.get_concept_id(chosen_node.name)
                            prop_id, is_negated = self.mapping.get_property_id(prop)

                            # Chain from node to chosen_node
                            if chosen_node != node:
                                chains = [node]
                                while chains[-1] != chosen_node:
                                    chains.append(chains[-1].parents()[0])

                                for i in range(1, len(chains)):
                                    child_id = self.mapping.get_concept_id(chains[i-1].name)
                                    parent_id = self.mapping.get_concept_id(chains[i].name)
                                    # Subtype relation
                                    cot.append(str(SymbolicFOL('concept', child_id, 'concept', parent_id)))
                                    # Derived membership
                                    cot.append(str(SymbolicFOL('entity', entity_id, 'concept', parent_id)))

                            # Property of chosen node
                            cot.append(str(SymbolicFOL('concept', chosen_id, 'property', prop_id,
                                                       predicate_negated=is_negated)))
                            # Suppose membership
                            cot.append(f"SUPPOSE: {SymbolicFOL('entity', entity_id, 'concept', node_id)}")
                            # Conclude property
                            cot.append(str(SymbolicFOL('entity', entity_id, 'property', prop_id,
                                                       predicate_negated=is_negated)))

                # CoT for hidden properties
                for prop in node.properties(visible=False):
                    prop_id, is_negated = self.mapping.get_property_id(prop)

                    for (chosen_node, member) in node.associated_members_for_recover_properties[prop]:
                        entity_id = self.mapping.get_entity_id(member)
                        chosen_id = self.mapping.get_concept_id(chosen_node.name)

                        # Chain from chosen_node up to node
                        if chosen_node != node:
                            _node = chosen_node
                            while _node.parents() and _node != node:
                                _node_id = self.mapping.get_concept_id(_node.name)
                                parent_id = self.mapping.get_concept_id(_node.parents()[0].name)
                                # Membership in current
                                cot.append(str(SymbolicFOL('entity', entity_id, 'concept', _node_id)))
                                # Subtype relation
                                cot.append(str(SymbolicFOL('concept', _node_id, 'concept', parent_id)))
                                _node = _node.parents()[0]

                        # Final membership
                        cot.append(str(SymbolicFOL('entity', entity_id, 'concept', node_id)))
                        # Suppose property
                        cot.append(f"SUPPOSE: {SymbolicFOL('concept', node_id, 'property', prop_id, predicate_negated=is_negated)}")
                        # Conclude entity has property
                        cot.append(str(SymbolicFOL('entity', entity_id, 'property', prop_id,
                                                   predicate_negated=is_negated)))

                # CoT for hidden subtype relations
                if node.parents(visible=False):
                    hidden_parent = node.parents(visible=False)[0]
                    parent_id = self.mapping.get_concept_id(hidden_parent.name)

                    for (chosen_node, member) in node.associated_members_for_recover_ontolog:
                        entity_id = self.mapping.get_entity_id(member)
                        chosen_id = self.mapping.get_concept_id(chosen_node.name)

                        # Chain from chosen_node up to node
                        if chosen_node != node:
                            _node = chosen_node
                            while _node.parents() and _node != node:
                                _node_id = self.mapping.get_concept_id(_node.name)
                                _parent_id = self.mapping.get_concept_id(_node.parents()[0].name)
                                cot.append(str(SymbolicFOL('entity', entity_id, 'concept', _node_id)))
                                cot.append(str(SymbolicFOL('concept', _node_id, 'concept', _parent_id)))
                                _node = _node.parents()[0]

                        # Membership in node
                        cot.append(str(SymbolicFOL('entity', entity_id, 'concept', node_id)))
                        # Suppose subtype
                        cot.append(f"SUPPOSE: {SymbolicFOL('concept', node_id, 'concept', parent_id)}")
                        # Conclude membership in parent
                        cot.append(str(SymbolicFOL('entity', entity_id, 'concept', parent_id)))

        return cot


def render_ontology(ontology) -> Dict[str, Any]:
    """
    Convenience function to render an ontology to symbolic FOL.

    Args:
        ontology: An Ontology object from inabhyd/ontology.py

    Returns:
        Dictionary with theories, observations, hypotheses, and mapping
    """
    renderer = SymbolicRenderer()
    return renderer.render(ontology)


def format_for_training(ontology, include_cot: bool = False) -> Dict[str, str]:
    """
    Format an ontology as input/target strings for model training.

    Args:
        ontology: An Ontology object from inabhyd/ontology.py
        include_cot: Whether to include chain-of-thought in target

    Returns:
        Dictionary with 'input', 'target', 'symbolic', and 'natural_language' keys
    """
    renderer = SymbolicRenderer()
    symbolic = renderer.render(ontology)

    # Build symbolic input
    input_lines = ["[WORLD_MODEL]"]
    input_lines.extend(symbolic["theories"])
    input_lines.append("[OBSERVATIONS]")
    input_lines.extend(symbolic["observations"])
    input_lines.append("[TASK]")
    input_lines.append("Infer the hidden axiom(s).")

    symbolic_input = "\n".join(input_lines)
    symbolic_target = "\n".join(symbolic["hypotheses"])

    # Include CoT if requested
    if include_cot:
        cot = renderer.render_cot(ontology)
        symbolic_target = "\n".join(cot) + "\n[ANSWER]\n" + symbolic_target

    return {
        "input": symbolic_input,
        "target": symbolic_target,
        "symbolic": symbolic,
        "natural_language": {
            "theories": ontology.theories,
            "observations": ontology.observations,
            "hypotheses": ontology.hypotheses,
        },
        "mapping": symbolic["mapping"],
    }


def generate_dataset(
    num_samples: int,
    hops: int,
    task_type: str = "property",
    difficulty: str = "single",
    mix_hops: bool = False,
    seed: int = 42,
    include_cot: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate a dataset of symbolic FOL samples from INABHYD.

    Args:
        num_samples: Number of samples to generate
        hops: Tree depth (number of reasoning hops + 1)
        task_type: 'property', 'membership', or 'ontology'
        difficulty: 'single', 'easy', 'medium', or 'hard'
        mix_hops: Whether to hide axioms at multiple levels
        seed: Random seed
        include_cot: Whether to include chain-of-thought

    Returns:
        List of sample dictionaries with input, target, symbolic, natural_language, mapping
    """
    import random
    try:
        # When running from within inabhyd directory
        from ontology import Ontology, OntologyConfig, Difficulty
    except ImportError:
        # When running from parent directory
        from inabhyd.ontology import Ontology, OntologyConfig, Difficulty

    random.seed(seed)

    # Parse difficulty
    difficulty_map = {
        "single": Difficulty.SINGLE,
        "easy": Difficulty.EASY,
        "medium": Difficulty.MEDIUM,
        "hard": Difficulty.HARD,
    }
    diff = difficulty_map.get(difficulty.lower(), Difficulty.SINGLE)

    # Build config
    config = OntologyConfig(
        hops=hops,
        recover_property=(task_type == "property"),
        recover_membership=(task_type == "membership"),
        recover_ontology=(task_type == "ontology"),
        difficulty=diff,
        mix_hops=mix_hops,
    )

    samples = []
    attempts = 0
    max_attempts = num_samples * 10

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            ontology = Ontology(config)
            sample = format_for_training(ontology, include_cot=include_cot)
            sample["id"] = len(samples)
            sample["config"] = {
                "hops": hops,
                "task_type": task_type,
                "difficulty": difficulty,
                "mix_hops": mix_hops,
            }
            samples.append(sample)
        except Exception as e:
            continue

    return samples


# For testing
if __name__ == "__main__":
    import random
    from ontology import Ontology, OntologyConfig, Difficulty

    print("=" * 60)
    print("SYMBOLIC FOL RENDERER TEST")
    print("=" * 60)

    # Test INFER_PROPERTY
    print("\n### INFER_PROPERTY (recover_property) ###")
    random.seed(42)
    config = OntologyConfig(hops=3, recover_property=True, difficulty=Difficulty.SINGLE)
    ontology = Ontology(config)

    print("\nOriginal NL:")
    print("Theories:", ontology.theories[:200], "...")
    print("Observations:", ontology.observations)
    print("Hypotheses:", ontology.hypotheses)

    result = render_ontology(ontology)
    print("\nSymbolic FOL:")
    print("Theories:", result["theories"])
    print("Observations:", result["observations"])
    print("Hypotheses:", result["hypotheses"])
    print("Mapping:", result["mapping"])

    # Test INFER_MEMBERSHIP
    print("\n### INFER_MEMBERSHIP (recover_membership) ###")
    random.seed(42)
    config = OntologyConfig(hops=2, recover_membership=True, difficulty=Difficulty.SINGLE)
    ontology = Ontology(config)

    print("\nOriginal NL:")
    print("Observations:", ontology.observations)
    print("Hypotheses:", ontology.hypotheses)

    result = render_ontology(ontology)
    print("\nSymbolic FOL:")
    print("Observations:", result["observations"])
    print("Hypotheses:", result["hypotheses"])

    # Test INFER_SUBTYPE
    print("\n### INFER_SUBTYPE (recover_ontology) ###")
    random.seed(42)
    config = OntologyConfig(hops=2, recover_ontology=True, difficulty=Difficulty.SINGLE)
    ontology = Ontology(config)

    print("\nOriginal NL:")
    print("Observations:", ontology.observations)
    print("Hypotheses:", ontology.hypotheses)

    result = render_ontology(ontology)
    print("\nSymbolic FOL:")
    print("Observations:", result["observations"])
    print("Hypotheses:", result["hypotheses"])
