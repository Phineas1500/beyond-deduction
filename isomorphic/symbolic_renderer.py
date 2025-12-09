"""
Renders ontology structures to symbolic FOL notation.

Symbolic format uses:
- Concepts: c0, c1, c2, ...
- Properties: p1, p2, ... (with ~p1 for negated)
- Entities: e0, e1, e2, ...
- Quantified rules: forall x: c0(x) -> c1(x)
- Ground facts: c0(e0), p1(e0)

This renderer produces output that can be tokenized by IsomorphicTokenizer.
"""
from typing import Dict, List, Any, Tuple, Optional
from .core import Ontology, OntologyNode, TaskType
from .morphology import Property


class SymbolicRenderer:
    """
    Renders ontology to symbolic FOL notation.

    Maintains mappings from NL names to symbolic IDs for isomorphism verification.
    """

    def __init__(self):
        """Initialize the renderer with empty mappings."""
        # Mapping from NL names to symbolic IDs
        self._concept_map: Dict[str, int] = {}
        self._property_map: Dict[Tuple[str, str, bool], int] = {}  # (family, name, negated) -> id
        self._entity_map: Dict[str, int] = {}

        self._next_concept_id = 0
        self._next_property_id = 1  # Start at 1 (p1, p2, ...)
        self._next_entity_id = 0

    def _get_concept_id(self, node: OntologyNode) -> str:
        """Get or assign symbolic concept ID."""
        # Use the symbolic_id already assigned during ontology construction
        return f"c{node.symbolic_id}"

    def _register_concept(self, node: OntologyNode):
        """Register a concept in the mapping."""
        self._concept_map[node.name] = node.symbolic_id

    def _get_property_id(self, prop: Property) -> str:
        """Get or assign symbolic property ID."""
        key = (prop.family, prop.name, prop.negated)
        # Use base property (non-negated) for ID
        base_key = (prop.family, prop.name, False)

        if base_key not in self._property_map:
            self._property_map[base_key] = self._next_property_id
            # Also register negated version with same base ID
            self._property_map[(prop.family, prop.name, True)] = self._next_property_id
            self._next_property_id += 1

        pid = self._property_map[base_key]
        if prop.negated:
            return f"~p{pid}"
        return f"p{pid}"

    def _get_entity_id(self, entity: str) -> str:
        """Get or assign symbolic entity ID."""
        if entity not in self._entity_map:
            self._entity_map[entity] = self._next_entity_id
            self._next_entity_id += 1
        return f"e{self._entity_map[entity]}"

    def render_theory(self, theory: Dict[str, Any]) -> str:
        """
        Render a single theory/axiom to symbolic FOL.

        Args:
            theory: Dict with axiom_type, subject, predicate

        Returns:
            Symbolic FOL string
        """
        axiom_type = theory["axiom_type"]

        if axiom_type == "subtype":
            # forall x: child(x) -> parent(x)
            child_id = self._get_concept_id(theory["subject"])
            parent_id = self._get_concept_id(theory["predicate"])
            self._register_concept(theory["subject"])
            self._register_concept(theory["predicate"])
            return f"forall x: {child_id}(x) -> {parent_id}(x)"

        elif axiom_type == "property":
            # forall x: concept(x) -> property(x)
            concept_id = self._get_concept_id(theory["subject"])
            prop_id = self._get_property_id(theory["predicate"])
            self._register_concept(theory["subject"])
            return f"forall x: {concept_id}(x) -> {prop_id}(x)"

        elif axiom_type == "membership":
            # concept(entity)
            entity_id = self._get_entity_id(theory["subject"])
            concept_id = self._get_concept_id(theory["predicate"])
            self._register_concept(theory["predicate"])
            return f"{concept_id}({entity_id})"

        raise ValueError(f"Unknown axiom type: {axiom_type}")

    def render_observation(self, obs: Dict[str, Any]) -> str:
        """
        Render a single observation to symbolic FOL.

        Args:
            obs: Dict with subject (entity), predicate (concept or property)

        Returns:
            Symbolic FOL string
        """
        entity_id = self._get_entity_id(obs["subject"])

        if isinstance(obs["predicate"], OntologyNode):
            # Entity belongs to concept
            concept_id = self._get_concept_id(obs["predicate"])
            self._register_concept(obs["predicate"])
            return f"{concept_id}({entity_id})"
        elif isinstance(obs["predicate"], Property):
            # Entity has property
            prop_id = self._get_property_id(obs["predicate"])
            return f"{prop_id}({entity_id})"

        raise ValueError(f"Unknown predicate type: {type(obs['predicate'])}")

    def render_hypothesis(self, hyp: Dict[str, Any]) -> str:
        """
        Render a hypothesis to symbolic FOL.

        Args:
            hyp: Dict with task_type, subject, predicate

        Returns:
            Symbolic FOL string
        """
        task_type = hyp["task_type"]

        if task_type == TaskType.INFER_SUBTYPE:
            child_id = self._get_concept_id(hyp["subject"])
            parent_id = self._get_concept_id(hyp["predicate"])
            self._register_concept(hyp["subject"])
            self._register_concept(hyp["predicate"])
            return f"forall x: {child_id}(x) -> {parent_id}(x)"

        elif task_type == TaskType.INFER_PROPERTY:
            concept_id = self._get_concept_id(hyp["subject"])
            prop_id = self._get_property_id(hyp["predicate"])
            self._register_concept(hyp["subject"])
            return f"forall x: {concept_id}(x) -> {prop_id}(x)"

        elif task_type == TaskType.INFER_MEMBERSHIP:
            entity_id = self._get_entity_id(hyp["subject"])
            concept_id = self._get_concept_id(hyp["predicate"])
            self._register_concept(hyp["predicate"])
            return f"{concept_id}({entity_id})"

        raise ValueError(f"Unknown task type: {task_type}")

    def render_full(self, ontology: Ontology) -> Dict[str, Any]:
        """
        Render full ontology to symbolic format.

        Args:
            ontology: The Ontology to render

        Returns:
            Dict with:
            - theories: List of symbolic theory strings
            - observations: List of symbolic observation strings
            - hypotheses: List of symbolic hypothesis strings
            - mapping: Dict mapping NL names -> symbolic IDs
        """
        # First register all concepts to ensure consistent IDs
        for node in ontology.get_all_nodes():
            self._register_concept(node)

        # Render all components
        theories = [self.render_theory(t) for t in ontology.get_visible_theories()]
        observations = [self.render_observation(o) for o in ontology.get_observations()]
        hypotheses = [self.render_hypothesis(h) for h in ontology.get_hypotheses()]

        return {
            "theories": theories,
            "observations": observations,
            "hypotheses": hypotheses,
            "mapping": self.get_mapping()
        }

    def get_mapping(self) -> Dict[str, Dict]:
        """
        Get the NL -> symbolic mapping for verification.

        Returns:
            Dict with concepts, properties, entities mappings
        """
        return {
            "concepts": dict(self._concept_map),
            "properties": {
                f"{k[0]}:{k[1]}:{k[2]}": v
                for k, v in self._property_map.items()
            },
            "entities": dict(self._entity_map)
        }

    def format_for_model(self, ontology: Ontology) -> str:
        """
        Format the ontology as a single string for model input.

        Format:
        [WORLD_MODEL]
        <theories>
        [OBSERVATIONS]
        <observations>
        [TASK]
        Infer the hidden axiom(s).
        [ANSWER]

        Returns:
            Formatted string ready for tokenization
        """
        rendered = self.render_full(ontology)

        lines = ["[WORLD_MODEL]"]
        lines.extend(rendered["theories"])
        lines.append("[OBSERVATIONS]")
        lines.extend(rendered["observations"])
        lines.append("[TASK]")
        lines.append("Infer the hidden axiom(s).")
        lines.append("[ANSWER]")

        return "\n".join(lines)

    def format_target(self, ontology: Ontology) -> str:
        """
        Format the hypotheses as the target output.

        Returns:
            Newline-separated hypotheses
        """
        rendered = self.render_full(ontology)
        return "\n".join(rendered["hypotheses"])

    def render_cot_step(self, step: Dict[str, Any]) -> str:
        """
        Render a single chain-of-thought step to symbolic FOL.

        Args:
            step: Dict with step_type, subject, predicate, text

        Returns:
            Symbolic FOL string for this step
        """
        step_type = step["step_type"]
        subject = step["subject"]
        predicate = step["predicate"]

        if step_type == "chain":
            # Concept -> Concept subtype relation
            child_id = self._get_concept_id(subject)
            parent_id = self._get_concept_id(predicate)
            return f"forall x: {child_id}(x) -> {parent_id}(x)"

        elif step_type == "membership" or step_type == "derive_membership":
            # Entity -> Concept membership
            entity_id = self._get_entity_id(subject)
            concept_id = self._get_concept_id(predicate)
            return f"{concept_id}({entity_id})"

        elif step_type == "property":
            # Concept -> Property
            concept_id = self._get_concept_id(subject)
            prop_id = self._get_property_id(predicate)
            return f"forall x: {concept_id}(x) -> {prop_id}(x)"

        elif step_type == "suppose":
            # Hypothesis introduction
            if isinstance(subject, str):
                # Entity membership hypothesis
                entity_id = self._get_entity_id(subject)
                concept_id = self._get_concept_id(predicate)
                return f"suppose {concept_id}({entity_id})"
            elif isinstance(predicate, Property):
                # Property hypothesis
                concept_id = self._get_concept_id(subject)
                prop_id = self._get_property_id(predicate)
                return f"suppose forall x: {concept_id}(x) -> {prop_id}(x)"
            else:
                # Subtype hypothesis
                child_id = self._get_concept_id(subject)
                parent_id = self._get_concept_id(predicate)
                return f"suppose forall x: {child_id}(x) -> {parent_id}(x)"

        elif step_type == "conclude":
            # Conclusion
            if isinstance(predicate, Property):
                entity_id = self._get_entity_id(subject)
                prop_id = self._get_property_id(predicate)
                return f"{prop_id}({entity_id})"
            else:
                entity_id = self._get_entity_id(subject)
                concept_id = self._get_concept_id(predicate)
                return f"{concept_id}({entity_id})"

        return ""

    def render_cot(self, ontology: Ontology) -> List[str]:
        """
        Render full chain-of-thought reasoning to symbolic FOL.

        Args:
            ontology: The Ontology to render

        Returns:
            List of symbolic FOL strings for each CoT step
        """
        # First register all concepts
        for node in ontology.get_all_nodes():
            self._register_concept(node)

        cot_steps = ontology.get_chain_of_thought()
        return [self.render_cot_step(step) for step in cot_steps]

    def format_for_model_with_cot(self, ontology: Ontology) -> str:
        """
        Format the ontology with chain-of-thought as a single string.

        Format:
        [WORLD_MODEL]
        <theories>
        [OBSERVATIONS]
        <observations>
        [TASK]
        Show your reasoning step by step, then infer the hidden axiom(s).
        [ANSWER]

        Returns:
            Formatted string ready for tokenization
        """
        rendered = self.render_full(ontology)

        lines = ["[WORLD_MODEL]"]
        lines.extend(rendered["theories"])
        lines.append("[OBSERVATIONS]")
        lines.extend(rendered["observations"])
        lines.append("[TASK]")
        lines.append("Show your reasoning step by step, then infer the hidden axiom(s).")
        lines.append("[ANSWER]")

        return "\n".join(lines)

    def format_cot_target(self, ontology: Ontology) -> str:
        """
        Format the CoT + hypotheses as the target output.

        Returns:
            CoT reasoning followed by hypotheses
        """
        cot_steps = self.render_cot(ontology)
        rendered = self.render_full(ontology)

        lines = cot_steps
        lines.append("[HYPOTHESIS]")
        lines.extend(rendered["hypotheses"])

        return "\n".join(lines)
