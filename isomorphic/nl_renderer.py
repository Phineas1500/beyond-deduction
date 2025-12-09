"""
Renders ontology structures to natural language (INABHYD format).

Natural language format uses:
- Quantified rules: "Every/Each wumpus is a yumpus" or "Wumpuses are yumpuses"
- Property rules: "Every wumpus is red" or "Wumpuses are red"
- Negated properties: "Every wumpus is not pale"
- Membership: "Alex is a wumpus"
- Property observations: "Alex is red"

This renderer matches INABHYD's fol.py exactly, including:
- 33% each: "Each X is Y", "Every X is Y", "Xs are Ys"
- Proper article selection (a/an)
- Proper pluralization
"""
import random
from typing import Dict, List, Any
from .core import Ontology, OntologyNode, TaskType
from .morphology import Property


class NLRenderer:
    """
    Renders ontology to INABHYD-style natural language.

    Matches INABHYD's FOL class in fol.py exactly.
    """

    def _pluralize(self, name: str) -> str:
        """
        Create plural form of concept name.

        Matches INABHYD's FOL_Concept.plural_names property.
        """
        if name.endswith('s'):
            return f"{name}es"
        return f"{name}s"

    def _article(self, name: str) -> str:
        """
        Get appropriate article (a/an).

        Matches INABHYD's FOL_Concept.name_with_article property.
        """
        if name[0].lower() in 'aeiou':
            return "an"
        return "a"

    def _name_with_article(self, name: str) -> str:
        """Get name with appropriate article."""
        return f"{self._article(name)} {name}"

    def _render_concept_to_concept(self, subject: str, predicate: str) -> str:
        """
        Render a concept-to-concept (subtype) statement with random variation.

        INABHYD uses CASCADING coin flips (not equal thirds):
        - 33% → "each"
        - 22% (33% of remaining 67%) → "every"
        - 45% (remaining) → plural

        Matches INABHYD's FOL.__str__ for Concept -> Concept exactly.
        """
        # First flip: 33% chance of "each"
        if random.random() < 1/3:
            return f"each {subject} is {self._name_with_article(predicate)}"
        # Second flip: 33% of remaining = 22% total for "every"
        if random.random() < 1/3:
            return f"every {subject} is {self._name_with_article(predicate)}"
        # Remaining 45% → plural
        return f"{self._pluralize(subject)} are {self._pluralize(predicate)}"

    def _render_concept_to_property(self, subject: str, prop: Property) -> str:
        """
        Render a concept-to-property statement with random variation.

        INABHYD uses CASCADING coin flips (not equal thirds):
        - 33% → "each"
        - 22% (33% of remaining 67%) → "every"
        - 45% (remaining) → plural

        Matches INABHYD's FOL.__str__ for Concept -> Property exactly.
        """
        prop_str = str(prop)  # Handles negation: "not pale"

        # First flip: 33% chance of "each"
        if random.random() < 1/3:
            return f"each {subject} is {prop_str}"
        # Second flip: 33% of remaining = 22% total for "every"
        if random.random() < 1/3:
            return f"every {subject} is {prop_str}"
        # Remaining 45% → plural
        return f"{self._pluralize(subject)} are {prop_str}"

    def _render_entity_to_concept(self, entity: str, concept: str) -> str:
        """
        Render an entity-to-concept (membership) statement.

        Always: "Entity is a concept"
        Matches INABHYD's FOL.__str__ for Entity -> Concept.
        """
        return f"{entity} is {self._name_with_article(concept)}"

    def _render_entity_to_property(self, entity: str, prop: Property) -> str:
        """
        Render an entity-to-property statement.

        Always: "Entity is property" or "Entity is not property"
        Matches INABHYD's FOL.__str__ for Entity -> Property.
        """
        return f"{entity} is {prop}"

    def render_theory(self, theory: Dict[str, Any]) -> str:
        """
        Render a single theory/axiom to natural language.

        Args:
            theory: Dict with axiom_type, subject, predicate

        Returns:
            Natural language string (lowercase, will be capitalized later)
        """
        axiom_type = theory["axiom_type"]

        if axiom_type == "subtype":
            child = theory["subject"].name
            parent = theory["predicate"].name
            return self._render_concept_to_concept(child, parent)

        elif axiom_type == "property":
            concept = theory["subject"].name
            prop: Property = theory["predicate"]
            return self._render_concept_to_property(concept, prop)

        elif axiom_type == "membership":
            entity = theory["subject"]
            concept = theory["predicate"].name
            return self._render_entity_to_concept(entity, concept)

        raise ValueError(f"Unknown axiom type: {axiom_type}")

    def render_observation(self, obs: Dict[str, Any]) -> str:
        """
        Render a single observation to natural language.

        Args:
            obs: Dict with subject (entity), predicate (concept or property)

        Returns:
            Natural language string (lowercase, will be capitalized later)
        """
        entity = obs["subject"]

        if isinstance(obs["predicate"], OntologyNode):
            concept = obs["predicate"].name
            return self._render_entity_to_concept(entity, concept)
        elif isinstance(obs["predicate"], Property):
            prop = obs["predicate"]
            return self._render_entity_to_property(entity, prop)

        raise ValueError(f"Unknown predicate type: {type(obs['predicate'])}")

    def render_hypothesis(self, hyp: Dict[str, Any]) -> str:
        """
        Render a hypothesis to natural language.

        Args:
            hyp: Dict with task_type, subject, predicate

        Returns:
            Natural language string (lowercase, will be capitalized later)
        """
        task_type = hyp["task_type"]

        if task_type == TaskType.INFER_SUBTYPE:
            child = hyp["subject"].name
            parent = hyp["predicate"].name
            return self._render_concept_to_concept(child, parent)

        elif task_type == TaskType.INFER_PROPERTY:
            concept = hyp["subject"].name
            prop = hyp["predicate"]
            return self._render_concept_to_property(concept, prop)

        elif task_type == TaskType.INFER_MEMBERSHIP:
            entity = hyp["subject"]
            concept = hyp["predicate"].name
            return self._render_entity_to_concept(entity, concept)

        raise ValueError(f"Unknown task type: {task_type}")

    def render_full(self, ontology: Ontology) -> Dict[str, Any]:
        """
        Render full ontology to natural language format.

        Matches INABHYD's Ontology.theories, .observations, .hypotheses properties.

        Args:
            ontology: The Ontology to render

        Returns:
            Dict with:
            - theories: String of theories (". " separated, capitalized)
            - observations: String of observations (". " separated, capitalized)
            - hypotheses: String of hypotheses (". " separated, capitalized)
        """
        theories = [self.render_theory(t) for t in ontology.get_visible_theories()]
        observations = [self.render_observation(o) for o in ontology.get_observations()]
        hypotheses = [self.render_hypothesis(h) for h in ontology.get_hypotheses()]

        # Shuffle and format (INABHYD style)
        random.shuffle(theories)
        random.shuffle(observations)

        # Capitalize first letter of each sentence
        def capitalize_sentences(sentences: List[str]) -> str:
            capitalized = [s[0].upper() + s[1:] if s else s for s in sentences]
            return ". ".join(capitalized) + "."

        return {
            "theories": capitalize_sentences(theories),
            "observations": capitalize_sentences(observations),
            "hypotheses": capitalize_sentences(hypotheses)
        }

    def format_for_model(self, ontology: Ontology, system_prompt: bool = True) -> str:
        """
        Format the ontology as a prompt for LLM evaluation.

        Matches INABHYD's generate.py prompt format.

        Args:
            ontology: The Ontology to render
            system_prompt: Whether to include the system prompt

        Returns:
            Formatted prompt string
        """
        rendered = self.render_full(ontology)

        if system_prompt:
            prompt = (
                "You are a helpful assistant that performs abduction and induction reasoning. "
                "Your job is to come up with hypotheses that explain observations with given theories. "
                "Each hypothesis should explain as many observations as possible. "
                "You can come up with multiple hypotheses and each hypothesis should take one line "
                "with the format A is B or A is not B. Only output final hypotheses.\n\n"
            )
        else:
            prompt = ""

        prompt += (
            f"Q: {rendered['theories']} "
            f"We observe that: {rendered['observations']} "
            "Please come up with hypothesis to explain observations."
        )

        return prompt

    def format_target(self, ontology: Ontology) -> str:
        """
        Format the hypotheses as the target output.

        Returns:
            The hypotheses string
        """
        rendered = self.render_full(ontology)
        return rendered["hypotheses"]

    def render_cot_step(self, step: Dict[str, Any]) -> str:
        """
        Render a single chain-of-thought step to natural language.

        Args:
            step: Dict with step_type, subject, predicate, text

        Returns:
            Natural language string for this step
        """
        step_type = step["step_type"]
        subject = step["subject"]
        predicate = step["predicate"]

        if step_type == "chain":
            # Concept -> Concept subtype relation
            return self._render_concept_to_concept(subject.name, predicate.name)

        elif step_type == "membership" or step_type == "derive_membership":
            # Entity -> Concept membership
            if isinstance(subject, str):
                return self._render_entity_to_concept(subject, predicate.name)
            else:
                return self._render_entity_to_concept(subject, predicate.name)

        elif step_type == "property":
            # Concept -> Property
            return self._render_concept_to_property(subject.name, predicate)

        elif step_type == "suppose":
            # Hypothesis introduction
            if isinstance(subject, str):
                # Entity membership hypothesis
                return f"suppose {self._render_entity_to_concept(subject, predicate.name)}"
            elif isinstance(predicate, Property):
                # Property hypothesis
                return f"suppose {self._render_concept_to_property(subject.name, predicate)}"
            else:
                # Subtype hypothesis (predicate is OntologyNode)
                return f"suppose {self._render_concept_to_concept(subject.name, predicate.name)}"

        elif step_type == "conclude":
            # Conclusion
            if isinstance(predicate, Property):
                return self._render_entity_to_property(subject, predicate)
            else:
                return self._render_entity_to_concept(subject, predicate.name)

        return step.get("text", "")

    def render_cot(self, ontology: Ontology) -> str:
        """
        Render full chain-of-thought reasoning to natural language.

        Matches INABHYD's CoT property format.

        Args:
            ontology: The Ontology to render

        Returns:
            Natural language CoT string with capitalized sentences
        """
        cot_steps = ontology.get_chain_of_thought()
        rendered_steps = [self.render_cot_step(step) for step in cot_steps]

        # Capitalize first letter of each step
        capitalized = [s[0].upper() + s[1:] if s else s for s in rendered_steps]
        return ". ".join(capitalized) + "."

    def format_for_model_with_cot(
        self,
        ontology: Ontology,
        system_prompt: bool = True,
        include_cot: bool = True
    ) -> str:
        """
        Format the ontology as a prompt with chain-of-thought.

        Args:
            ontology: The Ontology to render
            system_prompt: Whether to include the system prompt
            include_cot: Whether to include CoT in the prompt

        Returns:
            Formatted prompt string with CoT
        """
        rendered = self.render_full(ontology)

        if system_prompt:
            prompt = (
                "You are a helpful assistant that performs abduction and induction reasoning. "
                "Your job is to come up with hypotheses that explain observations with given theories. "
                "Each hypothesis should explain as many observations as possible. "
                "You can come up with multiple hypotheses and each hypothesis should take one line "
                "with the format A is B or A is not B. "
            )
            if include_cot:
                prompt += "Show your reasoning step by step before giving the final answer.\n\n"
            else:
                prompt += "Only output final hypotheses.\n\n"
        else:
            prompt = ""

        prompt += (
            f"Q: {rendered['theories']} "
            f"We observe that: {rendered['observations']} "
            "Please come up with hypothesis to explain observations."
        )

        return prompt

    def format_cot_target(self, ontology: Ontology) -> str:
        """
        Format the CoT + hypotheses as the target output.

        Returns:
            CoT reasoning followed by hypotheses
        """
        cot = self.render_cot(ontology)
        rendered = self.render_full(ontology)
        return f"{cot}\n\nTherefore: {rendered['hypotheses']}"
