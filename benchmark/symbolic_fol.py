"""
Symbolic FOL conversion utilities.
Converts human-readable FOL to abstract symbols:
- Concepts (dalpist, gwompant, etc.) -> c1, c2, c3...
- Properties (rainy, blue, etc.) -> p1, p2, p3...
- Entities (Amy, James, etc.) -> e1, e2, e3...

This reduces semantic bias by removing meaningful names from the logical structure.
"""

import re
from typing import Dict, Tuple

# Known properties from morphology.py (these are adjectives)
KNOWN_PROPERTIES = {
    # color
    "blue", "red", "brown", "orange",
    # size
    "small", "large",
    # material
    "metallic", "wooden", "luminous", "liquid",
    # light
    "transparent", "opaque", "translucent",
    # mood
    "nervous", "happy", "feisty", "shy", "sad",
    # meta_color
    "bright", "dull", "dark", "pale",
    # taste
    "sweet", "sour", "spicy", "bitter", "salty",
    # perfume
    "floral", "fruity", "earthy", "oriental",
    # temperature
    "hot", "cold", "temperate",
    # personality
    "kind", "mean", "angry", "amenable", "aggressive",
    # sound
    "melodic", "muffled", "discordant", "loud",
    # speed
    "slow", "moderate", "fast",
    # weather
    "windy", "sunny", "overcast", "rainy", "snowy"
}

# Known entity names from morphology.py (proper nouns, lowercase for matching)
KNOWN_ENTITIES = {
    "james", "mary", "michael", "patricia", "robert", "jennifer",
    "john", "linda", "david", "elizabeth", "william", "barbara",
    "richard", "susan", "joseph", "jessica", "thomas", "karen",
    "christopher", "sarah", "charles", "lisa", "daniel", "nancy",
    "matthew", "sandra", "anthony", "betty", "mark", "ashley",
    "donald", "emily", "steven", "kimberly", "andrew", "margaret",
    "paul", "donna", "joshua", "michelle", "kenneth", "carol",
    "kevin", "amanda", "brian", "melissa", "timothy", "deborah",
    "ronald", "stephanie", "george", "rebecca", "jason", "sharon",
    "edward", "laura", "jeffrey", "cynthia", "ryan", "dorothy",
    "jacob", "amy", "nicholas", "kathleen", "gary", "angela",
    "eric", "shirley", "jonathan", "emma", "stephen", "brenda",
    "larry", "pamela", "justin", "nicole", "scott", "anna",
    "brandon", "samantha", "gregory", "debra", "alexander", "rachel",
    "patrick", "carolyn", "frank", "janet", "raymond", "maria",
    "jack", "olivia", "dennis", "heather", "jerry", "helen"
}


class SymbolicFOLConverter:
    """Converts human-readable FOL to symbolic representation."""

    def __init__(self):
        self.concept_map: Dict[str, str] = {}
        self.property_map: Dict[str, str] = {}
        self.entity_map: Dict[str, str] = {}
        self._concept_counter = 0
        self._property_counter = 0
        self._entity_counter = 0

    def _get_or_create_symbol(self, name: str) -> str:
        """Get existing symbol or create new one for a name."""
        name_lower = name.lower()

        # Determine type and get/create symbol
        if name_lower in KNOWN_ENTITIES:
            if name_lower not in self.entity_map:
                self._entity_counter += 1
                self.entity_map[name_lower] = f"e{self._entity_counter}"
            return self.entity_map[name_lower]

        if name_lower in KNOWN_PROPERTIES:
            if name_lower not in self.property_map:
                self._property_counter += 1
                self.property_map[name_lower] = f"p{self._property_counter}"
            return self.property_map[name_lower]

        # Default: concept (nonsense words like dalpist, gwompant)
        if name_lower not in self.concept_map:
            self._concept_counter += 1
            self.concept_map[name_lower] = f"c{self._concept_counter}"
        return self.concept_map[name_lower]

    def convert_fol_string(self, fol_text: str) -> str:
        """
        Convert FOL text to symbolic representation.

        Handles:
        - Ground atoms: predicate(constant) -> symbol(symbol)
        - Universal rules: ∀x(P(x) → Q(x)) -> ∀x(symbol(x) → symbol(x))
        - Negation: ¬predicate(constant) -> ¬symbol(symbol)
        """
        result = fol_text

        # Find all predicate(argument) patterns and replace
        # This handles both ground atoms like dalpist(Amy) and universal like dalpist(x)
        def replace_predicate_arg(match):
            pred = match.group(1)
            arg = match.group(2)

            # Don't replace the variable 'x'
            if arg.lower() == 'x':
                sym_pred = self._get_or_create_symbol(pred)
                return f"{sym_pred}(x)"
            else:
                sym_pred = self._get_or_create_symbol(pred)
                sym_arg = self._get_or_create_symbol(arg)
                return f"{sym_pred}({sym_arg})"

        # Match predicate(argument) patterns
        result = re.sub(r'(\w+)\((\w+)\)', replace_predicate_arg, result)

        return result

    def get_mapping(self) -> Dict[str, Dict[str, str]]:
        """Return the complete mapping for analysis."""
        return {
            'concepts': dict(self.concept_map),
            'properties': dict(self.property_map),
            'entities': dict(self.entity_map)
        }

    def get_reverse_mapping(self) -> Dict[str, str]:
        """Return symbol -> name mapping."""
        reverse = {}
        for name, sym in self.concept_map.items():
            reverse[sym] = name
        for name, sym in self.property_map.items():
            reverse[sym] = name
        for name, sym in self.entity_map.items():
            reverse[sym] = name
        return reverse


def convert_ontology_to_symbolic(ontology) -> Tuple[str, str, str, SymbolicFOLConverter]:
    """
    Convert an ontology's FOL representations to symbolic form.

    Args:
        ontology: An Ontology object with fol_theories, fol_observations, fol_hypotheses

    Returns:
        (symbolic_theories, symbolic_observations, symbolic_hypotheses, converter)
    """
    converter = SymbolicFOLConverter()

    # Convert in order: theories first, then observations, then hypotheses
    # This ensures consistent mapping (same name -> same symbol)
    symbolic_theories = converter.convert_fol_string(ontology.fol_theories)
    symbolic_observations = converter.convert_fol_string(ontology.fol_observations)
    symbolic_hypotheses = converter.convert_fol_string(ontology.fol_hypotheses)

    return symbolic_theories, symbolic_observations, symbolic_hypotheses, converter


# Quick test
if __name__ == "__main__":
    # Test basic conversion
    converter = SymbolicFOLConverter()

    # Test case 1: Simple ground atoms
    test1 = "dalpist(Amy). dalpist(Jerry). dalpist(Pamela)."
    result1 = converter.convert_fol_string(test1)
    print("Test 1 - Ground atoms:")
    print(f"  Original: {test1}")
    print(f"  Symbolic: {result1}")
    print()

    # Test case 2: Observations with properties
    test2 = "rainy(Amy). rainy(Jerry). rainy(Pamela)."
    result2 = converter.convert_fol_string(test2)
    print("Test 2 - Properties:")
    print(f"  Original: {test2}")
    print(f"  Symbolic: {result2}")
    print()

    # Test case 3: Universal quantifier
    test3 = "∀x(dalpist(x) → rainy(x))"
    result3 = converter.convert_fol_string(test3)
    print("Test 3 - Universal rule:")
    print(f"  Original: {test3}")
    print(f"  Symbolic: {result3}")
    print()

    # Test case 4: Negation
    test4 = "¬slow(Amy). ∀x(gwompant(x) → ¬cold(x))"
    result4 = converter.convert_fol_string(test4)
    print("Test 4 - Negation:")
    print(f"  Original: {test4}")
    print(f"  Symbolic: {result4}")
    print()

    # Show mapping
    print("Mapping:")
    print(f"  {converter.get_mapping()}")
    print()

    # Test with real ontology if available
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ontology import Ontology, OntologyConfig, Difficulty
        from random import seed
        import numpy as np

        seed(62471893)
        np.random.seed(62471893)

        config = OntologyConfig(
            hops=1,
            recover_property=True,
            difficulty=Difficulty.SINGLE,
            mix_hops=False
        )

        ont = Ontology(config)

        print("=" * 60)
        print("Real Ontology Test:")
        print("=" * 60)
        print(f"\nOriginal FOL:")
        print(f"  Theories: {ont.fol_theories}")
        print(f"  Observations: {ont.fol_observations}")
        print(f"  Hypotheses: {ont.fol_hypotheses}")

        sym_t, sym_o, sym_h, conv = convert_ontology_to_symbolic(ont)

        print(f"\nSymbolic FOL:")
        print(f"  Theories: {sym_t}")
        print(f"  Observations: {sym_o}")
        print(f"  Hypotheses: {sym_h}")
        print(f"\nMapping: {conv.get_mapping()}")

    except Exception as e:
        print(f"Skipping ontology test: {e}")
