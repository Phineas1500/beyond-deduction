"""
INABHYD Vocabulary - Exact copy for compatibility.

This module provides the full INABHYD vocabulary:
- 90 concept names (9 families x 10 each)
- 13 property families (with negation support = ~104 total)
- 96 entity names

The vocabulary is copied exactly from INABHYD's morphology.py to ensure
perfect compatibility for comparison experiments.
"""
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import random


@dataclass
class Property:
    """
    A property with family and negation tracking.

    Properties within the same family are mutually exclusive
    (e.g., "blue" and "red" from "color" family cannot both apply).

    Attributes:
        family: The property family (e.g., "color", "size")
        name: The property name (e.g., "blue", "small")
        negated: Whether this is a negated property ("is not blue")
    """
    family: str
    name: str
    negated: bool = False

    def __eq__(self, other) -> bool:
        if not isinstance(other, Property):
            return False
        return (self.name == other.name and
                self.negated == other.negated and
                self.family == other.family)

    def __hash__(self):
        return hash((self.name, self.family, self.negated))

    def __str__(self) -> str:
        """Natural language representation."""
        if self.negated:
            return f"not {self.name}"
        return self.name

    def __repr__(self) -> str:
        return f"Property({self.family}:{self.name}, negated={self.negated})"

    def belong_to_same_family(self, other: 'Property') -> bool:
        """Check if two properties are from the same family."""
        return self.family == other.family


# =============================================================================
# INABHYD Vocabulary (exact copy from inabhyd/morphology.py)
# =============================================================================

PROPERTY_FAMILIES: Dict[str, List[str]] = {
    "color": ["blue", "red", "brown", "orange"],
    "size": ["small", "large"],
    "material": ["metallic", "wooden", "luminous", "liquid"],
    "light": ["transparent", "opaque", "translucent"],
    "mood": ["nervous", "happy", "feisty", "shy", "sad"],
    "meta_color": ["bright", "dull", "dark", "pale"],
    "taste": ["sweet", "sour", "spicy", "bitter", "salty"],
    "perfume": ["floral", "fruity", "earthy", "oriental"],
    "temperature": ["hot", "cold", "temperate"],
    "personality": ["kind", "mean", "angry", "amenable", "aggressive"],
    "sound": ["melodic", "muffled", "discordant", "loud"],
    "speed": ["slow", "moderate", "fast"],
    "weather": ["windy", "sunny", "overcast", "rainy", "snowy"]
}

CONCEPT_FAMILIES: List[List[str]] = [
    ["wumpus", "yumpus", "zumpus", "dumpus", "rompus",
     "numpus", "tumpus", "vumpus", "impus", "jompus"],
    ["timple", "yimple", "starple", "shumple", "zhomple",
     "remple", "fomple", "fimple", "worple", "sorple"],
    ["tergit", "gergit", "stergit", "kergit", "shergit",
     "pergit", "bongit", "orgit", "welgit", "jelgit"],
    ["felper", "dolper", "sarper", "irper", "chorper",
     "parper", "arper", "lemper", "hilper", "gomper"],
    ["dalpist", "umpist", "rifpist", "storpist", "shalpist",
     "yerpist", "ilpist", "boompist", "scrompist", "phorpist"],
    ["prilpant", "gwompant", "urpant", "grimpant", "shilpant",
     "zhorpant", "rorpant", "dropant", "lerpant", "quimpant"],
    ["zilpor", "frompor", "stirpor", "porpor", "kurpor",
     "shampor", "werpor", "zhimpor", "yempor", "jempor"],
    ["folpee", "drompee", "delpee", "lompee", "wolpee",
     "gorpee", "shimpee", "rimpee", "twimpee", "serpee"],
    ["daumpin", "thorpin", "borpin", "rofpin", "bempin",
     "dulpin", "harpin", "lirpin", "yompin", "stopin"]
]

ENTITY_NAMES: List[str] = [
    "James", "Mary", "Michael", "Patricia", "Robert", "Jennifer",
    "John", "Linda", "David", "Elizabeth", "William", "Barbara",
    "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Karen",
    "Christopher", "Sarah", "Charles", "Lisa", "Daniel", "Nancy",
    "Matthew", "Sandra", "Anthony", "Betty", "Mark", "Ashley",
    "Donald", "Emily", "Steven", "Kimberly", "Andrew", "Margaret",
    "Paul", "Donna", "Joshua", "Michelle", "Kenneth", "Carol",
    "Kevin", "Amanda", "Brian", "Melissa", "Timothy", "Deborah",
    "Ronald", "Stephanie", "George", "Rebecca", "Jason", "Sharon",
    "Edward", "Laura", "Jeffrey", "Cynthia", "Ryan", "Dorothy",
    "Jacob", "Amy", "Nicholas", "Kathleen", "Gary", "Angela",
    "Eric", "Shirley", "Jonathan", "Emma", "Stephen", "Brenda",
    "Larry", "Pamela", "Justin", "Nicole", "Scott", "Anna",
    "Brandon", "Samantha", "Gregory", "Debra", "Alexander", "Rachel",
    "Patrick", "Carolyn", "Frank", "Janet", "Raymond", "Maria",
    "Jack", "Olivia", "Dennis", "Heather", "Jerry", "Helen"
]

# Compute totals for reference
NUM_CONCEPTS = sum(len(family) for family in CONCEPT_FAMILIES)  # 90
NUM_PROPERTIES = sum(len(props) for props in PROPERTY_FAMILIES.values())  # 52 base
NUM_ENTITIES = len(ENTITY_NAMES)  # 96


class Morphology:
    """
    Vocabulary manager that tracks which concepts/properties/entities
    have been used in a given ontology generation.

    This matches INABHYD's Morphology class exactly:
    - Shuffles vocabulary on init
    - Tracks usage indices to prevent reuse
    - Creates properties with negation variants
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize morphology with optional random seed.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        # Flatten and shuffle concepts
        self._concepts = [c for family in CONCEPT_FAMILIES for c in family]
        random.shuffle(self._concepts)

        # Build property pool with negation variants
        self._property_pool: Dict[str, List[Property]] = defaultdict(list)
        for family, props in PROPERTY_FAMILIES.items():
            for prop_name in props:
                # Add positive and negated versions
                self._property_pool[family].append(Property(family, prop_name, negated=False))
                self._property_pool[family].append(Property(family, prop_name, negated=True))
            random.shuffle(self._property_pool[family])

        # Shuffle entities
        self._entities = list(ENTITY_NAMES)
        random.shuffle(self._entities)

        # Track usage indices
        self._concept_idx = 0
        self._entity_idx = 0
        self._property_idx: Dict[str, int] = {f: 0 for f in PROPERTY_FAMILIES}

    @property
    def next_concept(self) -> str:
        """
        Get next unused concept name.

        Returns:
            The next available concept name

        Raises:
            ValueError: If all concepts have been used
        """
        if self._concept_idx >= len(self._concepts):
            raise ValueError("Exhausted concept vocabulary")
        concept = self._concepts[self._concept_idx]
        self._concept_idx += 1
        return concept

    @property
    def next_entity(self) -> str:
        """
        Get next unused entity name.

        Returns:
            The next available entity name

        Raises:
            ValueError: If all entities have been used
        """
        if self._entity_idx >= len(self._entities):
            raise ValueError("Exhausted entity vocabulary")
        entity = self._entities[self._entity_idx]
        self._entity_idx += 1
        return entity

    def next_property(self, prohibited_families: Optional[Set[str]] = None) -> Property:
        """
        Get next unused property from a non-prohibited family.

        Properties from the same family cannot coexist on the same
        concept or its ancestors/descendants.

        Args:
            prohibited_families: Set of family names to exclude

        Returns:
            The next available property

        Raises:
            ValueError: If no properties are available
        """
        prohibited = prohibited_families or set()

        # Find available candidates
        candidates = []
        for family in self._property_pool:
            if family in prohibited:
                continue
            idx = self._property_idx[family]
            if idx < len(self._property_pool[family]):
                candidates.append((family, self._property_pool[family][idx]))

        if not candidates:
            raise ValueError("No available properties (all families prohibited or exhausted)")

        # Choose randomly from candidates
        family, prop = random.choice(candidates)
        self._property_idx[family] += 1
        return prop

    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        return {
            "concepts_used": self._concept_idx,
            "concepts_available": len(self._concepts) - self._concept_idx,
            "entities_used": self._entity_idx,
            "entities_available": len(self._entities) - self._entity_idx,
            "properties_by_family": {
                f: self._property_idx[f] for f in self._property_idx
            }
        }
