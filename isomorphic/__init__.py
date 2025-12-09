"""
Isomorphic data generation module for INABHYD-style reasoning tasks.

This module generates semantically identical symbolic FOL and natural language
representations of ontology-based reasoning problems.

Main components:
- Difficulty: SINGLE/EASY/MEDIUM/HARD difficulty levels
- Morphology: INABHYD vocabulary (89 concepts, 52 properties, 96 entities)
- Ontology: Core structure with visibility-based axiom hiding
- SymbolicRenderer: Render to FOL notation (forall x: c0(x) -> p1(x))
- NLRenderer: Render to natural language (Every wumpus is red)
- IsomorphicTokenizer: Expanded tokenizer (~311 tokens)
- ParsimonyEvaluator: Weak/strong accuracy metrics
- QualityScoreCalculator: Full INABHYD quality score (q(H) formula)
"""

from .difficulty import Difficulty, BiasedCoin
from .morphology import Property, Morphology, PROPERTY_FAMILIES, CONCEPT_FAMILIES, ENTITY_NAMES
from .core import OntologyNode, OntologyConfig, Ontology, TaskType
from .symbolic_renderer import SymbolicRenderer
from .nl_renderer import NLRenderer
from .tokenizer import IsomorphicTokenizer
from .evaluation import (
    ParsimonyEvaluator,
    weak_accuracy,
    strong_accuracy,
    QualityScoreCalculator,
    compute_quality_score_from_ontology,
)

__all__ = [
    # Difficulty
    'Difficulty',
    'BiasedCoin',
    # Morphology
    'Property',
    'Morphology',
    'PROPERTY_FAMILIES',
    'CONCEPT_FAMILIES',
    'ENTITY_NAMES',
    # Core
    'OntologyNode',
    'OntologyConfig',
    'Ontology',
    'TaskType',
    # Renderers
    'SymbolicRenderer',
    'NLRenderer',
    # Tokenizer
    'IsomorphicTokenizer',
    # Evaluation
    'ParsimonyEvaluator',
    'weak_accuracy',
    'strong_accuracy',
    'QualityScoreCalculator',
    'compute_quality_score_from_ontology',
]
