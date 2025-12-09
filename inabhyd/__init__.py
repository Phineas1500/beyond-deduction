"""
INABHYD - Induction, Abduction, and Hypothesis Discovery benchmark.

This package provides the original INABHYD ontology generation with an
additional symbolic FOL renderer for training interpretable models.
"""

from .ontology import Ontology, OntologyConfig, OntologyNode, Difficulty
from .morphology import Morphology, Prop
from .symbolic import (
    SymbolicRenderer,
    SymbolicMapping,
    SymbolicFOL,
    render_ontology,
    format_for_training,
    generate_dataset,
)

__all__ = [
    "Ontology",
    "OntologyConfig",
    "OntologyNode",
    "Difficulty",
    "Morphology",
    "Prop",
    "SymbolicRenderer",
    "SymbolicMapping",
    "SymbolicFOL",
    "render_ontology",
    "format_for_training",
    "generate_dataset",
]
