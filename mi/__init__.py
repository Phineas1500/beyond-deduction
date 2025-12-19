"""
Mechanistic Interpretability module for Beyond Deduction.

Implements Phase 3 (Probing) from the MI research plan to understand
how Gemma 2 9B IT encodes concept levels and output decisions.
"""

from .model_wrapper import Gemma2Wrapper
from .data_loader import load_matched_pairs, load_factorial_results, create_probing_dataset, stratify_gemma_results
from .tokenizer_utils import TokenPositionFinder
from .activation_extractor import ActivationExtractor

__all__ = [
    'Gemma2Wrapper',
    'load_matched_pairs',
    'load_factorial_results',
    'create_probing_dataset',
    'stratify_gemma_results',
    'TokenPositionFinder',
    'ActivationExtractor',
]
