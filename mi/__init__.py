"""
Mechanistic Interpretability module for Beyond Deduction.

Implements probing and attention analysis to understand how Gemma 2 9B IT
encodes concept levels and output decisions.

Key findings:
- Decision is locked by layer 8 (94.1% accuracy from probing)
- Conservation law: H1 + H2 accuracy ≈ 100%
- Model has p ≈ 0.05 for parent output

Phases:
- Phase 3: Probing (activation_extractor.py, probes/)
- Phase 4: Attention Analysis (attention_analysis.py) <- NEW
"""

from .model_wrapper import Gemma2Wrapper
from .data_loader import load_matched_pairs, load_factorial_results, create_probing_dataset, stratify_gemma_results
from .tokenizer_utils import TokenPositionFinder
from .activation_extractor import ActivationExtractor
from .attention_analysis import (
    AttentionAnalysisConfig,
    AttentionResult,
    run_attention_analysis,
    extract_attention_patterns,
    compare_attention_patterns,
)

__all__ = [
    # Model and data loading
    'Gemma2Wrapper',
    'load_matched_pairs',
    'load_factorial_results',
    'create_probing_dataset',
    'stratify_gemma_results',
    'TokenPositionFinder',
    # Probing (Phase 3)
    'ActivationExtractor',
    # Attention Analysis (Phase 4)
    'AttentionAnalysisConfig',
    'AttentionResult',
    'run_attention_analysis',
    'extract_attention_patterns',
    'compare_attention_patterns',
]
