"""
Mechanistic Interpretability module for Beyond Deduction.

Implements probing, attention analysis, and activation patching to understand
how Gemma 2 9B IT encodes concept levels and output decisions.

Key findings:
- Decision is locked by layer 8 (94.1% accuracy from probing)
- Conservation law: H1 + H2 accuracy ≈ 100%
- Model has p ≈ 0.05 for parent output
- H3 NOT supported (subsumption attention p=0.45)
- Child attention paradox: failures attend MORE to child

Phases:
- Phase 3: Probing (activation_extractor.py, probes/)
- Phase 4: Attention Analysis (attention_analysis.py)
- Phase 5: Activation Patching (activation_patching.py) <- NEW
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
from .activation_patching import (
    PatchingConfig,
    PatchingResult,
    run_all_experiments,
    run_residual_patching_experiment,
    run_attention_patching_experiment,
    run_bidirectional_patching,
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
    # Activation Patching (Phase 5)
    'PatchingConfig',
    'PatchingResult',
    'run_all_experiments',
    'run_residual_patching_experiment',
    'run_attention_patching_experiment',
    'run_bidirectional_patching',
]
