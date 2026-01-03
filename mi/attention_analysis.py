"""
Attention Analysis for Ontology Reasoning in Gemma 2 9B
========================================================

Tests Hypothesis H3: H1 failures show higher attention to subsumption
sentences ("Each X is a Y") compared to H1 successes.

Key finding from probing: Decision is locked by layer 8 (94.1% accuracy)
Key question: What attention patterns distinguish H1 failures (parent output)
from H1 successes (child output)?

Integrates with existing MI infrastructure:
- data_loader.py: ProbingExample, stratify_gemma_results()
- model_wrapper.py: Gemma2Wrapper with GQA helpers
- tokenizer_utils.py: TokenPositionFinder

Technical considerations for Gemma 2 9B:
- GQA: 16 query heads share 8 KV heads → analyze at KV-group level
- Sliding window: Only even layers (0, 2, 4, ..., 40) have global attention
- Must use eager attention (Flash Attention incompatible with soft-capping)

Author: Ram (CS 577 Project)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle
from scipy import stats

# Use Agg backend for headless environments (Modal, remote servers)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .data_loader import (
    ProbingExample,
    create_probing_dataset,
    stratify_gemma_results,
    load_factorial_results,
    load_matched_pairs,
)
from .model_wrapper import Gemma2Wrapper
from .tokenizer_utils import TokenPositionFinder


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AttentionAnalysisConfig:
    """Configuration for attention analysis."""

    # Focus layers based on probing results (decision at layer 8)
    # Only even layers have global attention
    focus_layers: List[int] = field(default_factory=lambda: [4, 6, 8, 10, 12, 14, 16, 18, 20])

    # Gemma 2 9B architecture
    num_layers: int = 42
    num_query_heads: int = 16
    num_kv_heads: int = 8  # GQA: 2 query heads per KV head

    # Analysis settings
    batch_size: int = 1  # Process one at a time for memory
    save_intermediate: bool = True

    # Output
    output_dir: str = "attention_results"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AttentionResult:
    """Results from attention analysis for a single example."""

    example_idx: int
    h1_or_h2: str
    h1_success: bool  # Did model correctly output child-level hypothesis?

    # Attention scores by layer and KV group
    # Maps: layer -> kv_group -> attention score
    subsumption_attention: Dict[int, Dict[int, float]] = field(default_factory=dict)
    child_attention: Dict[int, Dict[int, float]] = field(default_factory=dict)
    parent_attention: Dict[int, Dict[int, float]] = field(default_factory=dict)

    # Metadata
    child_concept: str = ""
    parent_concept: str = ""
    seq_len: int = 0


# ============================================================================
# ATTENTION EXTRACTION
# ============================================================================

def extract_attention_patterns(
    model: Gemma2Wrapper,
    example: ProbingExample,
    positions: Dict[str, List[int]],
    config: AttentionAnalysisConfig,
) -> AttentionResult:
    """
    Extract attention patterns from model for a single example.

    Returns attention FROM final position TO:
    - Subsumption sentence tokens
    - Child concept tokens
    - Parent concept tokens

    Handles GQA by averaging over query heads within each KV group.

    Args:
        model: Loaded Gemma2Wrapper
        example: ProbingExample with prompt and metadata
        positions: Dict from TokenPositionFinder with token positions
        config: Analysis configuration

    Returns:
        AttentionResult with attention scores by layer and KV group
    """
    device = next(model.model.parameters()).device

    # Tokenize
    inputs = model.tokenize(example.prompt)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    seq_len = input_ids.shape[1]
    final_pos = seq_len - 1

    # Forward pass with attention output
    with torch.no_grad():
        outputs = model.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )

    attentions = outputs.attentions  # Tuple of [batch, heads, seq, seq] per layer

    # Initialize result
    result = AttentionResult(
        example_idx=example.idx,
        h1_or_h2=example.h1_or_h2,
        h1_success=example.model_correct if example.h1_or_h2 == "h1" else None,
        child_concept=example.child_concept,
        parent_concept=example.root_concept,
        seq_len=seq_len,
    )

    # For H1 examples, h1_success is model_correct
    # For H2 examples, we need to look at the pair's H1 result
    if example.h1_or_h2 == "h1":
        result.h1_success = example.model_correct

    # Extract attention for each focus layer
    for layer in config.focus_layers:
        if layer >= len(attentions):
            continue
        if layer % 2 == 1:  # Skip odd layers (local attention only)
            continue

        layer_attn = attentions[layer][0]  # [heads, seq, seq]

        result.subsumption_attention[layer] = {}
        result.child_attention[layer] = {}
        result.parent_attention[layer] = {}

        # Analyze by KV group (2 query heads share 1 KV head)
        for kv_group in range(config.num_kv_heads):
            q_heads = model.get_query_heads_for_kv_group(kv_group)

            # Average attention across query heads in this KV group
            group_attn = layer_attn[q_heads, :, :].mean(dim=0)  # [seq, seq]

            # Attention from final position to regions
            if positions.get('subsumption_positions'):
                sub_positions = [p for p in positions['subsumption_positions'] if p < seq_len]
                if sub_positions:
                    sub_attn = group_attn[final_pos, sub_positions].mean().item()
                else:
                    sub_attn = 0.0
            else:
                sub_attn = 0.0

            if positions.get('child_positions'):
                child_positions = [p for p in positions['child_positions'] if p < seq_len]
                if child_positions:
                    child_attn = group_attn[final_pos, child_positions].mean().item()
                else:
                    child_attn = 0.0
            else:
                child_attn = 0.0

            if positions.get('parent_positions'):
                parent_positions = [p for p in positions['parent_positions'] if p < seq_len]
                if parent_positions:
                    parent_attn = group_attn[final_pos, parent_positions].mean().item()
                else:
                    parent_attn = 0.0
            else:
                parent_attn = 0.0

            result.subsumption_attention[layer][kv_group] = sub_attn
            result.child_attention[layer][kv_group] = child_attn
            result.parent_attention[layer][kv_group] = parent_attn

    # Clean up
    del attentions, outputs
    torch.cuda.empty_cache()

    return result


def extract_attention_batch(
    model: Gemma2Wrapper,
    examples: List[ProbingExample],
    config: AttentionAnalysisConfig,
    verbose: bool = True,
) -> List[AttentionResult]:
    """
    Extract attention patterns for multiple examples.

    Args:
        model: Loaded Gemma2Wrapper
        examples: List of ProbingExample
        config: Analysis configuration
        verbose: Print progress

    Returns:
        List of AttentionResult
    """
    position_finder = TokenPositionFinder(model.tokenizer)
    results = []

    for i, example in enumerate(examples):
        if verbose and i % 10 == 0:
            print(f"  Processing example {i+1}/{len(examples)}")

        # Get token positions
        positions = position_finder.get_positions_for_probing(
            example.prompt,
            example.child_concept,
            example.root_concept,
        )

        # Extract attention
        result = extract_attention_patterns(model, example, positions, config)
        results.append(result)

        # Memory cleanup every 20 examples
        if i % 20 == 0:
            torch.cuda.empty_cache()

    return results


# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

def compare_attention_patterns(
    h1_failures: List[AttentionResult],
    h1_successes: List[AttentionResult],
    config: AttentionAnalysisConfig,
) -> Dict:
    """
    Statistical comparison of attention patterns between H1 failures and successes.

    Tests hypothesis H3: H1 failures show higher attention to subsumption sentences.

    Args:
        h1_failures: Results from examples where model output parent (H1 wrong)
        h1_successes: Results from examples where model output child (H1 correct)
        config: Analysis configuration

    Returns:
        Dict with statistical comparisons by layer
    """
    results = {
        'by_layer': {},
        'aggregate': {},
        'by_kv_group': {},
    }

    # Analyze each layer
    for layer in config.focus_layers:
        if layer % 2 == 1:  # Skip odd layers
            continue

        failure_attn = []
        success_attn = []

        for result in h1_failures:
            if layer in result.subsumption_attention:
                # Average across KV groups
                avg_attn = np.mean(list(result.subsumption_attention[layer].values()))
                failure_attn.append(avg_attn)

        for result in h1_successes:
            if layer in result.subsumption_attention:
                avg_attn = np.mean(list(result.subsumption_attention[layer].values()))
                success_attn.append(avg_attn)

        if failure_attn and success_attn:
            # T-test
            t_stat, p_value = stats.ttest_ind(failure_attn, success_attn)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((len(failure_attn) - 1) * np.var(failure_attn) +
                 (len(success_attn) - 1) * np.var(success_attn)) /
                (len(failure_attn) + len(success_attn) - 2)
            )
            cohens_d = (np.mean(failure_attn) - np.mean(success_attn)) / pooled_std if pooled_std > 0 else 0

            results['by_layer'][layer] = {
                'failure_mean': np.mean(failure_attn),
                'failure_std': np.std(failure_attn),
                'success_mean': np.mean(success_attn),
                'success_std': np.std(success_attn),
                'difference': np.mean(failure_attn) - np.mean(success_attn),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'n_failures': len(failure_attn),
                'n_successes': len(success_attn),
            }

    # Aggregate analysis across all layers
    all_failure_attn = []
    all_success_attn = []

    for result in h1_failures:
        for layer in result.subsumption_attention:
            for kv_group in result.subsumption_attention[layer]:
                all_failure_attn.append(result.subsumption_attention[layer][kv_group])

    for result in h1_successes:
        for layer in result.subsumption_attention:
            for kv_group in result.subsumption_attention[layer]:
                all_success_attn.append(result.subsumption_attention[layer][kv_group])

    if all_failure_attn and all_success_attn:
        t_stat, p_value = stats.ttest_ind(all_failure_attn, all_success_attn)
        results['aggregate'] = {
            'failure_mean': np.mean(all_failure_attn),
            'failure_std': np.std(all_failure_attn),
            'success_mean': np.mean(all_success_attn),
            'success_std': np.std(all_success_attn),
            'difference': np.mean(all_failure_attn) - np.mean(all_success_attn),
            't_statistic': t_stat,
            'p_value': p_value,
        }

    return results


def analyze_layer_kv_patterns(
    h1_failures: List[AttentionResult],
    h1_successes: List[AttentionResult],
    config: AttentionAnalysisConfig,
) -> Dict:
    """
    Analyze which specific layers and KV groups show the largest differences.

    Identifies candidate components for intervention (Phase 7).

    Args:
        h1_failures: Results from H1 failure examples
        h1_successes: Results from H1 success examples
        config: Analysis configuration

    Returns:
        Dict with per-component analysis and rankings
    """
    differences = {}

    for layer in config.focus_layers:
        if layer % 2 == 1:
            continue

        differences[layer] = {}

        for kv_group in range(config.num_kv_heads):
            failure_vals = [
                r.subsumption_attention.get(layer, {}).get(kv_group, 0)
                for r in h1_failures
            ]
            success_vals = [
                r.subsumption_attention.get(layer, {}).get(kv_group, 0)
                for r in h1_successes
            ]

            if failure_vals and success_vals:
                diff = np.mean(failure_vals) - np.mean(success_vals)

                # T-test for this specific component
                if len(failure_vals) > 1 and len(success_vals) > 1:
                    t_stat, p_value = stats.ttest_ind(failure_vals, success_vals)
                else:
                    t_stat, p_value = 0.0, 1.0

                differences[layer][kv_group] = {
                    'difference': diff,
                    'failure_mean': np.mean(failure_vals),
                    'success_mean': np.mean(success_vals),
                    't_statistic': t_stat,
                    'p_value': p_value,
                }

    # Rank by absolute difference
    all_diffs = []
    for layer in differences:
        for kv_group in differences[layer]:
            all_diffs.append({
                'layer': layer,
                'kv_group': kv_group,
                **differences[layer][kv_group]
            })

    all_diffs.sort(key=lambda x: abs(x['difference']), reverse=True)

    return {
        'by_component': differences,
        'top_differences': all_diffs[:10],
    }


def compare_attention_to_regions(
    h1_failures: List[AttentionResult],
    h1_successes: List[AttentionResult],
    config: AttentionAnalysisConfig,
) -> Dict:
    """
    Compare attention to different regions: subsumption vs child vs parent.

    Tests whether failures attend more to parent-related tokens.

    Args:
        h1_failures: Results from H1 failure examples
        h1_successes: Results from H1 success examples
        config: Analysis configuration

    Returns:
        Dict comparing attention to different regions
    """
    regions = ['subsumption', 'child', 'parent']
    comparisons = {}

    for region in regions:
        attn_key = f'{region}_attention'

        failure_vals = []
        success_vals = []

        for r in h1_failures:
            region_attn = getattr(r, attn_key, {})
            for layer in region_attn:
                for kv_group in region_attn[layer]:
                    failure_vals.append(region_attn[layer][kv_group])

        for r in h1_successes:
            region_attn = getattr(r, attn_key, {})
            for layer in region_attn:
                for kv_group in region_attn[layer]:
                    success_vals.append(region_attn[layer][kv_group])

        if failure_vals and success_vals:
            t_stat, p_value = stats.ttest_ind(failure_vals, success_vals)
            comparisons[region] = {
                'failure_mean': np.mean(failure_vals),
                'success_mean': np.mean(success_vals),
                'difference': np.mean(failure_vals) - np.mean(success_vals),
                't_statistic': t_stat,
                'p_value': p_value,
            }

    return comparisons


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_attention_comparison(
    comparison_results: Dict,
    config: AttentionAnalysisConfig,
    save_path: Optional[str] = None,
) -> None:
    """
    Create visualization of attention comparison between H1 failures and successes.

    Args:
        comparison_results: Output from compare_attention_patterns()
        config: Analysis configuration
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Attention by layer
    ax1 = axes[0, 0]
    layers = sorted(comparison_results['by_layer'].keys())

    if layers:
        failure_means = [comparison_results['by_layer'][l]['failure_mean'] for l in layers]
        success_means = [comparison_results['by_layer'][l]['success_mean'] for l in layers]

        x = np.arange(len(layers))
        width = 0.35

        ax1.bar(x - width/2, failure_means, width, label='H1 Failures (parent output)', color='coral')
        ax1.bar(x + width/2, success_means, width, label='H1 Successes (child output)', color='steelblue')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Mean Attention to Subsumption')
        ax1.set_title('Attention to Subsumption Sentence by Layer')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layers)
        ax1.legend()

        # Mark layer 8 (decision point)
        if 8 in layers:
            idx8 = list(layers).index(8)
            ax1.axvline(x=idx8, color='red', linestyle='--', alpha=0.5, label='Layer 8')

    # Plot 2: Effect size by layer
    ax2 = axes[0, 1]
    if layers:
        effect_sizes = [comparison_results['by_layer'][l]['cohens_d'] for l in layers]
        colors = ['coral' if d > 0 else 'steelblue' for d in effect_sizes]
        ax2.bar(layers, effect_sizes, color=colors)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax2.axhline(y=0.5, color='gray', linestyle='-.', linewidth=0.5, alpha=0.5)
        ax2.axhline(y=0.8, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel("Cohen's d")
        ax2.set_title("Effect Size (Failures - Successes)")
        ax2.text(layers[-1], 0.2, 'small', fontsize=8, alpha=0.7)
        ax2.text(layers[-1], 0.5, 'medium', fontsize=8, alpha=0.7)
        ax2.text(layers[-1], 0.8, 'large', fontsize=8, alpha=0.7)

    # Plot 3: P-values by layer
    ax3 = axes[1, 0]
    if layers:
        p_values = [comparison_results['by_layer'][l]['p_value'] for l in layers]
        log_p = [-np.log10(max(p, 1e-10)) for p in p_values]
        ax3.bar(layers, log_p, color='purple', alpha=0.7)
        ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        ax3.axhline(y=-np.log10(0.01), color='orange', linestyle='--', label='p=0.01')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('-log10(p-value)')
        ax3.set_title('Statistical Significance by Layer')
        ax3.legend()

    # Plot 4: Aggregate comparison
    ax4 = axes[1, 1]
    if 'aggregate' in comparison_results and comparison_results['aggregate']:
        agg = comparison_results['aggregate']
        categories = ['H1 Failures\n(parent output)', 'H1 Successes\n(child output)']
        values = [agg['failure_mean'], agg['success_mean']]
        errors = [agg['failure_std'], agg['success_std']]
        colors = ['coral', 'steelblue']

        bars = ax4.bar(categories, values, yerr=errors, color=colors, capsize=5)
        ax4.set_ylabel('Mean Attention to Subsumption')
        p_str = f"p={agg['p_value']:.4f}" if agg['p_value'] >= 0.0001 else "p<0.0001"
        ax4.set_title(f'Aggregate Comparison ({p_str})')

        # Value labels
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    plt.close(fig)  # Close to free memory (no display in headless mode)


def visualize_attention_heatmap(
    layer_kv_analysis: Dict,
    config: AttentionAnalysisConfig,
    save_path: Optional[str] = None,
) -> None:
    """
    Create heatmap of attention differences by layer and KV group.

    Args:
        layer_kv_analysis: Output from analyze_layer_kv_patterns()
        config: Analysis configuration
        save_path: Path to save figure
    """
    by_component = layer_kv_analysis['by_component']
    layers = sorted(by_component.keys())
    kv_groups = list(range(config.num_kv_heads))

    # Build matrix
    diff_matrix = np.zeros((len(layers), len(kv_groups)))
    for i, layer in enumerate(layers):
        for j, kv_group in enumerate(kv_groups):
            if kv_group in by_component.get(layer, {}):
                diff_matrix[i, j] = by_component[layer][kv_group]['difference']

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        diff_matrix,
        xticklabels=[f'KV{g}' for g in kv_groups],
        yticklabels=[f'L{l}' for l in layers],
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.4f',
        cbar_kws={'label': 'Attention Difference (Failure - Success)'}
    )
    plt.xlabel('KV Group')
    plt.ylabel('Layer')
    plt.title('Attention Difference Heatmap: H1 Failures vs Successes\n(Positive = Failures attend more to subsumption)')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")

    plt.close()  # Close to free memory


def visualize_region_comparison(
    region_comparison: Dict,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize attention to different regions (subsumption, child, parent).

    Args:
        region_comparison: Output from compare_attention_to_regions()
        save_path: Path to save figure
    """
    regions = list(region_comparison.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(regions))
    width = 0.35

    failure_means = [region_comparison[r]['failure_mean'] for r in regions]
    success_means = [region_comparison[r]['success_mean'] for r in regions]

    ax.bar(x - width/2, failure_means, width, label='H1 Failures', color='coral')
    ax.bar(x + width/2, success_means, width, label='H1 Successes', color='steelblue')

    ax.set_xlabel('Region')
    ax.set_ylabel('Mean Attention')
    ax.set_title('Attention to Different Regions: Failures vs Successes')
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in regions])
    ax.legend()

    # Add p-values
    for i, region in enumerate(regions):
        p = region_comparison[region]['p_value']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        max_val = max(failure_means[i], success_means[i])
        ax.text(i, max_val + 0.002, sig, ha='center', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved region comparison to {save_path}")

    plt.close()  # Close to free memory


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_attention_analysis(
    model: Gemma2Wrapper,
    examples: List[ProbingExample],
    categories: Dict[str, List[int]],
    config: AttentionAnalysisConfig = None,
    max_per_category: int = None,
    verbose: bool = True,
) -> Dict:
    """
    Run full attention analysis pipeline.

    Args:
        model: Loaded Gemma2Wrapper
        examples: List of ProbingExample (H1 examples only)
        categories: Output from stratify_gemma_results()
        config: Analysis configuration
        max_per_category: Maximum examples per category (for testing)
        verbose: Print progress

    Returns:
        Dict with all analysis results
    """
    if config is None:
        config = AttentionAnalysisConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("ATTENTION ANALYSIS FOR ONTOLOGY REASONING")
    print("=" * 60)
    print(f"Hypothesis H3: H1 failures show higher attention to subsumption")
    print(f"Focus layers: {config.focus_layers}")
    print("=" * 60)

    # Filter to H1 examples only (we're testing H1 failure vs success)
    h1_examples = {e.idx: e for e in examples if e.h1_or_h2 == "h1"}

    # Get failure and success indices
    failure_indices = categories['h1_fail_h2_success'] + categories['h1_fail_h2_fail']
    success_indices = categories['h1_success_h2_fail'] + categories['h1_success_h2_success']

    if max_per_category:
        failure_indices = failure_indices[:max_per_category]
        success_indices = success_indices[:max_per_category]

    # Get examples
    failure_examples = [h1_examples[i] for i in failure_indices if i in h1_examples]
    success_examples = [h1_examples[i] for i in success_indices if i in h1_examples]

    print(f"\nData split:")
    print(f"  H1 failures (parent output): {len(failure_examples)}")
    print(f"  H1 successes (child output): {len(success_examples)}")

    # Extract attention patterns
    print("\nExtracting attention from H1 failures...")
    failure_results = extract_attention_batch(model, failure_examples, config, verbose)

    print("\nExtracting attention from H1 successes...")
    # Match sample size for fair comparison
    matched_success = success_examples[:len(failure_examples) * 5]  # 5x controls
    success_results = extract_attention_batch(model, matched_success, config, verbose)

    # Statistical comparison
    print("\nRunning statistical comparison...")
    comparison = compare_attention_patterns(failure_results, success_results, config)

    # Layer-KV analysis
    print("Analyzing layer-KV patterns...")
    layer_kv = analyze_layer_kv_patterns(failure_results, success_results, config)

    # Region comparison
    print("Comparing attention to different regions...")
    region_comparison = compare_attention_to_regions(failure_results, success_results, config)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS: Attention to Subsumption Sentence")
    print("=" * 60)

    print("\nBy Layer (even layers only - global attention):")
    print("-" * 60)
    for layer in sorted(comparison['by_layer'].keys()):
        r = comparison['by_layer'][layer]
        sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
        print(f"  Layer {layer:2d}: Fail={r['failure_mean']:.4f}, Succ={r['success_mean']:.4f}, "
              f"d={r['cohens_d']:+.3f}, p={r['p_value']:.4f} {sig}")

    if comparison.get('aggregate'):
        print(f"\nAggregate:")
        agg = comparison['aggregate']
        print(f"  Failures mean: {agg['failure_mean']:.4f}")
        print(f"  Successes mean: {agg['success_mean']:.4f}")
        print(f"  Difference: {agg['difference']:+.4f}")
        print(f"  p-value: {agg['p_value']:.4f}")

    print("\nTop Layer-KV Group Differences:")
    print("-" * 60)
    for item in layer_kv['top_differences'][:5]:
        sig = "***" if item['p_value'] < 0.001 else "**" if item['p_value'] < 0.01 else "*" if item['p_value'] < 0.05 else ""
        print(f"  Layer {item['layer']:2d}, KV {item['kv_group']}: "
              f"Diff={item['difference']:+.4f}, p={item['p_value']:.4f} {sig}")

    print("\nAttention to Different Regions:")
    print("-" * 60)
    for region, data in region_comparison.items():
        sig = "***" if data['p_value'] < 0.001 else "**" if data['p_value'] < 0.01 else "*" if data['p_value'] < 0.05 else ""
        print(f"  {region:12s}: Fail={data['failure_mean']:.4f}, Succ={data['success_mean']:.4f}, "
              f"Diff={data['difference']:+.4f} {sig}")

    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_attention_comparison(comparison, config, str(output_dir / "attention_comparison.png"))
    visualize_attention_heatmap(layer_kv, config, str(output_dir / "attention_heatmap.png"))
    visualize_region_comparison(region_comparison, str(output_dir / "region_comparison.png"))

    # Compile results
    results = {
        'config': config,
        'failure_results': failure_results,
        'success_results': success_results,
        'comparison': comparison,
        'layer_kv_analysis': layer_kv,
        'region_comparison': region_comparison,
        'n_failures': len(failure_results),
        'n_successes': len(success_results),
    }

    # Save results
    if config.save_intermediate:
        results_path = output_dir / "attention_analysis_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {results_path}")

    # Print interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    # Check if H3 is supported
    if comparison.get('aggregate'):
        agg = comparison['aggregate']
        if agg['difference'] > 0 and agg['p_value'] < 0.05:
            print("H3 SUPPORTED: Failures show significantly higher attention to subsumption")
            print("  -> The model 'notices' the subsumption relation more in failure cases")
            print("  -> This may trigger over-generalization to parent level")
        elif agg['difference'] < 0 and agg['p_value'] < 0.05:
            print("H3 REJECTED: Successes show higher attention to subsumption")
            print("  -> Attention to subsumption may PREVENT over-generalization")
        else:
            print("H3 INCONCLUSIVE: No significant difference in subsumption attention")
            print("  -> Consider: attention to other regions, SAE features, or later layers")

    return results
