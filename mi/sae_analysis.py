#!/usr/bin/env python3
"""
SAE Differential Feature Analysis for Layer 8

Identifies SAE features that distinguish parent output (gold) from child output (control)
using pre-cached activations from probing experiments.

Usage:
    python sae_analysis.py [--layer 8] [--width 16k]
"""

import os
import sys
import pickle
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_activation_cache(cache_path: str) -> list:
    """Load pre-extracted activations from probing experiments."""
    print(f"Loading activation cache from {cache_path}...")
    with open(cache_path, 'rb') as f:
        caches = pickle.load(f)
    print(f"Loaded {len(caches)} cached activations")
    return caches


def load_merged_data(results_path: str) -> Tuple[list, list, list, list]:
    """Load merged n2000 results to get gold indices."""
    print(f"Loading merged data from {results_path}...")
    with open(results_path, 'rb') as f:
        data = pickle.load(f)

    pairs = data['pairs']
    h1_results = data['h1_results']
    h2_results = data['h2_results']
    gold_indices = data['gold_indices']

    print(f"Found {len(gold_indices)} gold indices")
    return pairs, h1_results, h2_results, gold_indices


def get_control_indices(h1_results: list, h2_results: list, n_controls: int = 53, seed: int = 42) -> list:
    """
    Get control indices using same logic as probing script.
    Controls are h1_success + h2_fail pairs (child output behavior).
    """
    random.seed(seed)
    control_candidates = [
        i for i in range(len(h1_results))
        if h1_results[i]['strong'] == 1 and h2_results[i]['strong'] == 0
    ]
    control_indices = random.sample(control_candidates, min(n_controls, len(control_candidates)))
    print(f"Selected {len(control_indices)} control indices (seed={seed})")
    return control_indices


def load_sae(layer: int = 8, width: str = "16k", use_it: bool = True):
    """
    Load SAE from sae_lens.

    Args:
        layer: Target layer (default 8)
        width: SAE width (16k or 131k)
        use_it: Use IT-trained SAE (True) or PT-trained (False)

    Note: IT SAE (gemma-scope-9b-it-res-canonical) only has layers 9, 20, 31
          PT SAE (gemma-scope-9b-pt-res-canonical) has all layers 0-41
    """
    from sae_lens import SAE

    # IT SAE only has specific layers
    it_available_layers = {9, 20, 31}

    sae_id = f"layer_{layer}/width_{width}/canonical"

    if use_it and layer in it_available_layers:
        release = "gemma-scope-9b-it-res-canonical"
        print(f"Loading IT SAE: {release} / {sae_id}...")
        try:
            sae = SAE.from_pretrained(release=release, sae_id=sae_id)
            print(f"SAE loaded (IT): {sae.cfg.d_sae} features")
            return sae
        except Exception as e:
            print(f"Failed to load IT SAE: {e}")

    # Fall back to PT SAE (has all layers)
    release = "gemma-scope-9b-pt-res-canonical"
    print(f"Loading PT SAE: {release} / {sae_id}...")
    sae = SAE.from_pretrained(release=release, sae_id=sae_id)
    print(f"SAE loaded (PT): {sae.cfg.d_sae} features")
    return sae


def encode_activations_with_sae(
    caches: list,
    sae,
    layer: int,
    gold_indices: set,
    control_indices: set,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode cached activations through SAE.

    Groups by actual output (gold vs control), NOT by task type (h1 vs h2).
    Both H1 and H2 from same pair share behavior:
    - Gold indices: Both had parent output
    - Control indices: Both had child output
    """
    # Move SAE to device
    sae = sae.to(device)

    gold_features = []
    control_features = []

    print(f"Encoding {len(caches)} activations through SAE...")
    for cache in tqdm(caches, desc="Encoding"):
        # Get final position activation at target layer
        layer_acts = cache.layer_activations.get(layer, {})
        if 'final_position' not in layer_acts:
            print(f"Warning: No layer {layer} activation for example {cache.example_idx}")
            continue

        act = layer_acts['final_position']  # [1, 3584] or [3584]
        if act.ndim == 2:
            act = act.squeeze(0)
        act = act.float().to(device)  # [3584]

        # Encode with SAE
        with torch.no_grad():
            sae_act = sae.encode(act)  # [d_sae]

        # Group by actual output
        if cache.example_idx in gold_indices:
            gold_features.append(sae_act.cpu())
        elif cache.example_idx in control_indices:
            control_features.append(sae_act.cpu())

    if not gold_features:
        raise ValueError("No gold features found! Check gold_indices.")
    if not control_features:
        raise ValueError("No control features found! Check control_indices.")

    gold_tensor = torch.stack(gold_features)
    control_tensor = torch.stack(control_features)

    print(f"Gold features: {gold_tensor.shape}")
    print(f"Control features: {control_tensor.shape}")

    return gold_tensor, control_tensor


def find_differential_features(
    gold_features: torch.Tensor,
    control_features: torch.Tensor,
    top_k: int = 50
) -> Dict[str, Any]:
    """
    Find SAE features that distinguish gold (parent output) from control (child output).
    """
    # Mean activation per group
    gold_mean = gold_features.mean(dim=0)
    control_mean = control_features.mean(dim=0)

    # Differential activation (positive = more active for parent output)
    diff = gold_mean - control_mean

    # Top features for each direction
    parent_top = diff.topk(top_k)
    child_top = (-diff).topk(top_k)

    parent_features = parent_top.indices.tolist()
    child_features = child_top.indices.tolist()

    print(f"\nTop {top_k} parent-associated features: {parent_features[:10]}...")
    print(f"Top {top_k} child-associated features: {child_features[:10]}...")

    # Activation sparsity
    gold_sparsity = (gold_features == 0).float().mean().item()
    control_sparsity = (control_features == 0).float().mean().item()
    print(f"\nOverall sparsity - Gold: {gold_sparsity:.1%}, Control: {control_sparsity:.1%}")

    return {
        'parent_features': parent_features,
        'child_features': child_features,
        'diff_vector': diff,
        'gold_mean': gold_mean,
        'control_mean': control_mean,
        'parent_diffs': parent_top.values.tolist(),
        'child_diffs': child_top.values.tolist(),
    }


def validate_features_statistically(
    gold_features: torch.Tensor,
    control_features: torch.Tensor,
    feature_ids: List[int],
    alpha: float = 0.05,
    correction: str = "bonferroni"
) -> List[Dict]:
    """
    Statistical validation with multiple comparison correction.

    Args:
        correction: "bonferroni" (conservative) or "fdr" (less conservative)
    """
    results = []
    n_tests = len(feature_ids)

    if correction == "bonferroni":
        corrected_alpha = alpha / n_tests
    else:
        corrected_alpha = alpha  # Will apply FDR later

    for fid in feature_ids:
        gold_acts = gold_features[:, fid].numpy()
        control_acts = control_features[:, fid].numpy()

        # Welch's t-test (unequal variance)
        t_stat, p_value = stats.ttest_ind(gold_acts, control_acts, equal_var=False)

        # Cohen's d effect size
        pooled_std = np.sqrt((gold_acts.std()**2 + control_acts.std()**2) / 2)
        cohens_d = (gold_acts.mean() - control_acts.mean()) / pooled_std if pooled_std > 0 else 0

        # Activation statistics
        gold_nonzero = (gold_acts > 0).mean()
        control_nonzero = (control_acts > 0).mean()

        results.append({
            'feature_id': fid,
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < corrected_alpha,
            'cohens_d': cohens_d,
            'gold_mean': float(gold_acts.mean()),
            'control_mean': float(control_acts.mean()),
            'gold_sparsity': 1 - gold_nonzero,
            'control_sparsity': 1 - control_nonzero,
            'diff': float(gold_acts.mean() - control_acts.mean()),
        })

    # FDR correction (Benjamini-Hochberg)
    if correction == "fdr":
        p_values = np.array([r['p_value'] for r in results])
        sorted_idx = np.argsort(p_values)
        for rank, idx in enumerate(sorted_idx):
            threshold = alpha * (rank + 1) / n_tests
            results[idx]['significant'] = p_values[idx] <= threshold

    return results


def generate_neuronpedia_report(
    parent_stats: List[Dict],
    child_stats: List[Dict],
    layer: int = 8,
    width: str = "16k",
    output_path: str = "sae_analysis_report.md",
    n_gold: int = 0,
    n_control: int = 0
):
    """Generate markdown report with Neuronpedia links."""

    # Neuronpedia URL format for Gemma 2 9B
    base_url = "https://www.neuronpedia.org/gemma-2-9b"

    with open(output_path, 'w') as f:
        f.write("# SAE Differential Feature Analysis — Layer 8\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Dataset**: {n_gold} gold examples (parent output) vs {n_control} control examples (child output)\n\n")
        f.write(f"**SAE**: gemma-scope-9b-it-res / layer_{layer}/width_{width}/canonical\n\n")

        # Summary statistics
        n_parent_sig = sum(1 for s in parent_stats if s['significant'])
        n_child_sig = sum(1 for s in child_stats if s['significant'])
        f.write("## Summary\n\n")
        f.write(f"- Parent-associated features: **{n_parent_sig}/{len(parent_stats)}** significant (Bonferroni)\n")
        f.write(f"- Child-associated features: **{n_child_sig}/{len(child_stats)}** significant (Bonferroni)\n\n")

        # Parent features table
        f.write("---\n\n")
        f.write("## Parent-Associated Features\n\n")
        f.write("Features more active when model outputs parent concept.\n\n")
        f.write("| Rank | Feature | Cohen's d | Diff | p-value | Sig | Neuronpedia |\n")
        f.write("|------|---------|-----------|------|---------|-----|-------------|\n")

        for i, stat in enumerate(parent_stats[:20]):
            fid = stat['feature_id']
            sig = "✓" if stat['significant'] else ""
            url = f"{base_url}/{layer}-res-{width}/{fid}"
            f.write(f"| {i+1} | {fid} | {stat['cohens_d']:.2f} | {stat['diff']:.4f} | {stat['p_value']:.2e} | {sig} | [link]({url}) |\n")

        # Child features table
        f.write("\n---\n\n")
        f.write("## Child-Associated Features\n\n")
        f.write("Features more active when model outputs child concept.\n\n")
        f.write("| Rank | Feature | Cohen's d | Diff | p-value | Sig | Neuronpedia |\n")
        f.write("|------|---------|-----------|------|---------|-----|-------------|\n")

        for i, stat in enumerate(child_stats[:20]):
            fid = stat['feature_id']
            sig = "✓" if stat['significant'] else ""
            url = f"{base_url}/{layer}-res-{width}/{fid}"
            # Note: Cohen's d will be negative for child features (more active in control)
            f.write(f"| {i+1} | {fid} | {abs(stat['cohens_d']):.2f} | {stat['diff']:.4f} | {stat['p_value']:.2e} | {sig} | [link]({url}) |\n")

        # Interpretation guide
        f.write("\n---\n\n")
        f.write("## Interpretation Guide\n\n")
        f.write("- **Cohen's d**: Effect size (|d| > 0.8 is large, > 0.5 is medium, > 0.2 is small)\n")
        f.write("- **Diff**: Mean activation difference (gold - control)\n")
        f.write("- **Sig**: Significant after Bonferroni correction (α=0.05/40)\n")
        f.write("- Click Neuronpedia links to see feature interpretations and example activations\n")

    print(f"\nReport saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SAE Differential Feature Analysis")
    parser.add_argument('--layer', type=int, default=8, help='Target layer for SAE analysis')
    parser.add_argument('--width', type=str, default='16k', choices=['16k', '131k'], help='SAE width')
    parser.add_argument('--cache-path', default='mi/results_n2000/activation_cache_n2000.pkl')
    parser.add_argument('--results-path', default='benchmark/factorial_results/factorial_gemma29b_n2000_merged.pkl')
    parser.add_argument('--output-dir', default='mi/results_n2000')
    parser.add_argument('--top-k', type=int, default=50, help='Number of top features to analyze')
    parser.add_argument('--use-pt', action='store_true', help='Use PT SAE instead of IT SAE')
    parser.add_argument('--device', default='cuda', help='Device for SAE encoding')
    args = parser.parse_args()

    # Resolve paths
    project_dir = Path(__file__).parent.parent
    cache_path = project_dir / args.cache_path
    results_path = project_dir / args.results_path
    output_dir = project_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SAE DIFFERENTIAL FEATURE ANALYSIS")
    print("=" * 60)
    print(f"Layer: {args.layer}")
    print(f"Width: {args.width}")
    print(f"Device: {args.device}")

    # Step 1: Load data
    print("\n" + "-" * 40)
    print("STEP 1: Loading data")
    print("-" * 40)

    caches = load_activation_cache(str(cache_path))
    pairs, h1_results, h2_results, gold_indices = load_merged_data(str(results_path))
    control_indices = get_control_indices(h1_results, h2_results)

    gold_set = set(gold_indices)
    control_set = set(control_indices)

    # Step 2: Load SAE
    print("\n" + "-" * 40)
    print("STEP 2: Loading SAE")
    print("-" * 40)

    sae = load_sae(layer=args.layer, width=args.width, use_it=not args.use_pt)

    # Step 3: Encode activations
    print("\n" + "-" * 40)
    print("STEP 3: Encoding activations")
    print("-" * 40)

    gold_features, control_features = encode_activations_with_sae(
        caches, sae, args.layer, gold_set, control_set, device=args.device
    )

    # Step 4: Find differential features
    print("\n" + "-" * 40)
    print("STEP 4: Finding differential features")
    print("-" * 40)

    diff_results = find_differential_features(gold_features, control_features, top_k=args.top_k)

    # Step 5: Statistical validation
    print("\n" + "-" * 40)
    print("STEP 5: Statistical validation")
    print("-" * 40)

    all_features = diff_results['parent_features'][:20] + diff_results['child_features'][:20]

    parent_stats = validate_features_statistically(
        gold_features, control_features,
        diff_results['parent_features'][:20]
    )
    child_stats = validate_features_statistically(
        gold_features, control_features,
        diff_results['child_features'][:20]
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nTop 10 Parent-Associated Features:")
    print(f"{'Rank':<5} {'Feature':<10} {'Cohen d':<10} {'p-value':<12} {'Sig':<5}")
    print("-" * 45)
    for i, stat in enumerate(parent_stats[:10]):
        sig = "✓" if stat['significant'] else ""
        print(f"{i+1:<5} {stat['feature_id']:<10} {stat['cohens_d']:<10.2f} {stat['p_value']:<12.2e} {sig:<5}")

    print("\nTop 10 Child-Associated Features:")
    print(f"{'Rank':<5} {'Feature':<10} {'Cohen d':<10} {'p-value':<12} {'Sig':<5}")
    print("-" * 45)
    for i, stat in enumerate(child_stats[:10]):
        sig = "✓" if stat['significant'] else ""
        print(f"{i+1:<5} {stat['feature_id']:<10} {abs(stat['cohens_d']):<10.2f} {stat['p_value']:<12.2e} {sig:<5}")

    # Step 6: Save results
    print("\n" + "-" * 40)
    print("STEP 6: Saving results")
    print("-" * 40)

    # Save raw features
    features_path = output_dir / f"sae_features_layer{args.layer}.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump({
            'gold_features': gold_features,
            'control_features': control_features,
            'gold_indices': list(gold_set),
            'control_indices': list(control_set),
            'layer': args.layer,
            'width': args.width,
            'timestamp': datetime.now().isoformat(),
        }, f)
    print(f"Saved features to {features_path}")

    # Save differential analysis
    diff_path = output_dir / f"differential_features_layer{args.layer}.pkl"
    with open(diff_path, 'wb') as f:
        pickle.dump({
            'parent_features': diff_results['parent_features'],
            'child_features': diff_results['child_features'],
            'parent_stats': parent_stats,
            'child_stats': child_stats,
            'layer': args.layer,
            'width': args.width,
            'n_gold': len(gold_features),
            'n_control': len(control_features),
            'timestamp': datetime.now().isoformat(),
        }, f)
    print(f"Saved differential analysis to {diff_path}")

    # Generate report
    report_path = output_dir / f"sae_analysis_report_layer{args.layer}.md"
    generate_neuronpedia_report(
        parent_stats, child_stats,
        layer=args.layer, width=args.width,
        output_path=str(report_path),
        n_gold=len(gold_features),
        n_control=len(control_features)
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review the report: {report_path}")
    print(f"2. Click Neuronpedia links to interpret features")
    print(f"3. Look for patterns in significant features")


if __name__ == "__main__":
    main()
