#!/usr/bin/env python3
"""
Run Attention Analysis for Gemma 2 9B
======================================

Entry point for testing Hypothesis H3:
"H1 failures show higher attention to subsumption sentences"

Usage:
    python -m mi.run_attention_analysis [--test] [--max N] [--output-dir DIR]

Requirements:
    - CUDA GPU with >= 20GB VRAM (or use --4bit for 24GB GPUs)
    - benchmark/matched_pairs_set1_pure.pkl
    - benchmark/factorial_results/factorial_gemma29b.pkl
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mi.data_loader import (
    create_probing_dataset,
    stratify_gemma_results,
)
from mi.model_wrapper import Gemma2Wrapper
from mi.attention_analysis import (
    AttentionAnalysisConfig,
    run_attention_analysis,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run attention analysis to test H3: failures attend more to subsumption"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test with 5 examples per category",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum examples per category (failures and successes)",
    )
    parser.add_argument(
        "--4bit",
        action="store_true",
        dest="use_4bit",
        help="Use 4-bit quantization (for 24GB GPUs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mi/attention_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--pairs-path",
        type=str,
        default=None,
        help="Path to matched_pairs pickle (default: benchmark/matched_pairs_set1_pure.pkl)",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Path to factorial results pickle (default: benchmark/factorial_results/factorial_gemma29b.pkl)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="4,6,8,10,12,14,16,18,20",
        help="Comma-separated list of layers to analyze (default: 4,6,8,10,12,14,16,18,20)",
    )

    args = parser.parse_args()

    # Parse layers
    focus_layers = [int(l) for l in args.layers.split(",")]

    # Quick test mode
    if args.test:
        max_per_category = 5
        print("=" * 60)
        print("RUNNING IN TEST MODE (5 examples per category)")
        print("=" * 60)
    else:
        max_per_category = args.max

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Attention analysis requires GPU.")
        sys.exit(1)

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create config
    config = AttentionAnalysisConfig(
        focus_layers=focus_layers,
        output_dir=args.output_dir,
    )

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    examples = create_probing_dataset(
        pairs_path=args.pairs_path,
        results_path=args.results_path,
        set_key="set1",
    )
    print(f"Loaded {len(examples)} examples (H1 + H2)")

    categories = stratify_gemma_results(
        results_path=args.results_path,
        set_key="set1",
    )

    # Load model
    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    model = Gemma2Wrapper(use_4bit=args.use_4bit)

    # Run analysis
    print("\n")
    results = run_attention_analysis(
        model=model,
        examples=examples,
        categories=categories,
        config=config,
        max_per_category=max_per_category,
        verbose=True,
    )

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {config.output_dir}/")
    print(f"  - attention_comparison.png: Bar charts by layer")
    print(f"  - attention_heatmap.png: Layer x KV-group heatmap")
    print(f"  - region_comparison.png: Attention to different regions")
    print(f"  - attention_analysis_results.pkl: Full results")

    # Key findings summary
    if results.get('comparison', {}).get('aggregate'):
        agg = results['comparison']['aggregate']
        print(f"\nKey Finding:")
        print(f"  Subsumption attention difference: {agg['difference']:+.4f}")
        print(f"  Statistical significance: p={agg['p_value']:.4f}")

        if agg['p_value'] < 0.05:
            if agg['difference'] > 0:
                print("  -> H3 SUPPORTED: Failures attend MORE to subsumption")
            else:
                print("  -> H3 REJECTED: Successes attend MORE to subsumption")
        else:
            print("  -> H3 INCONCLUSIVE: No significant difference")

    return results


if __name__ == "__main__":
    main()
