#!/usr/bin/env python3
"""
Main experiment script for linear probing analysis.

Implements Phase 3 (Probing) from mi/mi_research_plan.md:
1. Load Gemma 2 9B IT
2. Run behavioral stratification to verify Conservation Law
3. Extract activations for all examples
4. Train probes at target layers
5. Save results and generate visualizations

Usage:
    python -m mi.run_probes

Or from the mi directory:
    python run_probes.py
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mi.model_wrapper import Gemma2Wrapper, load_gemma2_for_probing
from mi.data_loader import (
    create_probing_dataset,
    stratify_gemma_results,
    get_gold_pairs,
    ProbingExample
)
from mi.activation_extractor import (
    ActivationExtractor,
    MemoryEfficientExtractor,
    ActivationCache
)
from mi.probes.concept_level_probe import train_concept_level_probe
from mi.probes.output_prediction_probe import (
    train_output_prediction_probe,
    run_layer_sweep as run_output_sweep
)
from mi.probes.subsumption_probe import (
    train_subsumption_probe,
    run_subsumption_sweep
)


def get_default_paths() -> Dict[str, Path]:
    """Get default file paths relative to project root."""
    mi_dir = Path(__file__).parent
    project_root = mi_dir.parent

    return {
        'pairs_path': project_root / "benchmark" / "matched_pairs_set1_pure.pkl",
        'results_path': project_root / "benchmark" / "factorial_results" / "factorial_gemma29b.pkl",
        'output_dir': mi_dir / "results",
    }


def run_behavioral_stratification(
    results_path: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, List[int]]:
    """
    Run behavioral stratification to verify Conservation Law.

    Args:
        results_path: Path to factorial results
        verbose: Whether to print summary

    Returns:
        Category dict from stratify_gemma_results
    """
    paths = get_default_paths()
    results_path = results_path or paths['results_path']

    print("\n" + "=" * 60)
    print("PHASE 1: BEHAVIORAL STRATIFICATION")
    print("=" * 60)

    categories = stratify_gemma_results(
        results_path=str(results_path),
        verbose=verbose
    )

    return categories


def extract_activations(
    model_wrapper: Gemma2Wrapper,
    examples: List[ProbingExample],
    output_path: Optional[Path] = None,
    use_nnsight: bool = True,
    chunk_size: int = 10,
) -> List[ActivationCache]:
    """
    Extract activations for all examples.

    Args:
        model_wrapper: Loaded Gemma2Wrapper
        examples: List of ProbingExample
        output_path: Optional path to save activations
        use_nnsight: Whether to use nnsight (True) or HF hidden_states (False)
        chunk_size: Process this many examples before clearing memory

    Returns:
        List of ActivationCache
    """
    print("\n" + "=" * 60)
    print("PHASE 2: ACTIVATION EXTRACTION")
    print("=" * 60)
    print(f"Extracting activations for {len(examples)} examples...")
    print(f"Target layers: {model_wrapper.target_layers}")
    print(f"Using nnsight: {use_nnsight}")

    extractor = MemoryEfficientExtractor(
        model_wrapper,
        chunk_size=chunk_size,
        save_path=str(output_path) if output_path else None
    )

    caches = extractor.extract_all(examples, use_nnsight=use_nnsight)

    print(f"Extracted {len(caches)} activation caches")

    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(caches, f)
        print(f"Saved to {output_path}")

    return caches


def run_all_probes(
    activation_caches: List[ActivationCache],
    examples: List[ProbingExample],
    target_layers: List[int] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run all three probes across target layers.

    Args:
        activation_caches: Extracted activations
        examples: Original examples with behavioral data
        target_layers: Layers to probe
        output_dir: Directory to save results

    Returns:
        Dict with all probe results
    """
    target_layers = target_layers or [8, 15, 20, 25, 30, 35, 40]

    results = {
        'timestamp': datetime.now().isoformat(),
        'n_examples': len(examples),
        'n_caches': len(activation_caches),
        'target_layers': target_layers,
        'probes': {},
    }

    # Probe 1: Concept Level
    print("\n" + "=" * 60)
    print("PROBE 1: CONCEPT LEVEL (child vs parent)")
    print("=" * 60)

    concept_results = {}
    for layer in target_layers:
        probe, layer_result = train_concept_level_probe(
            activation_caches, layer, verbose=True
        )
        if 'error' not in layer_result:
            concept_results[layer] = {
                'accuracy': layer_result['accuracy'],
                'balanced_accuracy': layer_result['balanced_accuracy'],
                'interpretation': layer_result.get('interpretation', ''),
            }

    results['probes']['concept_level'] = concept_results

    # Probe 2: Output Prediction
    print("\n" + "=" * 60)
    print("PROBE 2: OUTPUT PREDICTION (when is decision made?)")
    print("=" * 60)

    output_results = run_output_sweep(
        activation_caches, examples, target_layers, verbose=True
    )
    results['probes']['output_prediction'] = {
        'layer_results': output_results['layer_results'],
        'interpretation': output_results['interpretation'],
    }

    # Probe 3: Subsumption
    print("\n" + "=" * 60)
    print("PROBE 3: SUBSUMPTION (is ontology encoded?)")
    print("=" * 60)

    subsumption_results = run_subsumption_sweep(
        activation_caches, examples, target_layers, verbose=True
    )
    results['probes']['subsumption'] = {
        'layer_results': subsumption_results['layer_results'],
        'all_results': {
            k: {kk: vv for kk, vv in v.items() if kk != 'classification_report'}
            for k, v in subsumption_results['all_results'].items()
        },
    }

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON (for easy viewing)
        json_path = output_dir / "probe_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {json_path}")

        # Save pickle (for full data)
        pkl_path = output_dir / "probe_results.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)

    return results


def plot_results(
    results: Dict[str, Any],
    output_dir: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Generate visualization of probe results.

    Args:
        results: Results dict from run_all_probes
        output_dir: Directory to save plots
        show: Whether to display plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Concept Level
    ax1 = axes[0]
    concept = results['probes'].get('concept_level', {})
    if concept:
        layers = sorted(concept.keys())
        accs = [concept[l]['accuracy'] for l in layers]
        ax1.plot(layers, accs, 'o-', linewidth=2, markersize=8)
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Chance')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Probe 1: Concept Level\n(child vs parent)')
        ax1.set_ylim(0.4, 1.0)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: Output Prediction
    ax2 = axes[1]
    output = results['probes'].get('output_prediction', {}).get('layer_results', {})
    if output:
        layers = sorted(output.keys())
        accs = [output[l] for l in layers]
        ax2.plot(layers, accs, 'o-', linewidth=2, markersize=8, color='green')
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Chance')

        # Mark peak
        peak_layer = layers[np.argmax(accs)]
        peak_acc = max(accs)
        ax2.axvline(x=peak_layer, color='orange', linestyle=':', alpha=0.7,
                    label=f'Peak: Layer {peak_layer}')

        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Probe 2: Output Prediction\n(when is decision made?)')
        ax2.set_ylim(0.4, 1.0)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Subsumption
    ax3 = axes[2]
    subsumption = results['probes'].get('subsumption', {}).get('layer_results', {})
    all_sub = results['probes'].get('subsumption', {}).get('all_results', {})
    if subsumption:
        layers = sorted(subsumption.keys())
        accs = [subsumption[l] for l in layers]

        # Get chance level (from first result with it)
        chance = 0.5
        for l in layers:
            if l in all_sub and 'chance_level' in all_sub[l]:
                chance = all_sub[l]['chance_level']
                break

        ax3.plot(layers, accs, 'o-', linewidth=2, markersize=8, color='purple')
        ax3.axhline(y=chance, color='r', linestyle='--', alpha=0.5,
                    label=f'Chance ({chance:.2f})')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Probe 3: Subsumption\n(is ontology encoded?)')
        ax3.set_ylim(0, 1.0)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        plot_path = output_dir / "probe_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")

    if show:
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run linear probing experiments")
    parser.add_argument('--pairs-path', type=str, help='Path to matched pairs pickle')
    parser.add_argument('--results-path', type=str, help='Path to factorial results pickle')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--use-4bit', action='store_true', help='Use 4-bit quantization')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip extraction, load from cache')
    parser.add_argument('--cache-path', type=str, help='Path to activation cache')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--layers', type=str, default='8,15,20,25,30,35,40',
                        help='Comma-separated list of layers to probe')
    parser.add_argument('--no-nnsight', action='store_true',
                        help='Use HuggingFace hidden_states instead of nnsight')

    args = parser.parse_args()

    # Get paths
    paths = get_default_paths()
    pairs_path = Path(args.pairs_path) if args.pairs_path else paths['pairs_path']
    results_path = Path(args.results_path) if args.results_path else paths['results_path']
    output_dir = Path(args.output_dir) if args.output_dir else paths['output_dir']
    target_layers = [int(l) for l in args.layers.split(',')]

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 3: LINEAR PROBING EXPERIMENT")
    print("=" * 60)
    print(f"Pairs path: {pairs_path}")
    print(f"Results path: {results_path}")
    print(f"Output dir: {output_dir}")
    print(f"Target layers: {target_layers}")
    print(f"4-bit quantization: {args.use_4bit}")

    # Step 1: Behavioral stratification
    categories = run_behavioral_stratification(results_path)
    gold_indices = categories['h1_fail_h2_success']
    print(f"\nGold pairs (H1 fail + H2 success): {gold_indices}")

    # Step 2: Load data
    print("\nLoading probing dataset...")
    examples = create_probing_dataset(
        pairs_path=str(pairs_path),
        results_path=str(results_path)
    )
    print(f"Loaded {len(examples)} examples")

    # Step 3: Extract or load activations
    cache_path = Path(args.cache_path) if args.cache_path else output_dir / "activation_cache.pkl"

    if args.skip_extraction and cache_path.exists():
        print(f"\nLoading activation cache from {cache_path}...")
        with open(cache_path, 'rb') as f:
            activation_caches = pickle.load(f)
        print(f"Loaded {len(activation_caches)} caches")
    else:
        print("\nLoading Gemma 2 9B IT...")
        model_wrapper = load_gemma2_for_probing(use_4bit=args.use_4bit)

        activation_caches = extract_activations(
            model_wrapper, examples,
            output_path=cache_path,
            use_nnsight=not args.no_nnsight
        )

    # Step 4: Run probes
    results = run_all_probes(
        activation_caches, examples,
        target_layers=target_layers,
        output_dir=output_dir
    )

    # Step 5: Plot results
    if not args.no_plot:
        plot_results(results, output_dir=output_dir, show=False)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Output prediction summary
    output_interp = results['probes'].get('output_prediction', {}).get('interpretation', {})
    if output_interp:
        print(f"\nOutput Prediction Probe:")
        print(f"  Peak layer: {output_interp.get('peak_layer', 'N/A')}")
        print(f"  Peak accuracy: {output_interp.get('peak_accuracy', 0):.3f}")
        print(f"  Timing: {output_interp.get('timing', 'N/A')}")
        print(f"  {output_interp.get('description', '')}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
