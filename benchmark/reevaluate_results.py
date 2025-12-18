"""
Re-evaluate saved experiment results with corrected evaluation logic.
"""

import pickle
import json
import numpy as np
from pathlib import Path

# Monkey-patch OntologyNode to handle missing name attribute during unpickling
import ontology
original_hash = ontology.OntologyNode.__hash__
def safe_hash(self):
    if hasattr(self, 'name'):
        return hash(self.name)
    return id(self)
ontology.OntologyNode.__hash__ = safe_hash

from evaluate import (
    parse_hypotheses_from_response,
    parse_ground_truth,
    compute_strong_accuracy,
    compute_weak_accuracy,
    compute_quality,
    wilson_confidence_interval
)


def reevaluate_pkl(pkl_path, verbose=False):
    """Re-evaluate a saved experiment pickle file."""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  Warning: Could not load {pkl_path}: {e}")
        return None

    examples = data.get('examples', [])
    replies = data.get('replies', [])

    if not examples or not replies:
        print(f"  Warning: Missing data in {pkl_path}")
        return None

    results = []

    for i, (example, reply) in enumerate(zip(examples, replies)):
        # Handle different example formats
        if isinstance(example, dict):
            ontology = example.get('test', example)
        else:
            ontology = example

        pred_hyps = parse_hypotheses_from_response(reply)
        gt_hyps = parse_ground_truth(ontology.hypotheses)

        strong_acc = compute_strong_accuracy(pred_hyps, gt_hyps)
        weak_acc = compute_weak_accuracy(pred_hyps, gt_hyps, ontology.observations, ontology.theories)
        quality = compute_quality(pred_hyps, gt_hyps, ontology.observations, ontology.theories)

        if strong_acc == 1:
            weak_acc = 1
            quality = 1.0

        results.append({
            'strong': strong_acc,
            'weak': weak_acc,
            'quality': quality
        })

        if verbose and i < 3:
            print(f"\n  Example {i+1}:")
            print(f"    GT: {gt_hyps}")
            print(f"    Pred: {pred_hyps}")
            print(f"    Strong: {strong_acc}, Weak: {weak_acc}, Quality: {quality:.2f}")

    # Compute aggregate metrics
    strong_mean = np.mean([r['strong'] for r in results])
    weak_mean = np.mean([r['weak'] for r in results])
    quality_mean = np.mean([r['quality'] for r in results])

    n = len(results)
    strong_ci = wilson_confidence_interval(strong_mean, n)
    weak_ci = wilson_confidence_interval(weak_mean, n)

    return {
        'strong_accuracy': strong_mean,
        'weak_accuracy': weak_mean,
        'quality': quality_mean,
        'strong_ci': list(strong_ci),
        'weak_ci': list(weak_ci),
        'n': n
    }


def main():
    results_dir = Path('results')

    print("=" * 70)
    print("RE-EVALUATING EXPERIMENTS WITH CORRECTED EVALUATION")
    print("=" * 70)

    all_metrics = {}

    # Zero-shot single hypothesis
    print("\n--- Zero-shot Single Hypothesis ---")
    for task in ['property', 'membership', 'ontology']:
        all_metrics[f'zeroshot_single_{task}'] = {}
        print(f"\n{task.upper()}:")

        for height in range(1, 5):
            pkl_path = results_dir / f'zeroshot_single_{task}_h{height}_gpt4o.pkl'
            if pkl_path.exists():
                metrics = reevaluate_pkl(pkl_path, verbose=(height == 1))
                if metrics:
                    all_metrics[f'zeroshot_single_{task}'][f'h{height}'] = metrics
                    print(f"  H{height}: Strong={metrics['strong_accuracy']:.3f}, "
                          f"Weak={metrics['weak_accuracy']:.3f}, "
                          f"Quality={metrics['quality']:.3f}")
            else:
                print(f"  H{height}: File not found")

    # Zero-shot multiple hypothesis
    print("\n--- Zero-shot Multiple Hypotheses ---")
    all_metrics['zeroshot_multi'] = {}

    for height in range(1, 5):
        pkl_path = results_dir / f'zeroshot_multi_h{height}_gpt4o.pkl'
        if pkl_path.exists():
            metrics = reevaluate_pkl(pkl_path, verbose=(height == 1))
            if metrics:
                all_metrics['zeroshot_multi'][f'h{height}'] = metrics
                print(f"  H{height}: Strong={metrics['strong_accuracy']:.3f}, "
                      f"Weak={metrics['weak_accuracy']:.3f}, "
                      f"Quality={metrics['quality']:.3f}")
        else:
            print(f"  H{height}: File not found")

    # Save re-evaluated metrics
    output_file = results_dir / 'reevaluated_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved re-evaluated metrics to {output_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    print("\nZero-shot Single Hypothesis (Weak Accuracy):")
    print(f"{'Task':<12} {'H1':>10} {'H2':>10} {'H3':>10} {'H4':>10}")
    print("-" * 52)
    for task in ['property', 'membership', 'ontology']:
        row = f"{task:<12}"
        for h in range(1, 5):
            key = f'h{h}'
            if key in all_metrics.get(f'zeroshot_single_{task}', {}):
                val = all_metrics[f'zeroshot_single_{task}'][key]['weak_accuracy']
                row += f" {val:>9.3f}"
            else:
                row += f" {'N/A':>9}"
        print(row)

    print("\nZero-shot Multiple Hypotheses:")
    print(f"{'Metric':<15} {'H1':>10} {'H2':>10} {'H3':>10} {'H4':>10}")
    print("-" * 55)
    for metric in ['weak_accuracy', 'strong_accuracy', 'quality']:
        row = f"{metric:<15}"
        for h in range(1, 5):
            key = f'h{h}'
            if key in all_metrics.get('zeroshot_multi', {}):
                val = all_metrics['zeroshot_multi'][key][metric]
                row += f" {val:>9.3f}"
            else:
                row += f" {'N/A':>9}"
        print(row)


if __name__ == '__main__':
    main()
