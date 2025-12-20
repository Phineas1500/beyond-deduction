#!/usr/bin/env python3
"""
Run probing experiments on the n=2000 merged dataset.

This adapts the existing run_probes.py infrastructure to work with the
new merged results format from factorial_gemma29b_n2000_merged.pkl.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))

from mi.model_wrapper import load_gemma2_for_probing
from mi.activation_extractor import ActivationExtractor, MemoryEfficientExtractor, ActivationCache
from mi.tokenizer_utils import TokenPositionFinder


# System prompt used in factorial experiments
SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
Only output final hypotheses."""


@dataclass
class ProbingExampleN2000:
    """Probing example adapted for n2000 merged format."""
    idx: int
    h1_or_h2: str
    seed: int
    prompt: str
    theories_nl: str
    observations_nl: str
    child_concept: str
    root_concept: str
    entities: List[str]
    property_name: str
    is_negated: bool
    ground_truth: str
    depth: int
    model_correct: Optional[bool]
    model_response: Optional[str]


def format_prompt(theories_nl: str, observations_nl: str) -> str:
    """Format prompt for Gemma 2 IT."""
    user_prompt = f"Q: {theories_nl} We observe that: {observations_nl} Please come up with hypothesis to explain observations."
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}"


def load_n2000_merged(results_path: str, pairs_path: str = None) -> tuple:
    """
    Load the n2000 merged results.

    Returns:
        (pairs, h1_results, h2_results, gold_indices)
    """
    with open(results_path, 'rb') as f:
        data = pickle.load(f)

    pairs = data['pairs']
    h1_results = data['h1_results']
    h2_results = data['h2_results']
    gold_indices = data['gold_indices']

    return pairs, h1_results, h2_results, gold_indices


def create_probing_examples_n2000(
    pairs: list,
    h1_results: list,
    h2_results: list,
    indices: List[int] = None
) -> List[ProbingExampleN2000]:
    """
    Create probing examples from n2000 merged data.

    Args:
        pairs: List of (h1_example, h2_example) tuples
        h1_results: List of H1 result dicts
        h2_results: List of H2 result dicts
        indices: Optional subset of indices to include
    """
    examples = []

    if indices is None:
        indices = range(len(pairs))

    for idx in indices:
        h1_ex, h2_ex = pairs[idx]
        h1_res = h1_results[idx]
        h2_res = h2_results[idx]

        for example, h_type, result in [(h1_ex, "h1", h1_res), (h2_ex, "h2", h2_res)]:
            prompt = format_prompt(example['theories_nl'], example['observations_nl'])

            prop = example['property']
            if hasattr(prop, 'name'):
                property_name = prop.name
            else:
                property_name = str(prop)

            probing_ex = ProbingExampleN2000(
                idx=idx,
                h1_or_h2=h_type,
                seed=example['seed'],
                prompt=prompt,
                theories_nl=example['theories_nl'],
                observations_nl=example['observations_nl'],
                child_concept=example['child_concept'],
                root_concept=example['root_concept'],
                entities=example['entities'],
                property_name=property_name,
                is_negated=example.get('is_negated', False),
                ground_truth=example['gt_hypothesis_nl'],
                depth=example['depth'],
                model_correct=bool(result['strong']),
                model_response=result.get('reply', '')
            )
            examples.append(probing_ex)

    return examples


def extract_activations_for_examples(
    model_wrapper,
    examples: List[ProbingExampleN2000],
    target_layers: List[int] = None,
    chunk_size: int = 10,
    save_path: str = None
) -> List[ActivationCache]:
    """Extract activations using existing infrastructure."""

    target_layers = target_layers or [8, 15, 20, 25, 30, 35, 40]

    print(f"\nExtracting activations for {len(examples)} examples...")
    print(f"Target layers: {target_layers}")

    extractor = ActivationExtractor(model_wrapper, target_layers=target_layers)

    caches = []

    from tqdm import tqdm
    for i, example in enumerate(tqdm(examples, desc="Extracting")):
        # Find positions
        positions = extractor.position_finder.get_positions_for_probing(
            example.prompt,
            example.child_concept,
            example.root_concept
        )

        # Extract activations
        activations = extractor.extract_activations_hf(example.prompt, positions, target_layers)

        cache = ActivationCache(
            example_idx=example.idx,
            h1_or_h2=example.h1_or_h2,
            layer_activations=activations,
            positions=positions
        )
        caches.append(cache)

        # Clear memory periodically
        if (i + 1) % chunk_size == 0:
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(caches, f)
        print(f"Saved {len(caches)} caches to {save_path}")

    return caches


def run_output_prediction_probe(
    caches: List[ActivationCache],
    examples: List[ProbingExampleN2000],
    target_layers: List[int],
    n_seeds: int = 10
) -> Dict[str, Any]:
    """Run output prediction probe with multi-seed evaluation."""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    # Build lookup
    example_lookup = {(ex.idx, ex.h1_or_h2): ex for ex in examples}
    LABEL_MAP = {'will_output_child': 0, 'will_output_parent': 1}

    def prepare_data_for_layer(layer_idx):
        X_list = []
        y_list = []

        for cache in caches:
            layer_acts = cache.layer_activations.get(layer_idx, {})
            if 'final_position' not in layer_acts:
                continue

            final_act = layer_acts['final_position']
            if hasattr(final_act, 'float'):
                final_act = final_act.float().numpy()
            elif hasattr(final_act, 'numpy'):
                final_act = final_act.numpy()

            if final_act.ndim == 2:
                final_act = final_act[0]

            example = example_lookup.get((cache.example_idx, cache.h1_or_h2))
            if example is None or not example.model_response:
                continue

            response = example.model_response.lower()
            child = example.child_concept.lower()
            parent = example.root_concept.lower()

            if parent in response:
                label = LABEL_MAP['will_output_parent']
            elif child in response:
                label = LABEL_MAP['will_output_child']
            else:
                continue

            X_list.append(final_act)
            y_list.append(label)

        return np.array(X_list), np.array(y_list)

    def evaluate_with_seeds(X, y, n_seeds=10):
        balanced_accs = []
        parent_recalls = []

        for seed in range(n_seeds):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=seed
                )
            except ValueError:
                continue

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = LogisticRegressionCV(
                cv=5,
                class_weight='balanced',
                max_iter=2000,
                random_state=seed
            )
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)

            bal_acc = balanced_accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            if cm.shape[0] > 1:
                parent_recall = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
            else:
                parent_recall = 0

            balanced_accs.append(bal_acc)
            parent_recalls.append(parent_recall)

        return {
            'bal_acc_mean': np.mean(balanced_accs) if balanced_accs else 0,
            'bal_acc_std': np.std(balanced_accs) if balanced_accs else 0,
            'parent_recall_mean': np.mean(parent_recalls) if parent_recalls else 0,
            'parent_recall_std': np.std(parent_recalls) if parent_recalls else 0,
        }

    # Get class distribution
    X, y = prepare_data_for_layer(target_layers[0])
    n_child = np.sum(y == 0)
    n_parent = np.sum(y == 1)

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    print(f"Will output child:  {n_child}")
    print(f"Will output parent: {n_parent}")
    print(f"Total: {len(y)}")
    print(f"Trivial baseline: {n_child / len(y) * 100:.1f}%")

    print("\n" + "=" * 60)
    print(f"MULTI-SEED ANALYSIS ({n_seeds} seeds per layer)")
    print("=" * 60)
    print(f"{'Layer':>6} | {'Balanced Acc':>15} | {'Parent Recall':>15}")
    print("-" * 45)

    layer_results = {}
    for layer_idx in target_layers:
        X, y = prepare_data_for_layer(layer_idx)
        results = evaluate_with_seeds(X, y, n_seeds=n_seeds)
        layer_results[layer_idx] = results

        bal_str = f"{results['bal_acc_mean']:.1%} ± {results['bal_acc_std']:.1%}"
        par_str = f"{results['parent_recall_mean']:.1%} ± {results['parent_recall_std']:.1%}"
        print(f"{layer_idx:>6} | {bal_str:>15} | {par_str:>15}")

    return {
        'layer_results': layer_results,
        'n_child': n_child,
        'n_parent': n_parent,
    }


def main():
    parser = argparse.ArgumentParser(description="Run probing on n2000 merged dataset")
    parser.add_argument('--results-path', default='benchmark/factorial_results/factorial_gemma29b_n2000_merged.pkl')
    parser.add_argument('--output-dir', default='mi/results_n2000')
    parser.add_argument('--use-4bit', action='store_true', help='Use 4-bit quantization')
    parser.add_argument('--skip-extraction', action='store_true', help='Load cached activations')
    parser.add_argument('--cache-path', default=None, help='Path to activation cache')
    parser.add_argument('--gold-only', action='store_true', help='Only process gold pairs + matched controls')
    parser.add_argument('--n-controls', type=int, default=53, help='Number of control pairs')
    parser.add_argument('--layers', type=str, default='8,15,20,25,30,35,40')
    parser.add_argument('--n-seeds', type=int, default=10, help='Number of random seeds for evaluation')
    args = parser.parse_args()

    # Resolve paths
    project_dir = Path(__file__).parent.parent
    results_path = project_dir / args.results_path
    output_dir = project_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target_layers = [int(l) for l in args.layers.split(',')]

    cache_path = Path(args.cache_path) if args.cache_path else output_dir / "activation_cache_n2000.pkl"

    print("=" * 60)
    print("PROBING EXPERIMENT - N=2000 DATASET")
    print("=" * 60)
    print(f"Results path: {results_path}")
    print(f"Output dir: {output_dir}")
    print(f"Target layers: {target_layers}")
    print(f"Gold only: {args.gold_only}")

    # Load data
    print("\nLoading n2000 merged results...")
    pairs, h1_results, h2_results, gold_indices = load_n2000_merged(str(results_path))
    print(f"Loaded {len(pairs)} pairs, {len(gold_indices)} gold indices")

    # Select indices
    if args.gold_only:
        # Gold + matched controls
        import random
        random.seed(42)
        control_indices = [i for i in range(len(h1_results))
                          if h1_results[i]['strong'] == 1 and h2_results[i]['strong'] == 0]
        control_indices = random.sample(control_indices, min(args.n_controls, len(control_indices)))
        selected_indices = list(gold_indices) + control_indices
        print(f"Selected {len(gold_indices)} gold + {len(control_indices)} controls = {len(selected_indices)} pairs")
    else:
        selected_indices = None  # Use all

    # Create examples
    examples = create_probing_examples_n2000(pairs, h1_results, h2_results, selected_indices)
    print(f"Created {len(examples)} probing examples")

    # Extract or load activations
    if args.skip_extraction and cache_path.exists():
        print(f"\nLoading cached activations from {cache_path}...")
        with open(cache_path, 'rb') as f:
            caches = pickle.load(f)
        print(f"Loaded {len(caches)} caches")
    else:
        print("\nLoading Gemma 2 9B IT...")
        model_wrapper = load_gemma2_for_probing(use_4bit=args.use_4bit)

        caches = extract_activations_for_examples(
            model_wrapper, examples,
            target_layers=target_layers,
            save_path=str(cache_path)
        )

    # Run probing
    print("\n" + "=" * 60)
    print("OUTPUT PREDICTION PROBE")
    print("=" * 60)

    probe_results = run_output_prediction_probe(
        caches, examples, target_layers, n_seeds=args.n_seeds
    )

    # Save results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'n_examples': len(examples),
        'n_caches': len(caches),
        'target_layers': target_layers,
        'gold_only': args.gold_only,
        'n_gold': len(gold_indices),
        'probe_results': probe_results,
    }

    results_file = output_dir / "probe_results_n2000.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved results to {results_file}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    layer_results = probe_results['layer_results']
    mean_bal_accs = [r['bal_acc_mean'] for r in layer_results.values()]

    if min(mean_bal_accs) > 0.85:
        print(f"ROBUST FINDING: Balanced accuracy >{min(mean_bal_accs):.0%} at ALL layers")
        print("Decision is locked in from early layers onwards.")
    elif min(mean_bal_accs) > 0.70:
        print(f"MODERATE FINDING: Balanced accuracy {min(mean_bal_accs):.0%}-{max(mean_bal_accs):.0%}")
    else:
        print(f"WEAK FINDING: Balanced accuracy varies {min(mean_bal_accs):.0%}-{max(mean_bal_accs):.0%}")

    early_acc = layer_results[8]['bal_acc_mean']
    late_acc = layer_results[40]['bal_acc_mean']
    print(f"\nLayer 8 → Layer 40: {early_acc:.1%} → {late_acc:.1%}")

    if abs(early_acc - late_acc) < 0.05:
        print("Flat trajectory: decision is made by layer 8 and maintained.")
    elif late_acc > early_acc + 0.1:
        print("Increasing trajectory: decision crystallizes in later layers.")
    else:
        print("Decreasing trajectory: early representation is most predictive.")


if __name__ == "__main__":
    main()
