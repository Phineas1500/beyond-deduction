"""
Probe 3: Does the model encode ontology structure (subsumption)?

Input: Activation at child concept token
Label: 'has_parent_X' (e.g., 'has_parent_rompus')
Question: Does the model represent "X is a Y" relationships?

This is a harder, multi-class probe that tests whether the model
internally represents the ontology structure.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Set, Optional

from .base_probe import BaseLinearProbe
from ..activation_extractor import ActivationCache
from ..data_loader import ProbingExample


class SubsumptionProbe(BaseLinearProbe):
    """
    Probe to determine if child concepts encode their parent.

    This is a multi-class classification problem where each unique
    parent concept is a class.

    Note: This may require many examples to learn reliably given
    the high dimensionality (hidden_size=3584) and number of classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parent_to_label: Dict[str, int] = {}
        self.label_to_parent: Dict[int, str] = {}

    def prepare_data(
        self,
        activation_caches: List[ActivationCache],
        examples: List[ProbingExample],
        layer_idx: int,
        min_samples_per_class: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare multi-class data for subsumption prediction.

        Each unique parent concept becomes a class.
        Labels are assigned to child concept activations based on
        their corresponding parent in the ontology.

        Args:
            activation_caches: List of ActivationCache
            examples: List of ProbingExample with concept info
            layer_idx: Layer to probe
            min_samples_per_class: Minimum samples needed per class

        Returns:
            X: (n_samples, hidden_size)
            y: (n_samples,) with integer class labels
        """
        # Create lookup: (idx, h1_or_h2) -> example
        example_lookup = {(e.idx, e.h1_or_h2): e for e in examples}

        # First pass: collect parent concept counts
        parent_counts: Dict[str, int] = {}

        for cache in activation_caches:
            example = example_lookup.get((cache.example_idx, cache.h1_or_h2))
            if example is None:
                continue

            layer_acts = cache.layer_activations.get(layer_idx, {})
            if 'child_positions' not in layer_acts:
                continue

            child_acts = layer_acts['child_positions']
            if hasattr(child_acts, 'float'):
                child_acts = child_acts.float().numpy()  # Convert bfloat16 to float32
            elif hasattr(child_acts, 'numpy'):
                child_acts = child_acts.numpy()
            if child_acts.ndim == 1:
                child_acts = child_acts.reshape(1, -1)

            parent = example.root_concept.lower()
            parent_counts[parent] = parent_counts.get(parent, 0) + len(child_acts)

        # Filter parents with enough samples
        valid_parents = [p for p, c in parent_counts.items()
                         if c >= min_samples_per_class]

        if len(valid_parents) < 2:
            # Need at least 2 classes for classification
            return np.array([]), np.array([])

        # Create label mapping for valid parents only
        self.parent_to_label = {p: i for i, p in enumerate(sorted(valid_parents))}
        self.label_to_parent = {i: p for p, i in self.parent_to_label.items()}

        # Second pass: build dataset
        X_list = []
        y_list = []

        for cache in activation_caches:
            example = example_lookup.get((cache.example_idx, cache.h1_or_h2))
            if example is None:
                continue

            layer_acts = cache.layer_activations.get(layer_idx, {})
            if 'child_positions' not in layer_acts:
                continue

            parent = example.root_concept.lower()
            if parent not in self.parent_to_label:
                continue  # Skip parents with too few samples

            parent_label = self.parent_to_label[parent]

            child_acts = layer_acts['child_positions']
            if hasattr(child_acts, 'float'):
                child_acts = child_acts.float().numpy()  # Convert bfloat16 to float32
            elif hasattr(child_acts, 'numpy'):
                child_acts = child_acts.numpy()
            if child_acts.ndim == 1:
                child_acts = child_acts.reshape(1, -1)

            for act in child_acts:
                X_list.append(act)
                y_list.append(parent_label)

        if len(X_list) == 0:
            return np.array([]), np.array([])

        return np.array(X_list), np.array(y_list)

    def get_class_info(self) -> Dict[str, Any]:
        """
        Get information about the learned classes.

        Returns:
            Dict with n_classes, parent_names, label_mapping
        """
        return {
            'n_classes': len(self.parent_to_label),
            'parent_names': list(self.parent_to_label.keys()),
            'label_mapping': self.parent_to_label,
        }

    def interpret_accuracy(self, accuracy: float, n_classes: int) -> str:
        """
        Interpret probe accuracy relative to chance level.

        Args:
            accuracy: Probe accuracy
            n_classes: Number of classes

        Returns:
            Human-readable interpretation
        """
        chance = 1.0 / n_classes if n_classes > 0 else 0

        if accuracy <= chance + 0.05:
            return f"At chance ({chance:.1%}) - subsumption NOT encoded"
        elif accuracy <= chance + 0.15:
            return f"Slightly above chance - weak subsumption encoding"
        elif accuracy <= chance + 0.30:
            return f"Moderately above chance - some subsumption information"
        else:
            return f"Well above chance ({chance:.1%}) - subsumption IS encoded"


def train_subsumption_probe(
    activation_caches: List[ActivationCache],
    examples: List[ProbingExample],
    layer_idx: int,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[Optional[SubsumptionProbe], Dict[str, Any]]:
    """
    Train a subsumption probe on activation caches.

    Args:
        activation_caches: List of ActivationCache
        examples: List of ProbingExample
        layer_idx: Layer to probe
        test_size: Test set fraction
        random_state: Random seed
        verbose: Whether to print results

    Returns:
        (fitted_probe, results_dict)
    """
    from sklearn.model_selection import train_test_split

    probe = SubsumptionProbe(random_state=random_state)
    X, y = probe.prepare_data(activation_caches, examples, layer_idx)

    if len(X) < 10:
        if verbose:
            print(f"  Layer {layer_idx}: Too few samples ({len(X)})")
        return None, {'error': f'Too few samples: {len(X)}', 'layer_idx': layer_idx}

    n_classes = len(probe.parent_to_label)
    if n_classes < 2:
        if verbose:
            print(f"  Layer {layer_idx}: Too few classes ({n_classes})")
        return None, {'error': f'Too few classes: {n_classes}', 'layer_idx': layer_idx}

    # Check minimum samples per class for stratified split
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    if min_count < 2:
        if verbose:
            print(f"  Layer {layer_idx}: Some classes have < 2 samples")
        return None, {
            'error': 'Some classes have < 2 samples',
            'layer_idx': layer_idx
        }

    # Split (may need to reduce test_size for small datasets)
    effective_test_size = min(test_size, 0.5)  # Don't use more than 50% for test

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=effective_test_size, stratify=y, random_state=random_state
        )
    except ValueError as e:
        if verbose:
            print(f"  Layer {layer_idx}: Stratified split failed: {e}")
        return None, {'error': str(e), 'layer_idx': layer_idx}

    # Fit
    probe.fit(X_train, y_train)

    # Evaluate
    results = probe.evaluate(X_test, y_test)
    results['n_train'] = len(X_train)
    results['n_test'] = len(X_test)
    results['n_classes'] = n_classes
    results['chance_level'] = 1.0 / n_classes
    results['layer_idx'] = layer_idx
    results['parent_names'] = list(probe.parent_to_label.keys())
    results['interpretation'] = probe.interpret_accuracy(
        results['accuracy'], n_classes
    )

    if verbose:
        print(f"  Layer {layer_idx}: Accuracy={results['accuracy']:.3f} "
              f"(chance={results['chance_level']:.3f}, n_classes={n_classes})")
        print(f"    {results['interpretation']}")

    return probe, results


def run_subsumption_sweep(
    activation_caches: List[ActivationCache],
    examples: List[ProbingExample],
    target_layers: List[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run subsumption probe across multiple layers.

    Args:
        activation_caches: List of ActivationCache
        examples: List of ProbingExample
        target_layers: Layers to probe
        test_size: Test set fraction
        random_state: Random seed
        verbose: Whether to print progress

    Returns:
        Dict with layer_results, probes
    """
    if target_layers is None:
        target_layers = [8, 15, 20, 25, 30, 35, 40]

    layer_results = {}
    probes = {}
    all_results = {}

    if verbose:
        print("\n" + "=" * 60)
        print("SUBSUMPTION PROBE - LAYER SWEEP")
        print("=" * 60)

    for layer_idx in target_layers:
        probe, results = train_subsumption_probe(
            activation_caches, examples, layer_idx,
            test_size=test_size, random_state=random_state, verbose=verbose
        )

        all_results[layer_idx] = results

        if 'error' not in results:
            layer_results[layer_idx] = results['accuracy']
            probes[layer_idx] = probe

    # Summary
    if verbose and layer_results:
        best_layer = max(layer_results, key=layer_results.get)
        best_acc = layer_results[best_layer]
        print("-" * 60)
        print(f"Best layer: {best_layer} (accuracy={best_acc:.3f})")
        print("=" * 60)

    return {
        'layer_results': layer_results,
        'all_results': all_results,
        'probes': probes,
    }
