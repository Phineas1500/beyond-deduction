"""
Probe 1: Does the model distinguish concept levels (child vs parent)?

Input: Activation at concept token
Label: 'child' or 'parent'
Question: Does model encode hierarchical level in representation?
"""

import numpy as np
from typing import List, Tuple, Dict, Any

from .base_probe import BaseLinearProbe
from ..activation_extractor import ActivationCache


class ConceptLevelProbe(BaseLinearProbe):
    """
    Probe to determine if concept hierarchy level is encoded in activations.

    Train on:
    - Activations at child concept positions (label='child')
    - Activations at parent concept positions (label='parent')

    High accuracy = model distinguishes hierarchy levels in representations
    """

    # Label encoding
    LABEL_MAP = {'child': 0, 'parent': 1}
    INV_LABEL_MAP = {0: 'child', 1: 'parent'}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_data(
        self,
        activation_caches: List[ActivationCache],
        layer_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from activation caches.

        Extracts activations at child and parent concept positions
        and labels them accordingly.

        Args:
            activation_caches: List of ActivationCache from extraction
            layer_idx: Which layer to use

        Returns:
            X: (n_samples, hidden_size)
            y: (n_samples,) with 0=child, 1=parent
        """
        X_list = []
        y_list = []

        for cache in activation_caches:
            layer_acts = cache.layer_activations.get(layer_idx, {})

            # Child concept activations
            if 'child_positions' in layer_acts:
                child_acts = layer_acts['child_positions']
                if hasattr(child_acts, 'float'):
                    child_acts = child_acts.float().numpy()  # Convert bfloat16 to float32
                elif hasattr(child_acts, 'numpy'):
                    child_acts = child_acts.numpy()

                # Handle both single and multiple positions
                if child_acts.ndim == 1:
                    child_acts = child_acts.reshape(1, -1)

                for act in child_acts:
                    X_list.append(act)
                    y_list.append(self.LABEL_MAP['child'])

            # Parent concept activations
            if 'parent_positions' in layer_acts:
                parent_acts = layer_acts['parent_positions']
                if hasattr(parent_acts, 'float'):
                    parent_acts = parent_acts.float().numpy()  # Convert bfloat16 to float32
                elif hasattr(parent_acts, 'numpy'):
                    parent_acts = parent_acts.numpy()

                if parent_acts.ndim == 1:
                    parent_acts = parent_acts.reshape(1, -1)

                for act in parent_acts:
                    X_list.append(act)
                    y_list.append(self.LABEL_MAP['parent'])

        if len(X_list) == 0:
            return np.array([]), np.array([])

        return np.array(X_list), np.array(y_list)

    def prepare_data_from_examples(
        self,
        activation_caches: List[ActivationCache],
        examples: List[Any],  # List of ProbingExample
        layer_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Prepare data with additional metadata tracking.

        Args:
            activation_caches: Activation caches
            examples: Original ProbingExample objects
            layer_idx: Layer to use

        Returns:
            X, y, metadata dict
        """
        X, y = self.prepare_data(activation_caches, layer_idx)

        # Build metadata
        metadata = {
            'layer_idx': layer_idx,
            'n_child_samples': np.sum(y == self.LABEL_MAP['child']),
            'n_parent_samples': np.sum(y == self.LABEL_MAP['parent']),
            'n_caches': len(activation_caches),
        }

        return X, y, metadata

    def interpret_accuracy(self, accuracy: float) -> str:
        """
        Interpret the probe accuracy.

        Args:
            accuracy: Probe accuracy (0-1)

        Returns:
            Human-readable interpretation
        """
        if accuracy < 0.55:
            return "No distinction - child/parent encoded identically"
        elif accuracy < 0.70:
            return "Weak distinction - some level information encoded"
        elif accuracy < 0.85:
            return "Moderate distinction - hierarchy level is represented"
        elif accuracy < 0.95:
            return "Strong distinction - clear child/parent separation"
        else:
            return "Very strong distinction - hierarchy level fully encoded"


def train_concept_level_probe(
    activation_caches: List[ActivationCache],
    layer_idx: int,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[ConceptLevelProbe, Dict[str, Any]]:
    """
    Train a concept level probe on activation caches.

    Args:
        activation_caches: List of ActivationCache
        layer_idx: Layer to probe
        test_size: Test set fraction
        random_state: Random seed
        verbose: Whether to print results

    Returns:
        (fitted_probe, results_dict)
    """
    from sklearn.model_selection import train_test_split

    probe = ConceptLevelProbe(random_state=random_state)
    X, y = probe.prepare_data(activation_caches, layer_idx)

    if len(X) < 10:
        return None, {'error': f'Too few samples: {len(X)}'}

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Fit
    probe.fit(X_train, y_train)

    # Evaluate
    results = probe.evaluate(X_test, y_test)
    results['n_train'] = len(X_train)
    results['n_test'] = len(X_test)
    results['layer_idx'] = layer_idx
    results['interpretation'] = probe.interpret_accuracy(results['accuracy'])

    if verbose:
        print(f"  Layer {layer_idx}: Accuracy={results['accuracy']:.3f} "
              f"(balanced={results['balanced_accuracy']:.3f})")
        print(f"    {results['interpretation']}")

    return probe, results
