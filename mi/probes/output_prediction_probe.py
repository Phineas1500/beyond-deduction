"""
Probe 2: When is the output level (child vs parent) decided?

Input: Activation at final position (before generation)
Label: 'will_output_child' or 'will_output_parent'
Question: At which layer does the model "decide" what to output?

This is the KEY probe for understanding the "conservation law":
- Gemma 2 9B has H1_acc + H2_acc ~ 100%
- Model seems to have fixed p(parent) ~ 3-5%
- If probe accuracy peaks early: decision is made early (representation-level)
- If probe accuracy peaks late: decision is made late (computation-level)
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from .base_probe import BaseLinearProbe
from ..activation_extractor import ActivationCache
from ..data_loader import ProbingExample


class OutputPredictionProbe(BaseLinearProbe):
    """
    Probe to predict whether model will output child or parent concept.

    This is the most important probe for understanding when the output
    level decision is made in the network.

    Expected trajectory (from MI research plan Part 9.3):
    - Layers 0-10: ~50% accuracy (decision not yet made)
    - Layers 10-20: ~65% accuracy (decision emerging)
    - Layers 20-30: ~80% accuracy (decision crystallizing)
    - Layers 30-40: ~90% accuracy (decision made)
    """

    # Label encoding
    LABEL_MAP = {'will_output_child': 0, 'will_output_parent': 1}
    INV_LABEL_MAP = {0: 'will_output_child', 1: 'will_output_parent'}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_data(
        self,
        activation_caches: List[ActivationCache],
        examples: List[ProbingExample],
        layer_idx: int,
        use_response_parsing: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data using behavioral outcomes.

        Label is determined by what the model ACTUALLY output:
        - If model response contains parent concept -> will_output_parent
        - If model response contains child concept -> will_output_child

        Args:
            activation_caches: List of ActivationCache from extraction
            examples: List of ProbingExample with model_response fields
            layer_idx: Which layer to use
            use_response_parsing: If True, parse response to determine label.
                                  If False, use model_correct field.

        Returns:
            X: (n_samples, hidden_size)
            y: (n_samples,) with 0=child, 1=parent
        """
        X_list = []
        y_list = []

        # Create lookup: (idx, h1_or_h2) -> example
        example_lookup = {(e.idx, e.h1_or_h2): e for e in examples}

        for cache in activation_caches:
            layer_acts = cache.layer_activations.get(layer_idx, {})

            # Get final position activation
            if 'final_position' not in layer_acts:
                continue

            final_act = layer_acts['final_position']
            if hasattr(final_act, 'float'):
                final_act = final_act.float().numpy()  # Convert bfloat16 to float32
            elif hasattr(final_act, 'numpy'):
                final_act = final_act.numpy()

            # Handle shape
            if final_act.ndim == 2:
                final_act = final_act[0]  # Take first if multiple

            # Find corresponding example
            example = example_lookup.get((cache.example_idx, cache.h1_or_h2))
            if example is None:
                continue

            # Determine label from behavioral outcome
            if use_response_parsing and example.model_response:
                response = example.model_response.lower()
                child = example.child_concept.lower()
                parent = example.root_concept.lower()

                if parent in response:
                    label = self.LABEL_MAP['will_output_parent']
                elif child in response:
                    label = self.LABEL_MAP['will_output_child']
                else:
                    # Can't determine from response, skip
                    continue
            else:
                # Use model_correct field
                # For H1: correct means child output, wrong means parent output
                # For H2: correct means parent output, wrong means child output
                if example.h1_or_h2 == 'h1':
                    if example.model_correct:
                        label = self.LABEL_MAP['will_output_child']
                    else:
                        label = self.LABEL_MAP['will_output_parent']
                else:  # h2
                    if example.model_correct:
                        label = self.LABEL_MAP['will_output_parent']
                    else:
                        label = self.LABEL_MAP['will_output_child']

            X_list.append(final_act)
            y_list.append(label)

        if len(X_list) == 0:
            return np.array([]), np.array([])

        return np.array(X_list), np.array(y_list)

    def interpret_trajectory(
        self,
        layer_accuracies: Dict[int, float]
    ) -> Dict[str, Any]:
        """
        Interpret the probe accuracy trajectory across layers.

        Args:
            layer_accuracies: Dict mapping layer_idx to accuracy

        Returns:
            Interpretation dict with peak layer, decision timing, etc.
        """
        if not layer_accuracies:
            return {'error': 'No layer accuracies provided'}

        layers = sorted(layer_accuracies.keys())
        accuracies = [layer_accuracies[l] for l in layers]

        peak_layer = layers[np.argmax(accuracies)]
        peak_accuracy = max(accuracies)

        # Determine decision timing
        if peak_layer <= 15:
            timing = "EARLY_DECISION"
            description = "Output level is determined in early layers (representation-level)"
        elif peak_layer <= 25:
            timing = "MID_DECISION"
            description = "Output level crystallizes in middle layers (around layer 20)"
        else:
            timing = "LATE_DECISION"
            description = "Output level is determined in late layers (computation-level)"

        # Find first layer where accuracy exceeds threshold
        thresholds = {'60%': 0.60, '70%': 0.70, '80%': 0.80}
        first_above = {}
        for name, thresh in thresholds.items():
            for layer, acc in zip(layers, accuracies):
                if acc >= thresh:
                    first_above[name] = layer
                    break

        return {
            'peak_layer': peak_layer,
            'peak_accuracy': peak_accuracy,
            'timing': timing,
            'description': description,
            'first_above_threshold': first_above,
            'layer_accuracies': layer_accuracies,
        }


def train_output_prediction_probe(
    activation_caches: List[ActivationCache],
    examples: List[ProbingExample],
    layer_idx: int,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[Optional[OutputPredictionProbe], Dict[str, Any]]:
    """
    Train an output prediction probe on activation caches.

    Args:
        activation_caches: List of ActivationCache
        examples: List of ProbingExample with behavioral outcomes
        layer_idx: Layer to probe
        test_size: Test set fraction
        random_state: Random seed
        verbose: Whether to print results

    Returns:
        (fitted_probe, results_dict)
    """
    from sklearn.model_selection import train_test_split

    probe = OutputPredictionProbe(random_state=random_state)
    X, y = probe.prepare_data(activation_caches, examples, layer_idx)

    if len(X) < 10:
        if verbose:
            print(f"  Layer {layer_idx}: Too few samples ({len(X)})")
        return None, {'error': f'Too few samples: {len(X)}', 'layer_idx': layer_idx}

    # Check class balance
    n_child = np.sum(y == probe.LABEL_MAP['will_output_child'])
    n_parent = np.sum(y == probe.LABEL_MAP['will_output_parent'])

    if n_child < 3 or n_parent < 3:
        if verbose:
            print(f"  Layer {layer_idx}: Class imbalance (child={n_child}, parent={n_parent})")
        return None, {
            'error': f'Class imbalance: child={n_child}, parent={n_parent}',
            'layer_idx': layer_idx
        }

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
    results['n_child'] = int(n_child)
    results['n_parent'] = int(n_parent)
    results['layer_idx'] = layer_idx

    if verbose:
        print(f"  Layer {layer_idx}: Accuracy={results['accuracy']:.3f} "
              f"(balanced={results['balanced_accuracy']:.3f}) "
              f"[child={n_child}, parent={n_parent}]")

    return probe, results


def run_layer_sweep(
    activation_caches: List[ActivationCache],
    examples: List[ProbingExample],
    target_layers: List[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run output prediction probe across multiple layers.

    This is the key experiment for understanding decision timing.

    Args:
        activation_caches: List of ActivationCache
        examples: List of ProbingExample
        target_layers: Layers to probe (default: [8, 15, 20, 25, 30, 35, 40])
        test_size: Test set fraction
        random_state: Random seed
        verbose: Whether to print progress

    Returns:
        Dict with layer_results, interpretation, probes
    """
    if target_layers is None:
        target_layers = [8, 15, 20, 25, 30, 35, 40]

    layer_results = {}
    probes = {}

    if verbose:
        print("\n" + "=" * 60)
        print("OUTPUT PREDICTION PROBE - LAYER SWEEP")
        print("=" * 60)

    for layer_idx in target_layers:
        probe, results = train_output_prediction_probe(
            activation_caches, examples, layer_idx,
            test_size=test_size, random_state=random_state, verbose=verbose
        )

        if 'error' not in results:
            layer_results[layer_idx] = results['accuracy']
            probes[layer_idx] = probe

    # Interpret trajectory
    dummy_probe = OutputPredictionProbe()
    interpretation = dummy_probe.interpret_trajectory(layer_results)

    if verbose:
        print("-" * 60)
        print(f"Peak layer: {interpretation.get('peak_layer', 'N/A')}")
        print(f"Peak accuracy: {interpretation.get('peak_accuracy', 0):.3f}")
        print(f"Timing: {interpretation.get('timing', 'N/A')}")
        print(f"Description: {interpretation.get('description', 'N/A')}")
        print("=" * 60)

    return {
        'layer_results': layer_results,
        'interpretation': interpretation,
        'probes': probes,
    }
