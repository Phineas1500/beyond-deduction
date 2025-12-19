"""
Check class balance and confusion matrix for output prediction probe.
Multi-seed, multi-layer analysis for robustness.
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
cache_path = Path(__file__).parent / "results" / "activation_cache.pkl"
pairs_path = Path(__file__).parent.parent / "benchmark" / "matched_pairs_set1_pure.pkl"
results_path = Path(__file__).parent.parent / "benchmark" / "factorial_results" / "factorial_gemma29b.pkl"

print("Loading activation caches...")
with open(cache_path, 'rb') as f:
    activation_caches = pickle.load(f)

print("Loading examples...")
from mi.data_loader import create_probing_dataset
examples = create_probing_dataset(str(pairs_path), str(results_path))

print(f"Loaded {len(activation_caches)} caches, {len(examples)} examples")

# Build lookup
example_lookup = {(ex.idx, ex.h1_or_h2): ex for ex in examples}
LABEL_MAP = {'will_output_child': 0, 'will_output_parent': 1}

def prepare_data_for_layer(layer_idx):
    """Extract activations and labels for a specific layer."""
    X_list = []
    y_list = []

    for cache in activation_caches:
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
        if example is None:
            continue

        # Label based on model response
        if example.model_response:
            response = example.model_response.lower()
            child = example.child_concept.lower()
            parent = example.root_concept.lower()

            if parent in response:
                label = LABEL_MAP['will_output_parent']
            elif child in response:
                label = LABEL_MAP['will_output_child']
            else:
                continue
        else:
            continue

        X_list.append(final_act)
        y_list.append(label)

    return np.array(X_list), np.array(y_list)


def evaluate_with_seeds(X, y, n_seeds=10):
    """Run evaluation with multiple random seeds."""
    balanced_accs = []
    parent_recalls = []

    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )

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

        # Parent recall
        if cm.shape[0] > 1:
            parent_recall = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
        else:
            parent_recall = 0

        balanced_accs.append(bal_acc)
        parent_recalls.append(parent_recall)

    return {
        'bal_acc_mean': np.mean(balanced_accs),
        'bal_acc_std': np.std(balanced_accs),
        'parent_recall_mean': np.mean(parent_recalls),
        'parent_recall_std': np.std(parent_recalls),
    }


# Run analysis
TARGET_LAYERS = [8, 15, 20, 25, 30, 35, 40]
N_SEEDS = 10

# Get class distribution from layer 8
X, y = prepare_data_for_layer(8)
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
print(f"MULTI-SEED ANALYSIS ({N_SEEDS} seeds per layer)")
print("=" * 60)
print(f"{'Layer':>6} | {'Balanced Acc':>15} | {'Parent Recall':>15}")
print("-" * 45)

layer_results = {}
for layer_idx in TARGET_LAYERS:
    X, y = prepare_data_for_layer(layer_idx)
    results = evaluate_with_seeds(X, y, n_seeds=N_SEEDS)
    layer_results[layer_idx] = results

    bal_str = f"{results['bal_acc_mean']:.1%} ± {results['bal_acc_std']:.1%}"
    par_str = f"{results['parent_recall_mean']:.1%} ± {results['parent_recall_std']:.1%}"
    print(f"{layer_idx:>6} | {bal_str:>15} | {par_str:>15}")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

# Check if balanced accuracy is consistently high
mean_bal_accs = [r['bal_acc_mean'] for r in layer_results.values()]
min_bal_acc = min(mean_bal_accs)
max_bal_acc = max(mean_bal_accs)

if min_bal_acc > 0.85:
    print(f"ROBUST FINDING: Balanced accuracy >{min_bal_acc:.0%} at ALL layers")
    print("Decision is locked in from layer 8 onwards.")
    print("This supports the 'representation-level decision' hypothesis.")
elif min_bal_acc > 0.70:
    print(f"MODERATE FINDING: Balanced accuracy {min_bal_acc:.0%}-{max_bal_acc:.0%}")
    print("Some predictive signal, but not as definitive.")
else:
    print(f"WEAK FINDING: Balanced accuracy varies {min_bal_acc:.0%}-{max_bal_acc:.0%}")
    print("Results may not be robust.")

# Check trajectory
early_acc = layer_results[8]['bal_acc_mean']
late_acc = layer_results[40]['bal_acc_mean']
print(f"\nLayer 8 → Layer 40: {early_acc:.1%} → {late_acc:.1%}")

if abs(early_acc - late_acc) < 0.05:
    print("Flat trajectory: decision is made by layer 8 and maintained.")
elif late_acc > early_acc + 0.1:
    print("Increasing trajectory: decision crystallizes in later layers.")
else:
    print("Decreasing trajectory: early representation is most predictive.")
