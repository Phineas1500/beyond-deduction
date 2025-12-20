"""
Probe Causal Consistency Test + Layer 20 Pivot

Test whether steering changes the probe's prediction even when generation doesn't change.

Smoking Gun Scenario: Steering flips probe prediction (child → parent),
but actual output stays child. This proves the probe measures something causally impotent.
"""

import sys
sys.path.insert(0, 'C:/Users/skiron/beyond-deduction')
sys.path.insert(0, 'C:/Users/skiron/beyond-deduction/benchmark')

import torch
import pickle
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from tqdm import tqdm


def load_sae(layer: int, use_it: bool = False):
    """Load SAE for the specified layer."""
    from sae_lens import SAE

    it_available_layers = {9, 20, 31}

    if use_it and layer in it_available_layers:
        release = "gemma-scope-9b-it-res-canonical"
    else:
        release = "gemma-scope-9b-pt-res-canonical"

    sae_id = f"layer_{layer}/width_16k/canonical"
    print(f"Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device="cuda")
    return sae


def train_probe_from_activations(activations, labels):
    """Train a logistic regression probe."""
    X = np.array(activations)
    y = np.array(labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    probe = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced')
    probe.fit(X_scaled, y)

    return probe, scaler


def run_part1_causal_consistency(model, tokenizer, sae, examples, probe, scaler):
    """
    Part 1: Test if steering changes probe prediction without changing output.
    """
    print("\n" + "="*70)
    print("PART 1: CAUSAL CONSISTENCY TEST")
    print("Does steering change probe prediction?")
    print("="*70)

    steering_idx = 2639  # Parent-associated feature from our SAE analysis
    steering_vec = sae.W_dec[steering_idx].to(model.device).to(torch.bfloat16)

    steering_strengths = [0, 5, 10, 20, 50]

    print(f"\nTesting {len(examples)} examples across steering strengths: {steering_strengths}")
    print(f"Steering vector: Feature {steering_idx} (parent-associated)")

    results = {strength: {'baseline_preds': [], 'steered_preds': [],
                          'baseline_probs': [], 'steered_probs': [], 'flips': 0}
               for strength in steering_strengths}

    for ex_idx, ex in enumerate(tqdm(examples, desc="Testing examples")):
        prompt = ex['prompt']
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        for strength in steering_strengths:
            activations = {'baseline': None, 'steered': None}

            # Baseline (no steering)
            def baseline_hook(module, input, output):
                activations['baseline'] = output[0][:, -1, :].detach().cpu().float().numpy()
                return output

            h = model.model.layers[8].register_forward_hook(baseline_hook)
            with torch.no_grad():
                model(**inputs)
            h.remove()

            # Steered
            def steering_hook(module, input, output):
                modified = output[0].clone()
                modified[:, -1, :] += strength * steering_vec
                activations['steered'] = modified[:, -1, :].detach().cpu().float().numpy()
                return (modified,) + output[1:]

            h = model.model.layers[8].register_forward_hook(steering_hook)
            with torch.no_grad():
                model(**inputs)
            h.remove()

            # Scale activations
            baseline_scaled = scaler.transform(activations['baseline'])
            steered_scaled = scaler.transform(activations['steered'])

            # Get probe predictions
            baseline_pred = probe.predict(baseline_scaled)[0]
            steered_pred = probe.predict(steered_scaled)[0]

            baseline_prob = probe.predict_proba(baseline_scaled)[0, 1]  # P(parent)
            steered_prob = probe.predict_proba(steered_scaled)[0, 1]

            results[strength]['baseline_preds'].append(baseline_pred)
            results[strength]['steered_preds'].append(steered_pred)
            results[strength]['baseline_probs'].append(baseline_prob)
            results[strength]['steered_probs'].append(steered_prob)

            if baseline_pred != steered_pred:
                results[strength]['flips'] += 1

    # Summarize results
    print("\n" + "="*70)
    print("RESULTS: Probe Prediction Changes Under Steering")
    print("="*70)
    print(f"{'Strength':<10} | {'Flips':<15} | {'P(parent) baseline→steered':<30}")
    print("-"*70)

    for strength in steering_strengths:
        r = results[strength]
        flips = r['flips']
        flip_pct = 100 * flips / len(examples)

        baseline_prob_mean = np.mean(r['baseline_probs'])
        steered_prob_mean = np.mean(r['steered_probs'])

        print(f"{strength:<10} | {flips:>3} ({flip_pct:>5.1f}%)   | "
              f"{baseline_prob_mean:.3f} → {steered_prob_mean:.3f}")

    # Compute max flip rate for interpretation
    max_flip_pct = max(100 * results[s]['flips'] / len(examples)
                       for s in steering_strengths if s > 0)

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if max_flip_pct > 30:
        interpretation = 'scenario_b'
        print("""
SCENARIO B CONFIRMED: Probe flips but generation doesn't change.

The probe is detecting the steering vector direction, but this direction
is NOT causally controlling the output. The probe is measuring the
"speedometer" (what's in the residual stream) not the "steering wheel"
(what determines output).

Layer 8 is "The Reporter, Not the Commander" - it reports on the problem
type but the decision is made elsewhere.

RECOMMENDATION: Pivot to Layer 20 IT SAE for causal analysis.
""")
    else:
        interpretation = 'inconclusive'
        print("""
Probe is relatively robust to steering - predictions don't flip dramatically.

This could mean:
1. The steering vector is too weak to shift the probe direction
2. The probe is detecting something orthogonal to our steering
3. The probe genuinely captures a stable representation

Need further investigation.
""")

    return results, interpretation, max_flip_pct


def run_part2_layer20_analysis(model, tokenizer, results_dir):
    """
    Part 2: Analyze Layer 20 with IT SAE.
    """
    print("\n" + "="*70)
    print("PART 2: LAYER 20 ANALYSIS")
    print("Finding the Actual Decision Point")
    print("="*70)

    # Load Layer 20 IT SAE
    sae_l20 = load_sae(layer=20, use_it=True)
    print(f"Loaded Layer 20 IT SAE: {sae_l20.cfg.d_sae} features")

    # Load the merged data to get examples with labels
    with open("C:/Users/skiron/beyond-deduction/benchmark/factorial_results/factorial_gemma29b_n2000_merged.pkl", "rb") as f:
        merged_data = pickle.load(f)

    pairs = merged_data['pairs']
    h1_results = merged_data['h1_results']
    h2_results = merged_data['h2_results']
    gold_indices = merged_data['gold_indices']

    # Get control indices (child output)
    control_indices = [i for i in range(len(h1_results))
                       if h1_results[i]['strong'] == 1 and h2_results[i]['strong'] == 0]

    # System prompt for prompts
    SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
Only output final hypotheses."""

    def format_prompt(theories_nl, observations_nl):
        user_prompt = f"Q: {theories_nl} We observe that: {observations_nl} Please come up with hypothesis to explain observations."
        return f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    # Sample examples for analysis
    import random
    random.seed(42)

    n_gold = min(53, len(gold_indices))
    n_control = min(53, len(control_indices))

    gold_sample = list(gold_indices)[:n_gold]
    control_sample = random.sample(control_indices, n_control)

    print(f"\nExtracting Layer 20 activations:")
    print(f"  Gold (parent output): {n_gold} examples")
    print(f"  Control (child output): {n_control} examples")

    parent_features_l20 = []
    child_features_l20 = []
    parent_activations_raw = []
    child_activations_raw = []

    # Process gold examples (parent output)
    for idx in tqdm(gold_sample, desc="Gold examples (parent)"):
        h1_ex, h2_ex = pairs[idx]
        prompt = format_prompt(h1_ex['theories_nl'], h1_ex['observations_nl'])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_l20 = outputs.hidden_states[21][:, -1, :].float()

            # Encode with SAE
            features = sae_l20.encode(hidden_l20)
            features_np = features.cpu().numpy().flatten()
            raw_np = hidden_l20.cpu().numpy().flatten()

        parent_features_l20.append(features_np)
        parent_activations_raw.append(raw_np)

    # Process control examples (child output)
    for idx in tqdm(control_sample, desc="Control examples (child)"):
        h1_ex, h2_ex = pairs[idx]
        prompt = format_prompt(h1_ex['theories_nl'], h1_ex['observations_nl'])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_l20 = outputs.hidden_states[21][:, -1, :].float()

            # Encode with SAE
            features = sae_l20.encode(hidden_l20)
            features_np = features.cpu().numpy().flatten()
            raw_np = hidden_l20.cpu().numpy().flatten()

        child_features_l20.append(features_np)
        child_activations_raw.append(raw_np)

    parent_features_l20 = np.array(parent_features_l20)
    child_features_l20 = np.array(child_features_l20)
    parent_activations_raw = np.array(parent_activations_raw)
    child_activations_raw = np.array(child_activations_raw)

    print(f"\nParent examples: {len(parent_features_l20)}")
    print(f"Child examples: {len(child_features_l20)}")

    # Differential analysis on SAE features
    parent_mean = parent_features_l20.mean(axis=0)
    child_mean = child_features_l20.mean(axis=0)
    diff = parent_mean - child_mean

    # Top differential features
    top_parent_idx = np.argsort(diff)[-20:][::-1]
    top_child_idx = np.argsort(diff)[:20]

    print("\n" + "="*70)
    print("TOP DIFFERENTIAL FEATURES AT LAYER 20 (IT SAE)")
    print("="*70)

    print("\nParent-associated features (higher in parent outputs):")
    for idx in top_parent_idx[:10]:
        print(f"  Feature {idx}: diff={diff[idx]:.4f}, "
              f"parent_mean={parent_mean[idx]:.4f}, child_mean={child_mean[idx]:.4f}")
        print(f"    Neuronpedia: https://www.neuronpedia.org/gemma-2-9b-it/20-res-16k/{idx}")

    print("\nChild-associated features (higher in child outputs):")
    for idx in top_child_idx[:10]:
        print(f"  Feature {idx}: diff={diff[idx]:.4f}, "
              f"parent_mean={parent_mean[idx]:.4f}, child_mean={child_mean[idx]:.4f}")
        print(f"    Neuronpedia: https://www.neuronpedia.org/gemma-2-9b-it/20-res-16k/{idx}")

    # Train probe at Layer 20 on RAW activations (for comparison)
    print("\n" + "="*70)
    print("TRAINING PROBE AT LAYER 20 (Raw Activations)")
    print("="*70)

    X_raw = np.vstack([parent_activations_raw, child_activations_raw])
    y = np.array([1]*len(parent_activations_raw) + [0]*len(child_activations_raw))

    # Shuffle
    shuffle_idx = np.random.permutation(len(X_raw))
    X_raw = X_raw[shuffle_idx]
    y_shuffled = y[shuffle_idx]

    # Cross-validation on raw activations
    scaler_l20 = StandardScaler()
    X_raw_scaled = scaler_l20.fit_transform(X_raw)

    probe_l20_raw = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced')
    scores_raw = cross_val_score(probe_l20_raw, X_raw_scaled, y_shuffled, cv=5, scoring='balanced_accuracy')

    print(f"Layer 20 Raw Activation Probe: {scores_raw.mean():.1%} (+/- {scores_raw.std()*2:.1%})")

    # Train probe on SAE features
    print("\n" + "="*70)
    print("TRAINING PROBE AT LAYER 20 (SAE Features)")
    print("="*70)

    X_sae = np.vstack([parent_features_l20, child_features_l20])
    X_sae = X_sae[shuffle_idx]

    probe_l20_sae = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced')
    scores_sae = cross_val_score(probe_l20_sae, X_sae, y_shuffled, cv=5, scoring='balanced_accuracy')

    print(f"Layer 20 SAE Feature Probe: {scores_sae.mean():.1%} (+/- {scores_sae.std()*2:.1%})")

    # Fit final probes
    probe_l20_raw.fit(X_raw_scaled, y_shuffled)
    probe_l20_sae.fit(X_sae, y_shuffled)

    # Save results
    l20_results = {
        'parent_features': parent_features_l20,
        'child_features': child_features_l20,
        'parent_activations_raw': parent_activations_raw,
        'child_activations_raw': child_activations_raw,
        'differential': diff,
        'top_parent_features': top_parent_idx.tolist(),
        'top_child_features': top_child_idx.tolist(),
        'probe_raw_accuracy': scores_raw.mean(),
        'probe_sae_accuracy': scores_sae.mean(),
        'probe_l20_raw': probe_l20_raw,
        'probe_l20_sae': probe_l20_sae,
        'scaler_l20': scaler_l20,
    }

    with open(results_dir / "layer20_sae_analysis.pkl", "wb") as f:
        pickle.dump(l20_results, f)

    print(f"\nLayer 20 analysis saved to {results_dir / 'layer20_sae_analysis.pkl'}")

    return l20_results


def main():
    results_dir = Path("C:/Users/skiron/beyond-deduction/mi/results_n2000")

    print("="*70)
    print("PROBE CAUSAL CONSISTENCY TEST + LAYER 20 PIVOT")
    print("="*70)

    # Load model
    print("\n1. Loading Gemma 2 9B...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

    # Load SAE for Layer 8
    print("\n2. Loading Layer 8 SAE...")
    sae_l8 = load_sae(layer=8, use_it=False)

    # Load intervention results to get examples with prompts
    print("\n3. Loading test examples...")
    with open(results_dir / "intervention_generation_results.pkl", "rb") as f:
        intervention_results = pickle.load(f)

    examples = []
    for detail in intervention_results['experiment_a'][0]['details']:
        examples.append({
            'prompt': detail.get('prompt', ''),
            'parent_concept': detail['parent_concept'],
            'child_concept': detail['child_concept'],
        })

    # We need the prompts - let's reconstruct them from the merged data
    with open("C:/Users/skiron/beyond-deduction/benchmark/factorial_results/factorial_gemma29b_n2000_merged.pkl", "rb") as f:
        merged_data = pickle.load(f)

    pairs = merged_data['pairs']
    h1_results = merged_data['h1_results']
    h2_results = merged_data['h2_results']

    SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
Only output final hypotheses."""

    def format_prompt(theories_nl, observations_nl):
        user_prompt = f"Q: {theories_nl} We observe that: {observations_nl} Please come up with hypothesis to explain observations."
        return f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    # Get control indices and sample
    control_indices = [i for i in range(len(h1_results))
                       if h1_results[i]['strong'] == 1 and h2_results[i]['strong'] == 0]

    import random
    random.seed(42)
    selected_controls = random.sample(control_indices, min(50, len(control_indices)))

    examples = []
    for idx in selected_controls:
        h1_ex, h2_ex = pairs[idx]
        prompt = format_prompt(h1_ex['theories_nl'], h1_ex['observations_nl'])
        examples.append({
            'prompt': prompt,
            'parent_concept': h1_ex['root_concept'],
            'child_concept': h1_ex['child_concept'],
        })

    print(f"   Prepared {len(examples)} examples for testing")

    # Train a probe on Layer 8 activations first
    print("\n4. Training Layer 8 probe...")

    # Extract activations for probe training
    gold_indices = merged_data['gold_indices']
    n_gold = min(53, len(gold_indices))
    n_control = min(53, len(control_indices))

    gold_sample = list(gold_indices)[:n_gold]
    control_sample = random.sample(control_indices, n_control)

    activations_l8 = []
    labels = []

    print("   Extracting activations for probe training...")
    for idx in tqdm(gold_sample + control_sample, desc="Extracting"):
        h1_ex, h2_ex = pairs[idx]
        prompt = format_prompt(h1_ex['theories_nl'], h1_ex['observations_nl'])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_l8 = outputs.hidden_states[9][:, -1, :].float()  # Layer 8 = index 9

        activations_l8.append(hidden_l8.cpu().numpy().flatten())
        labels.append(1 if idx in gold_indices else 0)

    activations_l8 = np.array(activations_l8)
    labels = np.array(labels)

    probe_l8, scaler_l8 = train_probe_from_activations(activations_l8, labels)

    # Cross-validate
    X_scaled = scaler_l8.transform(activations_l8)
    scores = cross_val_score(probe_l8, X_scaled, labels, cv=5, scoring='balanced_accuracy')
    print(f"   Layer 8 Probe Accuracy: {scores.mean():.1%} (+/- {scores.std()*2:.1%})")

    # Run Part 1: Causal Consistency Test
    part1_results, interpretation, max_flip_pct = run_part1_causal_consistency(
        model, tokenizer, sae_l8, examples, probe_l8, scaler_l8
    )

    # Save Part 1 results
    with open(results_dir / "probe_causal_consistency.pkl", "wb") as f:
        pickle.dump({
            'results': part1_results,
            'steering_feature': 2639,
            'n_examples': len(examples),
            'interpretation': interpretation,
            'max_flip_pct': max_flip_pct,
        }, f)

    print(f"\nPart 1 results saved to {results_dir / 'probe_causal_consistency.pkl'}")

    # Run Part 2: Layer 20 Analysis (regardless of Part 1 outcome for comparison)
    print("\n" + "="*70)
    print("Running Layer 20 Analysis for comparison...")
    print("="*70)

    l20_results = run_part2_layer20_analysis(model, tokenizer, results_dir)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Layer 8 Probe Accuracy: {scores.mean():.1%}")
    print(f"Layer 8 Max Probe Flip Rate (under steering): {max_flip_pct:.1f}%")
    print(f"Layer 20 Raw Probe Accuracy: {l20_results['probe_raw_accuracy']:.1%}")
    print(f"Layer 20 SAE Probe Accuracy: {l20_results['probe_sae_accuracy']:.1%}")

    if interpretation == 'scenario_b':
        print("\n>>> CONCLUSION: Layer 8 probe is non-causal. Decision happens later.")
        print(">>> Top Layer 20 parent features may be better intervention targets.")
    else:
        print("\n>>> CONCLUSION: Results inconclusive. Need more investigation.")


if __name__ == "__main__":
    main()
