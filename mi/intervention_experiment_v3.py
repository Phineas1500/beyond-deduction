"""
Intervention Experiment v3: Measure Actual Generation Output

The definitive causal test - does steering actually change what the model outputs?

Previous experiments failed because:
- v1: Token probabilities at wrong position (~10^-6 noise)
- v2: Hook timing issue - hidden_states captured before modification

This version runs full generation and checks if output contains parent vs child concept.
"""

import sys
sys.path.insert(0, 'C:/Users/skiron/beyond-deduction')
sys.path.insert(0, 'C:/Users/skiron/beyond-deduction/benchmark')

import pickle
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from collections import defaultdict


def load_sae(layer: int = 8, width: str = "16k"):
    """Load SAE for the specified layer."""
    from sae_lens import SAE

    # PT SAE has all layers, IT SAE only has 9, 20, 31
    release = "gemma-scope-9b-pt-res-canonical"
    sae_id = f"layer_{layer}/width_{width}/canonical"

    print(f"Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device="cuda")
    return sae


def steering_intervention(direction, strength):
    """Returns intervention function that adds steering vector to last token position."""
    def intervene(hidden_states):
        # hidden_states: [batch, seq_len, d_model]
        # Add direction only to the last token position
        modified = hidden_states.clone()
        modified[:, -1, :] = modified[:, -1, :] + strength * direction.to(hidden_states.device)
        return modified
    return intervene


def ablation_intervention(direction):
    """Returns intervention function that projects out direction from last token."""
    def intervene(hidden_states):
        # Project out the direction from last token position
        # Match dtype and device with hidden_states
        dir_norm = (direction / direction.norm()).to(hidden_states.device).to(hidden_states.dtype)

        modified = hidden_states.clone()
        last_token = modified[:, -1, :]  # [batch, d_model]
        proj_coeff = (last_token @ dir_norm)  # [batch]
        modified[:, -1, :] = last_token - proj_coeff.unsqueeze(-1) * dir_norm
        return modified
    return intervene


def generate_with_intervention(model, tokenizer, prompt, intervention_fn=None, layer=8, max_new_tokens=100):
    """
    Generate text with optional intervention at specified layer.
    Returns generated text (excluding prompt).
    """
    tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = tokens.input_ids.shape[1]

    handles = []
    if intervention_fn:
        def hook(module, input, output):
            # output is tuple: (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                modified = intervention_fn(output[0])
                return (modified,) + output[1:]
            else:
                return intervention_fn(output)

        handle = model.model.layers[layer].register_forward_hook(hook)
        handles.append(handle)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                tokens.input_ids,
                attention_mask=tokens.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.pad_token_id,
            )
        # Decode only the generated part
        generated_ids = output_ids[0, prompt_len:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()

    return generated


def check_output_level(generated_text, parent_concept, child_concept):
    """
    Check if generated text contains parent and/or child concept.
    Returns: 'parent', 'child', 'both', or 'neither'
    """
    gen_lower = generated_text.lower()
    has_parent = parent_concept.lower() in gen_lower
    has_child = child_concept.lower() in gen_lower

    if has_parent and has_child:
        return 'both'
    elif has_parent:
        return 'parent'
    elif has_child:
        return 'child'
    else:
        return 'neither'


def run_experiment_a(model, tokenizer, examples, steering_direction, layer=8):
    """
    Experiment A: Steering Strength Sweep

    For each strength, count how many examples output parent vs child.
    """
    strengths = [0, 1, 2, 5, 10, 20, 50]
    results = {}

    for strength in strengths:
        print(f"\n  Steering strength: {strength}")
        counts = {'parent': 0, 'child': 0, 'both': 0, 'neither': 0}
        details = []

        intervention_fn = steering_intervention(steering_direction, strength) if strength > 0 else None

        for ex in tqdm(examples, desc=f"Strength {strength}"):
            prompt = ex['prompt']
            parent = ex['parent_concept']
            child = ex['child_concept']

            generated = generate_with_intervention(
                model, tokenizer, prompt,
                intervention_fn=intervention_fn,
                layer=layer
            )

            level = check_output_level(generated, parent, child)
            counts[level] += 1
            details.append({
                'parent_concept': parent,
                'child_concept': child,
                'generated': generated,
                'output_level': level
            })

        results[strength] = {
            'counts': counts,
            'details': details
        }

        # Print progress
        total = len(examples)
        print(f"    Parent: {counts['parent']}/{total} ({100*counts['parent']/total:.1f}%)")
        print(f"    Child: {counts['child']}/{total} ({100*counts['child']/total:.1f}%)")

    return results


def run_experiment_b(model, tokenizer, examples, ablation_direction, layer=8):
    """
    Experiment B: Ablation

    Compare baseline vs ablated generation.
    """
    print("\n  Running ablation experiment...")

    ablation_fn = ablation_intervention(ablation_direction)

    results = {
        'baseline': {'parent': 0, 'child': 0, 'both': 0, 'neither': 0},
        'ablated': {'parent': 0, 'child': 0, 'both': 0, 'neither': 0},
        'details': []
    }

    for ex in tqdm(examples, desc="Ablation"):
        prompt = ex['prompt']
        parent = ex['parent_concept']
        child = ex['child_concept']

        # Baseline
        baseline_gen = generate_with_intervention(model, tokenizer, prompt, None, layer)
        baseline_level = check_output_level(baseline_gen, parent, child)
        results['baseline'][baseline_level] += 1

        # Ablated
        ablated_gen = generate_with_intervention(model, tokenizer, prompt, ablation_fn, layer)
        ablated_level = check_output_level(ablated_gen, parent, child)
        results['ablated'][ablated_level] += 1

        results['details'].append({
            'parent_concept': parent,
            'child_concept': child,
            'baseline_generated': baseline_gen,
            'baseline_level': baseline_level,
            'ablated_generated': ablated_gen,
            'ablated_level': ablated_level,
            'changed': baseline_level != ablated_level
        })

    return results


def print_summary_table(exp_a_results, exp_b_results, n_examples):
    """Print summary tables."""

    print("\n" + "="*70)
    print("EXPERIMENT A: Steering Feature 2639 (Layer 8)")
    print("="*70)
    print(f"{'Strength':<10} | {'Parent':<12} | {'Child':<12} | {'Both':<12} | {'Neither':<12}")
    print("-"*70)

    for strength in sorted(exp_a_results.keys()):
        counts = exp_a_results[strength]['counts']
        parent_pct = 100 * counts['parent'] / n_examples
        child_pct = 100 * counts['child'] / n_examples
        both_pct = 100 * counts['both'] / n_examples
        neither_pct = 100 * counts['neither'] / n_examples

        print(f"{strength:<10} | {parent_pct:>5.1f}% ({counts['parent']:>3}) | "
              f"{child_pct:>5.1f}% ({counts['child']:>3}) | "
              f"{both_pct:>5.1f}% ({counts['both']:>3}) | "
              f"{neither_pct:>5.1f}% ({counts['neither']:>3})")

    print("\n" + "="*70)
    print("EXPERIMENT B: Ablation of Feature 9112 (Layer 8)")
    print("="*70)
    print(f"{'Condition':<12} | {'Parent':<12} | {'Child':<12} | {'Both':<12} | {'Neither':<12}")
    print("-"*70)

    for condition in ['baseline', 'ablated']:
        counts = exp_b_results[condition]
        parent_pct = 100 * counts['parent'] / n_examples
        child_pct = 100 * counts['child'] / n_examples
        both_pct = 100 * counts['both'] / n_examples
        neither_pct = 100 * counts['neither'] / n_examples

        print(f"{condition:<12} | {parent_pct:>5.1f}% ({counts['parent']:>3}) | "
              f"{child_pct:>5.1f}% ({counts['child']:>3}) | "
              f"{both_pct:>5.1f}% ({counts['both']:>3}) | "
              f"{neither_pct:>5.1f}% ({counts['neither']:>3})")

    # Count changes
    n_changed = sum(1 for d in exp_b_results['details'] if d['changed'])
    print(f"\nExamples where output changed: {n_changed}/{n_examples} ({100*n_changed/n_examples:.1f}%)")


SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
Only output final hypotheses."""


def format_prompt(theories_nl: str, observations_nl: str) -> str:
    """Format prompt for Gemma 2 IT."""
    user_prompt = f"Q: {theories_nl} We observe that: {observations_nl} Please come up with hypothesis to explain observations."
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}"


def main():
    print("="*70)
    print("INTERVENTION EXPERIMENT V3: MEASURE ACTUAL GENERATION OUTPUT")
    print("="*70)

    results_dir = Path("C:/Users/skiron/beyond-deduction/mi/results_n2000")
    project_dir = Path("C:/Users/skiron/beyond-deduction")

    # Load n2000 merged results - the primary data source
    print("\n1. Loading n2000 merged results...")
    with open(project_dir / "benchmark/factorial_results/factorial_gemma29b_n2000_merged.pkl", "rb") as f:
        merged_data = pickle.load(f)

    pairs = merged_data['pairs']
    h1_results = merged_data['h1_results']
    h2_results = merged_data['h2_results']
    gold_indices = merged_data['gold_indices']

    print(f"   Loaded {len(pairs)} pairs")
    print(f"   Gold indices (parent output): {len(gold_indices)}")

    # Get control examples: H1 correct (outputs child), H2 incorrect (also outputs child)
    # These are the examples that baseline behavior outputs child concept
    control_indices = [i for i in range(len(h1_results))
                       if h1_results[i]['strong'] == 1 and h2_results[i]['strong'] == 0]
    print(f"   Control indices (child output): {len(control_indices)}")

    # Load SAE and get feature directions
    print("\n2. Loading SAE...")
    sae = load_sae(layer=8, width="16k")

    # Feature 2639: top parent-associated feature (steering target)
    # Feature 9112: top child-associated feature (ablation target)
    steering_feature_idx = 2639
    ablation_feature_idx = 9112

    # Get decoder directions
    steering_direction = sae.W_dec[steering_feature_idx]  # [d_model]
    ablation_direction = sae.W_dec[ablation_feature_idx]  # [d_model]

    print(f"   Steering direction (feature {steering_feature_idx}): shape {steering_direction.shape}")
    print(f"   Ablation direction (feature {ablation_feature_idx}): shape {ablation_direction.shape}")

    # Load model
    print("\n3. Loading Gemma 2 9B...")
    model_name = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Prepare examples from control indices (baseline outputs child)
    print("\n4. Preparing examples...")
    examples = []
    n_to_use = min(50, len(control_indices))  # Use 50 control examples

    import random
    random.seed(42)
    selected_controls = random.sample(control_indices, n_to_use)

    for idx in selected_controls:
        h1_ex, h2_ex = pairs[idx]

        # Use H1 examples (1-hop, should output child)
        prompt = format_prompt(h1_ex['theories_nl'], h1_ex['observations_nl'])
        parent_concept = h1_ex['root_concept']
        child_concept = h1_ex['child_concept']

        examples.append({
            'prompt': prompt,
            'parent_concept': parent_concept,
            'child_concept': child_concept,
            'original_idx': idx
        })

    print(f"   Prepared {len(examples)} examples for intervention experiments")
    print(f"   Example concepts: parent='{examples[0]['parent_concept']}', child='{examples[0]['child_concept']}'")

    # Run experiments
    print("\n5. Running Experiment A: Steering Strength Sweep")
    exp_a_results = run_experiment_a(
        model, tokenizer, examples,
        steering_direction, layer=8
    )

    print("\n6. Running Experiment B: Ablation")
    exp_b_results = run_experiment_b(
        model, tokenizer, examples,
        ablation_direction, layer=8
    )

    # Print summary
    print_summary_table(exp_a_results, exp_b_results, len(examples))

    # Save results
    print("\n7. Saving results...")
    all_results = {
        'experiment_a': exp_a_results,
        'experiment_b': exp_b_results,
        'n_examples': len(examples),
        'steering_feature': steering_feature_idx,
        'ablation_feature': ablation_feature_idx,
        'layer': 8,
        'examples_metadata': [
            {'parent': e['parent_concept'], 'child': e['child_concept']}
            for e in examples
        ]
    }

    with open(results_dir / "intervention_generation_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print(f"   Results saved to {results_dir / 'intervention_generation_results.pkl'}")

    # Print some example outputs
    print("\n" + "="*70)
    print("SAMPLE OUTPUTS (first 3 examples)")
    print("="*70)

    for i in range(min(3, len(examples))):
        print(f"\nExample {i+1}:")
        print(f"  Parent concept: {examples[i]['parent_concept']}")
        print(f"  Child concept: {examples[i]['child_concept']}")

        # Baseline (strength=0)
        baseline = exp_a_results[0]['details'][i]
        print(f"  Baseline output ({baseline['output_level']}): {baseline['generated'][:200]}...")

        # High steering (strength=20)
        steered = exp_a_results[20]['details'][i]
        print(f"  Steered (s=20) ({steered['output_level']}): {steered['generated'][:200]}...")


if __name__ == "__main__":
    main()
