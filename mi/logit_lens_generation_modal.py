"""
Logit Lens at Concept Generation Position (Large Sample)
=========================================================

This analysis measures P(child) vs P(parent) at the position where the model
actually generates the concept token, NOT at the last input position.

KEY IMPROVEMENT: Samples from ALL examples and classifies by ACTUAL generation
output, not by factorial results. This captures the natural ~5% parent-output rate.

Usage:
    modal run mi/logit_lens_generation_modal.py
    modal run mi/logit_lens_generation_modal.py --n-total 500
    modal run mi/logit_lens_generation_modal.py --n-total 1000
    modal run --detach mi/logit_lens_generation_modal.py
"""

import modal
import os
from pathlib import Path

# Modal app
app = modal.App("logit-lens-generation-gemma2")

# Project path
PROJECT_DIR = Path(__file__).parent.parent

# Image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "numpy==1.26.4",
        "scipy==1.14.1",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
    )
    .add_local_dir(PROJECT_DIR / "mi", remote_path="/root/beyond-deduction/mi")
    .add_local_dir(PROJECT_DIR / "benchmark", remote_path="/root/beyond-deduction/benchmark")
)

# Volumes
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("logit-lens-generation-results", create_if_missing=True)

# Secrets
hf_secret = modal.Secret.from_name("huggingface-secret")


# System prompt (same as in other MI analyses)
SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
Only output final hypotheses."""


def format_prompt(theories_nl: str, observations_nl: str, tokenizer) -> str:
    """Format prompt for Gemma 2 IT with proper chat template."""
    user_prompt = f"Q: {theories_nl} We observe that: {observations_nl} Please come up with hypothesis to explain observations."
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    messages = [{"role": "user", "content": full_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def find_concept_token_ids(tokenizer, concept: str):
    """
    Find token IDs for a concept (may span multiple tokens).

    Returns:
        dict with 'first_token_id', 'all_token_ids', 'decoded_tokens'
    """
    # Try with leading space (how it appears in generation)
    tokens_with_space = tokenizer.encode(f" {concept}", add_special_tokens=False)

    # Also try capitalized
    tokens_capitalized = tokenizer.encode(f" {concept.capitalize()}", add_special_tokens=False)

    # Use whichever works
    if tokens_with_space:
        tokens = tokens_with_space
    elif tokens_capitalized:
        tokens = tokens_capitalized
    else:
        tokens = tokenizer.encode(concept, add_special_tokens=False)

    if not tokens:
        return None

    return {
        'first_token_id': tokens[0],
        'all_token_ids': tokens,
        'decoded_tokens': [tokenizer.decode([t]) for t in tokens],
    }


def find_concept_step(generated_ids, tokenizer, child_concept: str, parent_concept: str):
    """
    Find which generation step produces the concept token.

    Returns:
        tuple: (step_index, is_child, is_parent, token_str)
    """
    child_tokens = find_concept_token_ids(tokenizer, child_concept)
    parent_tokens = find_concept_token_ids(tokenizer, parent_concept)

    if child_tokens is None or parent_tokens is None:
        return 2, False, False, None  # Default fallback

    child_first = child_tokens['first_token_id']
    parent_first = parent_tokens['first_token_id']

    for step, token_id in enumerate(generated_ids):
        token_id_item = token_id.item() if hasattr(token_id, 'item') else token_id
        token_str = tokenizer.decode([token_id_item]).lower().strip()

        # Check if this token matches child or parent first token
        is_child = token_id_item == child_first
        is_parent = token_id_item == parent_first

        # Also check by string matching (for subword tokens)
        if not is_child and not is_parent:
            child_lower = child_concept.lower()
            parent_lower = parent_concept.lower()

            if len(token_str) > 0:
                if child_lower.startswith(token_str) or token_str.startswith(child_lower[:len(token_str)]):
                    is_child = True
                if parent_lower.startswith(token_str) or token_str.startswith(parent_lower[:len(token_str)]):
                    is_parent = True

        if is_child or is_parent:
            return step, is_child, is_parent, token_str

    return 2, False, False, None


def classify_by_actual_output(generated_text: str, child_concept: str, parent_concept: str) -> str:
    """
    Classify based on what the model actually generated.

    Returns one of: 'parent_output', 'child_output', 'both', 'neither'
    """
    gen_lower = generated_text.lower()
    child_lower = child_concept.lower()
    parent_lower = parent_concept.lower()

    has_child = child_lower in gen_lower
    has_parent = parent_lower in gen_lower

    if has_parent and not has_child:
        return 'parent_output'
    elif has_child and not has_parent:
        return 'child_output'
    elif has_child and has_parent:
        return 'both'
    else:
        return 'neither'


def get_early_logits(model, hidden_states, layer_idx):
    """Project hidden states through unembedding to get early logits."""
    # Apply final layer norm (RMSNorm for Gemma 2)
    normed = model.model.norm(hidden_states)
    # Project through unembedding (lm_head)
    logits = model.lm_head(normed)
    return logits


def run_single_example_generation(
    model,
    tokenizer,
    prompt: str,
    child_concept: str,
    parent_concept: str,
    target_layers: list,
    max_new_tokens: int = 10,
    debug: bool = False,
):
    """
    Run logit lens at concept generation position.

    Returns:
        dict with layer-wise probabilities and ranks at concept position,
        plus classification by actual output
    """
    import torch
    import torch.nn.functional as F

    # Get token info
    child_tokens = find_concept_token_ids(tokenizer, child_concept)
    parent_tokens = find_concept_token_ids(tokenizer, parent_concept)

    if child_tokens is None or parent_tokens is None:
        return None

    child_token_id = child_tokens['first_token_id']
    parent_token_id = parent_tokens['first_token_id']

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]

    # Generate with hidden states at each step
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=False,  # Greedy for reproducibility
        )

    # Get generated tokens (excluding input)
    generated_ids = outputs.sequences[0, input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if debug:
        print(f"  Generated: {generated_text}")
        print(f"  Tokens: {[tokenizer.decode([t]) for t in generated_ids]}")

    # Classify by actual output
    classification = classify_by_actual_output(generated_text, child_concept, parent_concept)

    if debug:
        print(f"  Classification: {classification}")

    # Find which step generates the concept
    concept_step, is_child, is_parent, token_str = find_concept_step(
        generated_ids, tokenizer, child_concept, parent_concept
    )

    if debug:
        print(f"  Concept step: {concept_step}, is_child={is_child}, is_parent={is_parent}")

    # Validate concept_step is within bounds
    if concept_step >= len(outputs.hidden_states):
        concept_step = min(2, len(outputs.hidden_states) - 1)

    # Get hidden states at concept step
    hidden_at_concept = outputs.hidden_states[concept_step]

    layer_probs = {}
    layer_ranks = {}

    for layer in target_layers:
        if layer + 1 >= len(hidden_at_concept):
            continue

        hs = hidden_at_concept[layer + 1]
        hs_last = hs[:, -1:, :]

        early_logits = get_early_logits(model, hs_last, layer)
        last_logits = early_logits[0, 0, :]

        probs = F.softmax(last_logits, dim=-1)

        p_child = probs[child_token_id].item()
        p_parent = probs[parent_token_id].item()
        layer_probs[layer] = (p_child, p_parent)

        sorted_indices = torch.argsort(last_logits, descending=True)
        ranks = torch.zeros_like(last_logits, dtype=torch.long)
        ranks[sorted_indices] = torch.arange(len(last_logits), device=last_logits.device)

        child_rank = ranks[child_token_id].item()
        parent_rank = ranks[parent_token_id].item()
        layer_ranks[layer] = (child_rank, parent_rank)

        if debug and layer in [0, 8, 20, 40]:
            print(f"    Layer {layer}: ranks: child={child_rank}, parent={parent_rank}")

    return {
        'probs': layer_probs,
        'ranks': layer_ranks,
        'concept_step': concept_step,
        'generated_text': generated_text,
        'classification': classification,
        'child_concept': child_concept,
        'parent_concept': parent_concept,
        'child_token_id': child_token_id,
        'parent_token_id': parent_token_id,
    }


def compute_statistics_by_class(results_by_class, target_layers):
    """Compute aggregate statistics for each classification group."""
    import numpy as np

    stats = {
        'counts': {k: len(v) for k, v in results_by_class.items()},
        'by_layer': {},
    }

    for layer in target_layers:
        stats['by_layer'][layer] = {}

        for class_name, results in results_by_class.items():
            if not results:
                stats['by_layer'][layer][class_name] = {
                    'child_rank_mean': 0, 'child_rank_std': 0,
                    'parent_rank_mean': 0, 'parent_rank_std': 0,
                    'rank_gap_mean': 0, 'rank_gap_std': 0,
                    'p_child_mean': 0, 'p_parent_mean': 0,
                }
                continue

            child_ranks = [r['ranks'][layer][0] for r in results if layer in r['ranks']]
            parent_ranks = [r['ranks'][layer][1] for r in results if layer in r['ranks']]
            rank_gaps = [r['ranks'][layer][1] - r['ranks'][layer][0] for r in results if layer in r['ranks']]
            p_child = [r['probs'][layer][0] for r in results if layer in r['probs']]
            p_parent = [r['probs'][layer][1] for r in results if layer in r['probs']]

            stats['by_layer'][layer][class_name] = {
                'child_rank_mean': float(np.mean(child_ranks)) if child_ranks else 0,
                'child_rank_std': float(np.std(child_ranks)) if child_ranks else 0,
                'parent_rank_mean': float(np.mean(parent_ranks)) if parent_ranks else 0,
                'parent_rank_std': float(np.std(parent_ranks)) if parent_ranks else 0,
                'rank_gap_mean': float(np.mean(rank_gaps)) if rank_gaps else 0,
                'rank_gap_std': float(np.std(rank_gaps)) if rank_gaps else 0,
                'p_child_mean': float(np.mean(p_child)) if p_child else 0,
                'p_parent_mean': float(np.mean(p_parent)) if p_parent else 0,
            }

    # Find divergence layer between child_output and parent_output
    divergence_layer = None
    for layer in sorted(target_layers):
        c_gap = stats['by_layer'][layer].get('child_output', {}).get('rank_gap_mean', 0)
        p_gap = stats['by_layer'][layer].get('parent_output', {}).get('rank_gap_mean', 0)
        if abs(c_gap - p_gap) > 1000:
            divergence_layer = layer
            break

    stats['divergence_layer'] = divergence_layer

    return stats


def create_visualizations(results, output_dir, highlight_layers=None):
    """Create and save visualizations for large sample analysis."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = results['statistics']
    layers = sorted(stats['by_layer'].keys())
    counts = stats['counts']

    if highlight_layers is None:
        highlight_layers = [8, 12, 20, 40]

    # =====================
    # Plot 1: Rank Gap Comparison (Child-Output vs Parent-Output)
    # =====================
    fig, ax = plt.subplots(figsize=(14, 6))

    child_gaps = [stats['by_layer'][l].get('child_output', {}).get('rank_gap_mean', 0) for l in layers]
    child_gaps_std = [stats['by_layer'][l].get('child_output', {}).get('rank_gap_std', 0) for l in layers]
    parent_gaps = [stats['by_layer'][l].get('parent_output', {}).get('rank_gap_mean', 0) for l in layers]
    parent_gaps_std = [stats['by_layer'][l].get('parent_output', {}).get('rank_gap_std', 0) for l in layers]

    ax.plot(layers, child_gaps, 'g-o',
            label=f'Child-output (n={counts.get("child_output", 0)})',
            linewidth=2, markersize=8)
    ax.fill_between(layers,
                   [g - s for g, s in zip(child_gaps, child_gaps_std)],
                   [g + s for g, s in zip(child_gaps, child_gaps_std)],
                   alpha=0.2, color='green')

    ax.plot(layers, parent_gaps, 'm-s',
            label=f'Parent-output (n={counts.get("parent_output", 0)})',
            linewidth=2, markersize=8)
    ax.fill_between(layers,
                   [g - s for g, s in zip(parent_gaps, parent_gaps_std)],
                   [g + s for g, s in zip(parent_gaps, parent_gaps_std)],
                   alpha=0.2, color='magenta')

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    if stats.get('divergence_layer'):
        ax.axvline(x=stats['divergence_layer'], color='red', linestyle='--',
                   alpha=0.8, linewidth=2, label=f"Divergence at L{stats['divergence_layer']}")

    for hl in highlight_layers:
        if hl in layers:
            ax.axvline(x=hl, color='gray', linestyle=':', alpha=0.4)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Rank Gap (parent_rank - child_rank)', fontsize=12)
    ax.set_title('Rank Gap by Actual Generation Output\n(Positive = child preferred, Negative = parent preferred)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'rank_gap_by_output.png', dpi=150)
    plt.savefig(output_path / 'rank_gap_by_output.pdf')
    plt.close()

    # =====================
    # Plot 2: Rank Evolution for Each Group
    # =====================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Child-output
    ax = axes[0]
    child_child_ranks = [stats['by_layer'][l].get('child_output', {}).get('child_rank_mean', 0) for l in layers]
    child_parent_ranks = [stats['by_layer'][l].get('child_output', {}).get('parent_rank_mean', 0) for l in layers]

    ax.plot(layers, child_child_ranks, 'b-o', label='Child concept rank', linewidth=2)
    ax.plot(layers, child_parent_ranks, 'r-s', label='Parent concept rank', linewidth=2)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Rank (lower = more likely)', fontsize=12)
    ax.set_title(f'Child-Output Cases (n={counts.get("child_output", 0)})\nChild should rank lower', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Right: Parent-output
    ax = axes[1]
    parent_child_ranks = [stats['by_layer'][l].get('parent_output', {}).get('child_rank_mean', 0) for l in layers]
    parent_parent_ranks = [stats['by_layer'][l].get('parent_output', {}).get('parent_rank_mean', 0) for l in layers]

    ax.plot(layers, parent_child_ranks, 'b-o', label='Child concept rank', linewidth=2)
    ax.plot(layers, parent_parent_ranks, 'r-s', label='Parent concept rank', linewidth=2)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Rank (lower = more likely)', fontsize=12)
    ax.set_title(f'Parent-Output Cases (n={counts.get("parent_output", 0)})\nParent should rank lower', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path / 'rank_evolution_by_output.png', dpi=150)
    plt.savefig(output_path / 'rank_evolution_by_output.pdf')
    plt.close()

    # =====================
    # Plot 3: Final Layers Comparison (Bar Chart)
    # =====================
    fig, ax = plt.subplots(figsize=(12, 6))

    final_layers = [28, 32, 36, 40]
    x = np.arange(len(final_layers))
    width = 0.35

    child_final = [stats['by_layer'][l].get('child_output', {}).get('rank_gap_mean', 0) for l in final_layers]
    parent_final = [stats['by_layer'][l].get('parent_output', {}).get('rank_gap_mean', 0) for l in final_layers]

    bars1 = ax.bar(x - width/2, child_final, width, label=f'Child-output (n={counts.get("child_output", 0)})', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, parent_final, width, label=f'Parent-output (n={counts.get("parent_output", 0)})', color='magenta', alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Rank Gap', fontsize=12)
    ax.set_title('Rank Gap at Final Layers\n(Where the decision should crystallize)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(final_layers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / 'final_layers_comparison.png', dpi=150)
    plt.savefig(output_path / 'final_layers_comparison.pdf')
    plt.close()

    # =====================
    # Plot 4: Classification Distribution
    # =====================
    fig, ax = plt.subplots(figsize=(8, 6))

    classes = ['child_output', 'parent_output', 'both', 'neither']
    class_counts = [counts.get(c, 0) for c in classes]
    colors = ['green', 'magenta', 'orange', 'gray']

    bars = ax.bar(classes, class_counts, color=colors, alpha=0.7, edgecolor='black')

    # Add count labels on bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12)

    total = sum(class_counts)
    ax.set_xlabel('Classification', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Classification by Actual Generation Output (n={total})\nParent-output rate: {counts.get("parent_output", 0)/total*100:.1f}%', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / 'classification_distribution.png', dpi=150)
    plt.savefig(output_path / 'classification_distribution.pdf')
    plt.close()

    print(f"Visualizations saved to {output_path}")

    return {
        'rank_gap_by_output': str(output_path / 'rank_gap_by_output.png'),
        'rank_evolution_by_output': str(output_path / 'rank_evolution_by_output.png'),
        'final_layers_comparison': str(output_path / 'final_layers_comparison.png'),
        'classification_distribution': str(output_path / 'classification_distribution.png'),
    }


def print_summary(results):
    """Print a summary of the large sample analysis."""
    stats = results['statistics']
    layers = sorted(stats['by_layer'].keys())
    counts = stats['counts']

    print("\n" + "=" * 80)
    print("LOGIT LENS AT CONCEPT GENERATION - LARGE SAMPLE ANALYSIS")
    print("=" * 80)

    total = sum(counts.values())
    print(f"\nTotal samples: {total}")
    print(f"  Child-output: {counts.get('child_output', 0)} ({counts.get('child_output', 0)/total*100:.1f}%)")
    print(f"  Parent-output: {counts.get('parent_output', 0)} ({counts.get('parent_output', 0)/total*100:.1f}%)")
    print(f"  Both: {counts.get('both', 0)} ({counts.get('both', 0)/total*100:.1f}%)")
    print(f"  Neither: {counts.get('neither', 0)} ({counts.get('neither', 0)/total*100:.1f}%)")

    print("\n" + "-" * 80)
    print("RANK GAP BY LAYER (parent_rank - child_rank)")
    print("Positive = child preferred, Negative = parent preferred")
    print("-" * 80)
    print(f"{'Layer':<6} | {'Child-Output':<20} | {'Parent-Output':<20} | {'Divergence':<15}")
    print("-" * 80)

    for layer in layers:
        c_gap = stats['by_layer'][layer].get('child_output', {}).get('rank_gap_mean', 0)
        p_gap = stats['by_layer'][layer].get('parent_output', {}).get('rank_gap_mean', 0)
        divergence = c_gap - p_gap

        c_indicator = "child>>" if c_gap > 1000 else "child>" if c_gap > 0 else "parent>" if c_gap < -1000 else "~"
        p_indicator = "child>>" if p_gap > 1000 else "child>" if p_gap > 0 else "parent>" if p_gap < -1000 else "~"

        print(f"{layer:<6} | {c_gap:>+10.0f} {c_indicator:<8} | {p_gap:>+10.0f} {p_indicator:<8} | {divergence:>+10.0f}")

    print("\n" + "-" * 50)
    print("KEY FINDINGS")
    print("-" * 50)

    if stats.get('divergence_layer') is not None:
        print(f"\nDivergence layer: {stats['divergence_layer']}")
        print(f"  (First layer where child-output and parent-output gaps differ by >1000)")
    else:
        print("\nNo clear divergence layer found")

    # Layer 40 analysis
    c_gap_40 = stats['by_layer'].get(40, {}).get('child_output', {}).get('rank_gap_mean', 0)
    p_gap_40 = stats['by_layer'].get(40, {}).get('parent_output', {}).get('rank_gap_mean', 0)

    print(f"\nLayer 40 (final layer) rank gaps:")
    print(f"  Child-output: {c_gap_40:+.0f} ({'child preferred' if c_gap_40 > 0 else 'PARENT preferred'})")
    print(f"  Parent-output: {p_gap_40:+.0f} ({'child preferred' if p_gap_40 > 0 else 'PARENT preferred'})")

    if p_gap_40 < 0 and c_gap_40 > 0:
        print("\n  EXPECTED PATTERN CONFIRMED:")
        print("  - Child-output cases prefer child at layer 40")
        print("  - Parent-output cases prefer parent at layer 40")
    elif p_gap_40 > 0 and c_gap_40 > 0:
        print("\n  UNEXPECTED: Both groups prefer child at layer 40")
        print("  This suggests the logit lens may not capture the final decision mechanism")


@app.function(
    image=image,
    gpu="H100",
    timeout=2 * 60 * 60,  # 2 hours for large samples
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/results": results_vol,
    },
    secrets=[hf_secret],
)
def run_large_sample_analysis(
    pairs_path: str = "benchmark/matched_pairs_set1_pure.pkl",
    n_total: int = 500,
    layers: str = "0,4,8,12,16,20,24,28,32,36,40",
    max_new_tokens: int = 10,
):
    """Run logit lens analysis on a large sample, classified by actual output."""
    import sys
    import torch
    import pickle
    import random
    from tqdm import tqdm

    # Setup paths
    sys.path.insert(0, "/root/beyond-deduction")
    sys.path.insert(0, "/root/beyond-deduction/benchmark")

    # Parse layers
    target_layers = [int(x) for x in layers.split(",")]

    print("=" * 80)
    print("LOGIT LENS - LARGE SAMPLE ANALYSIS")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Pairs: {pairs_path}")
    print(f"Total samples: {n_total}")
    print(f"Target layers: {target_layers}")
    print(f"Max new tokens: {max_new_tokens}")
    print("=" * 80)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nLoading Gemma 2 9B IT...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded successfully")

    # Load data
    print("\nLoading data...")
    with open(f"/root/beyond-deduction/{pairs_path}", "rb") as f:
        pairs = pickle.load(f)

    print(f"Loaded {len(pairs)} pairs")

    # Sample from ALL examples (not filtered by factorial results)
    random.seed(42)
    all_indices = list(range(len(pairs)))
    sample_indices = random.sample(all_indices, min(n_total, len(pairs)))

    print(f"Sampling {len(sample_indices)} examples from all pairs")
    print(f"Expected parent-output: ~{len(sample_indices) * 0.05:.0f} (at 5% rate)")

    # Initialize results by class
    results_by_class = {
        'parent_output': [],
        'child_output': [],
        'both': [],
        'neither': [],
    }

    # Process all examples
    print("\nProcessing examples...")
    for i, idx in enumerate(tqdm(sample_indices)):
        h1, _ = pairs[idx]

        child_concept = h1['child_concept']
        parent_concept = h1['root_concept']

        prompt = format_prompt(h1['theories_nl'], h1['observations_nl'], tokenizer)

        # Run generation with logit lens
        result = run_single_example_generation(
            model, tokenizer, prompt,
            child_concept, parent_concept,
            target_layers,
            max_new_tokens=max_new_tokens,
            debug=(i < 5),  # Debug first 5
        )

        if result is not None:
            result['idx'] = idx
            classification = result['classification']
            results_by_class[classification].append(result)

        # Print progress every 100 examples
        if (i + 1) % 100 == 0:
            counts = {k: len(v) for k, v in results_by_class.items()}
            print(f"\n  Progress: {i+1}/{len(sample_indices)}")
            print(f"  Counts: {counts}")

    # Final counts
    counts = {k: len(v) for k, v in results_by_class.items()}
    print(f"\nFinal counts: {counts}")
    print(f"Parent-output rate: {counts['parent_output']/sum(counts.values())*100:.1f}%")

    # Compute statistics
    stats = compute_statistics_by_class(results_by_class, target_layers)

    results = {
        'config': {
            'n_total': n_total,
            'target_layers': target_layers,
            'max_new_tokens': max_new_tokens,
            'model': 'google/gemma-2-9b-it',
        },
        'results_by_class': results_by_class,
        'statistics': stats,
    }

    # Print summary
    print_summary(results)

    # Create visualizations
    print("\nCreating visualizations...")
    viz_paths = create_visualizations(results, "/root/results", highlight_layers=[8, 12, 20, 40])

    # Save results
    with open("/root/results/large_sample_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Results saved to /root/results/large_sample_results.pkl")

    results_vol.commit()

    # Build summary for return
    summary = {
        'n_total': sum(counts.values()),
        'counts': counts,
        'parent_output_rate': counts['parent_output'] / sum(counts.values()),
        'target_layers': target_layers,
        'divergence_layer': stats.get('divergence_layer'),
        'layer_40': {
            'child_output_gap': stats['by_layer'].get(40, {}).get('child_output', {}).get('rank_gap_mean', 0),
            'parent_output_gap': stats['by_layer'].get(40, {}).get('parent_output', {}).get('rank_gap_mean', 0),
        },
    }

    return summary


@app.function(
    image=image,
    volumes={"/root/results": results_vol},
)
def download_results():
    """Download results from Modal volume."""
    import os

    files = {}
    results_dir = "/root/results"

    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            filepath = os.path.join(results_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, "rb") as f:
                    files[filename] = f.read()

    return files


@app.local_entrypoint()
def main(
    pairs: str = "benchmark/matched_pairs_set1_pure.pkl",
    n_total: int = 500,
    layers: str = "0,4,8,12,16,20,24,28,32,36,40",
    max_new_tokens: int = 10,
):
    """Run large sample logit lens analysis."""
    import json

    print("=" * 80)
    print("LOGIT LENS - LARGE SAMPLE ANALYSIS")
    print("=" * 80)
    print(f"Pairs: {pairs}")
    print(f"Total samples: {n_total}")
    print(f"Layers: {layers}")
    print(f"Max new tokens: {max_new_tokens}")
    print("=" * 80)

    # Run analysis
    summary = run_large_sample_analysis.remote(
        pairs_path=pairs,
        n_total=n_total,
        layers=layers,
        max_new_tokens=max_new_tokens,
    )

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2, default=str))

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    counts = summary.get('counts', {})
    parent_rate = summary.get('parent_output_rate', 0)
    layer_40 = summary.get('layer_40', {})

    print(f"""
Sample Distribution:
  Child-output: {counts.get('child_output', 0)}
  Parent-output: {counts.get('parent_output', 0)} ({parent_rate*100:.1f}%)
  Both: {counts.get('both', 0)}
  Neither: {counts.get('neither', 0)}

Layer 40 Rank Gaps:
  Child-output: {layer_40.get('child_output_gap', 0):+.0f}
  Parent-output: {layer_40.get('parent_output_gap', 0):+.0f}
""")

    c_gap = layer_40.get('child_output_gap', 0)
    p_gap = layer_40.get('parent_output_gap', 0)

    if c_gap > 0 and p_gap < 0:
        print("""
KEY FINDING: EXPECTED DIVERGENCE CONFIRMED

At layer 40:
- Child-output cases show CHILD preference (positive gap)
- Parent-output cases show PARENT preference (negative gap)

This confirms that the logit lens at concept generation step
captures the actual decision mechanism.
""")
    elif c_gap > 0 and p_gap > 0:
        gap_diff = c_gap - p_gap
        if gap_diff > 1000:
            print(f"""
PARTIAL DIVERGENCE: Both prefer child, but child-output has stronger preference

Gap difference: {gap_diff:+.0f}
- Child-output cases have {gap_diff:.0f} higher rank gap than parent-output

The parent-output decision may happen at an even later stage (unembedding),
or through a different mechanism not captured by logit lens.
""")
        else:
            print("""
NO CLEAR DIVERGENCE: Both groups show similar child preference

This suggests the concept decision happens through a mechanism
not captured by the logit lens projection.
""")

    # Download results
    print("\nDownloading results...")
    files = download_results.remote()

    local_path = PROJECT_DIR / "mi" / "logit_lens_generation_results"
    local_path.mkdir(exist_ok=True)

    for filename, data in files.items():
        filepath = local_path / filename
        with open(filepath, "wb") as f:
            f.write(data)
        print(f"  Saved: {filepath}")

    print(f"\nAll results saved to {local_path}")
