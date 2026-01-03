"""
Logit Lens Analysis for Gemma 2 9B IT
======================================

Tracks how P(child) vs P(parent) concept tokens evolve across layers.

The logit lens technique projects intermediate hidden states through the
unembedding matrix to see what the model "would predict" at each layer.

Key questions:
- When does the probability gap between child and parent emerge?
- Is the decision made early (supporting probe findings) or late?
- Do H1 successes and failures diverge at a specific layer?

Usage:
    from mi.logit_lens import run_logit_lens_analysis
    results = run_logit_lens_analysis(model, tokenizer, pairs, h1_results)
"""

import torch
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Modal


@dataclass
class LogitLensConfig:
    """Configuration for logit lens analysis."""

    # Layers to analyze (Gemma 2 9B has 42 layers)
    target_layers: List[int] = field(default_factory=lambda: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])

    # Number of examples per group
    n_per_group: int = 25  # 25 successes, 25 failures = 50 total

    # Key layers to highlight in plots
    highlight_layers: List[int] = field(default_factory=lambda: [8, 12])

    # Random seed
    seed: int = 42

    # Output directory
    output_dir: str = "logit_lens_results"


@dataclass
class LogitLensResult:
    """Result for a single example."""

    idx: int
    is_success: bool

    # Concepts
    child_concept: str
    parent_concept: str

    # Token IDs
    child_token_id: int
    parent_token_id: int

    # Probabilities at each layer
    # layer -> (p_child, p_parent)
    layer_probs: Dict[int, Tuple[float, float]]

    # Gap at each layer (p_child - p_parent)
    layer_gaps: Dict[int, float]

    # Ranks at each layer (lower = more likely)
    # layer -> (rank_child, rank_parent)
    layer_ranks: Optional[Dict[int, Tuple[int, int]]] = None

    # Rank gap at each layer (rank_parent - rank_child)
    # Positive = child preferred, Negative = parent preferred
    layer_rank_gaps: Optional[Dict[int, int]] = None


SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
Only output final hypotheses."""


def format_prompt(theories_nl: str, observations_nl: str, tokenizer=None) -> str:
    """Format prompt for Gemma 2 IT with proper chat template."""
    user_prompt = f"Q: {theories_nl} We observe that: {observations_nl} Please come up with hypothesis to explain observations."
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    if tokenizer is not None:
        messages = [{"role": "user", "content": full_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return full_prompt


def find_concept_token_id(tokenizer, concept: str, return_all: bool = False):
    """
    Find the token ID(s) for a concept.

    Concepts in this benchmark are nonsense words (like "lerpant", "timple")
    that tokenize into multiple tokens. We return the FIRST token with a
    leading space, since that's how the concept would appear in generated text.

    Args:
        tokenizer: The tokenizer
        concept: The concept string
        return_all: If True, return all tokens; otherwise just the first

    Returns:
        int (first token id) or list of ints (if return_all=True)
    """
    # Try with leading space first (most common in generation context)
    tokens_with_space = tokenizer.encode(f" {concept}", add_special_tokens=False)

    # Also try capitalized (might appear at start of sentence)
    tokens_capitalized = tokenizer.encode(f" {concept.capitalize()}", add_special_tokens=False)

    # Use whichever gives a valid result
    if tokens_with_space:
        tokens = tokens_with_space
    elif tokens_capitalized:
        tokens = tokens_capitalized
    else:
        # Fallback: try without space
        tokens = tokenizer.encode(concept, add_special_tokens=False)

    if not tokens:
        return None if not return_all else []

    if return_all:
        return tokens

    return tokens[0]


def find_concept_tokens_info(tokenizer, concept: str) -> dict:
    """
    Get detailed tokenization info for a concept.

    Useful for debugging and understanding how concepts tokenize.
    """
    tokens_plain = tokenizer.encode(concept, add_special_tokens=False)
    tokens_space = tokenizer.encode(f" {concept}", add_special_tokens=False)
    tokens_cap = tokenizer.encode(concept.capitalize(), add_special_tokens=False)
    tokens_space_cap = tokenizer.encode(f" {concept.capitalize()}", add_special_tokens=False)

    return {
        "concept": concept,
        "plain": {
            "tokens": tokens_plain,
            "decoded": [tokenizer.decode([t]) for t in tokens_plain]
        },
        "with_space": {
            "tokens": tokens_space,
            "decoded": [tokenizer.decode([t]) for t in tokens_space]
        },
        "capitalized": {
            "tokens": tokens_cap,
            "decoded": [tokenizer.decode([t]) for t in tokens_cap]
        },
        "space_capitalized": {
            "tokens": tokens_space_cap,
            "decoded": [tokenizer.decode([t]) for t in tokens_space_cap]
        },
        "first_token": tokens_space[0] if tokens_space else None,
    }


def get_early_logits(
    model,
    hidden_states: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """
    Project hidden states through the unembedding matrix to get early logits.

    For Gemma 2, we need to apply RMSNorm before the lm_head.

    Args:
        model: Gemma 2 model
        hidden_states: Hidden states tensor [batch, seq_len, hidden_size]
        layer_idx: Layer index (for reference)

    Returns:
        Logits tensor [batch, seq_len, vocab_size]
    """
    # Apply final layer norm (RMSNorm for Gemma 2)
    normed = model.model.norm(hidden_states)

    # Project through unembedding (lm_head)
    logits = model.lm_head(normed)

    return logits


def run_single_example(
    model,
    tokenizer,
    prompt: str,
    child_token_id: int,
    parent_token_id: int,
    target_layers: List[int],
    return_extended: bool = False,
) -> Dict:
    """
    Run logit lens on a single example.

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: Input prompt
        child_token_id: Token ID for child concept
        parent_token_id: Token ID for parent concept
        target_layers: Layers to analyze
        return_extended: If True, return extended info (log probs, top tokens)

    Returns:
        Dict with 'probs', 'ranks' at minimum
        If return_extended=True, also includes 'log_probs', 'top_tokens'
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # hidden_states is tuple of (embedding, layer_0, layer_1, ..., layer_N)
    # So hidden_states[i] is the output after layer i-1 (or embedding for i=0)
    # hidden_states[layer+1] gives output after layer `layer`
    hidden_states = outputs.hidden_states

    layer_probs = {}
    layer_ranks = {}
    layer_log_probs = {}
    layer_top_tokens = {}

    for layer in target_layers:
        # Get hidden state after this layer
        # hidden_states[0] is embedding, hidden_states[1] is after layer 0, etc.
        hs = hidden_states[layer + 1]  # [batch, seq_len, hidden_size]

        # Get early logits
        early_logits = get_early_logits(model, hs, layer)

        # Get logits at last token position
        last_logits = early_logits[0, -1, :]  # [vocab_size]

        # Convert to probabilities
        probs = F.softmax(last_logits, dim=-1)

        # Extract probabilities for child and parent tokens
        p_child = probs[child_token_id].item()
        p_parent = probs[parent_token_id].item()
        layer_probs[layer] = (p_child, p_parent)

        # Always compute ranks (this is the primary metric for rare tokens)
        sorted_indices = torch.argsort(last_logits, descending=True)
        ranks = torch.zeros_like(last_logits, dtype=torch.long)
        ranks[sorted_indices] = torch.arange(len(last_logits), device=last_logits.device)

        child_rank = ranks[child_token_id].item()
        parent_rank = ranks[parent_token_id].item()
        layer_ranks[layer] = (child_rank, parent_rank)

        if return_extended:
            # Log probabilities
            log_probs = F.log_softmax(last_logits, dim=-1)
            log_p_child = log_probs[child_token_id].item()
            log_p_parent = log_probs[parent_token_id].item()
            layer_log_probs[layer] = (log_p_child, log_p_parent)

            # Top 5 tokens
            top_k = 5
            top_indices = sorted_indices[:top_k].cpu().numpy()
            top_tokens_decoded = [tokenizer.decode([idx]) for idx in top_indices]
            layer_top_tokens[layer] = top_tokens_decoded

    result = {
        "probs": layer_probs,
        "ranks": layer_ranks,
    }

    if return_extended:
        result["log_probs"] = layer_log_probs
        result["top_tokens"] = layer_top_tokens

    return result


def run_logit_lens_analysis(
    model,
    tokenizer,
    pairs: List,
    h1_results: List,
    config: LogitLensConfig,
    verbose: bool = True,
    debug_first_n: int = 2,
) -> Dict:
    """
    Run logit lens analysis on matched pairs.

    Args:
        model: Gemma 2 model
        tokenizer: Tokenizer
        pairs: List of (h1, h2) matched pairs
        h1_results: List of H1 results with 'strong' accuracy
        config: Configuration
        verbose: Print progress
        debug_first_n: Print detailed debug info for first N examples

    Returns:
        Dict with analysis results and statistics
    """
    import random
    random.seed(config.seed)

    # Identify successes and failures
    success_indices = [i for i, r in enumerate(h1_results) if r['strong'] == 1]
    failure_indices = [i for i, r in enumerate(h1_results) if r['strong'] == 0]

    if verbose:
        print(f"H1 Successes (child output): {len(success_indices)}")
        print(f"H1 Failures (parent output): {len(failure_indices)}")

    # Sample examples
    n_success = min(config.n_per_group, len(success_indices))
    n_failure = min(config.n_per_group, len(failure_indices))

    sampled_successes = random.sample(success_indices, n_success)
    sampled_failures = random.sample(failure_indices, n_failure)

    if verbose:
        print(f"\nSampled {n_success} successes, {n_failure} failures")

    # Run analysis
    success_results = []
    failure_results = []
    debug_info = []

    # Process successes
    if verbose:
        print("\nProcessing successes...")

    for i, idx in enumerate(tqdm(sampled_successes, disable=not verbose)):
        h1, _ = pairs[idx]

        child_concept = h1['child_concept']
        parent_concept = h1['root_concept']

        # Find token IDs
        child_token_id = find_concept_token_id(tokenizer, child_concept)
        parent_token_id = find_concept_token_id(tokenizer, parent_concept)

        if child_token_id is None or parent_token_id is None:
            if verbose:
                print(f"  Skipping idx {idx}: couldn't find tokens for {child_concept}/{parent_concept}")
            continue

        # Format prompt
        prompt = format_prompt(h1['theories_nl'], h1['observations_nl'], tokenizer)

        # Run logit lens with extended info for debugging
        should_debug = i < debug_first_n
        extended_result = run_single_example(
            model, tokenizer, prompt,
            child_token_id, parent_token_id,
            config.target_layers,
            return_extended=should_debug,
        )

        layer_probs = extended_result["probs"]
        layer_ranks = extended_result["ranks"]

        if should_debug:
            # Store debug info
            debug_info.append({
                "idx": idx,
                "is_success": True,
                "child_concept": child_concept,
                "parent_concept": parent_concept,
                "child_token_id": child_token_id,
                "parent_token_id": parent_token_id,
                "child_token_decoded": tokenizer.decode([child_token_id]),
                "parent_token_decoded": tokenizer.decode([parent_token_id]),
                "ranks": layer_ranks,
                "log_probs": extended_result.get("log_probs", {}),
                "top_tokens": extended_result.get("top_tokens", {}),
            })

        # Compute gaps
        layer_gaps = {layer: p[0] - p[1] for layer, p in layer_probs.items()}
        # Rank gap: parent_rank - child_rank (positive = child preferred)
        layer_rank_gaps = {layer: r[1] - r[0] for layer, r in layer_ranks.items()}

        result = LogitLensResult(
            idx=idx,
            is_success=True,
            child_concept=child_concept,
            parent_concept=parent_concept,
            child_token_id=child_token_id,
            parent_token_id=parent_token_id,
            layer_probs=layer_probs,
            layer_gaps=layer_gaps,
            layer_ranks=layer_ranks,
            layer_rank_gaps=layer_rank_gaps,
        )
        success_results.append(result)

    # Process failures
    if verbose:
        print("\nProcessing failures...")

    for i, idx in enumerate(tqdm(sampled_failures, disable=not verbose)):
        h1, _ = pairs[idx]

        child_concept = h1['child_concept']
        parent_concept = h1['root_concept']

        # Find token IDs
        child_token_id = find_concept_token_id(tokenizer, child_concept)
        parent_token_id = find_concept_token_id(tokenizer, parent_concept)

        if child_token_id is None or parent_token_id is None:
            if verbose:
                print(f"  Skipping idx {idx}: couldn't find tokens for {child_concept}/{parent_concept}")
            continue

        # Format prompt
        prompt = format_prompt(h1['theories_nl'], h1['observations_nl'], tokenizer)

        # Run logit lens with extended info for debugging
        should_debug = i < debug_first_n
        extended_result = run_single_example(
            model, tokenizer, prompt,
            child_token_id, parent_token_id,
            config.target_layers,
            return_extended=should_debug,
        )

        layer_probs = extended_result["probs"]
        layer_ranks = extended_result["ranks"]

        if should_debug:
            # Store debug info
            debug_info.append({
                "idx": idx,
                "is_success": False,
                "child_concept": child_concept,
                "parent_concept": parent_concept,
                "child_token_id": child_token_id,
                "parent_token_id": parent_token_id,
                "child_token_decoded": tokenizer.decode([child_token_id]),
                "parent_token_decoded": tokenizer.decode([parent_token_id]),
                "ranks": layer_ranks,
                "log_probs": extended_result.get("log_probs", {}),
                "top_tokens": extended_result.get("top_tokens", {}),
            })

        # Compute gaps
        layer_gaps = {layer: p[0] - p[1] for layer, p in layer_probs.items()}
        # Rank gap: parent_rank - child_rank (positive = child preferred)
        layer_rank_gaps = {layer: r[1] - r[0] for layer, r in layer_ranks.items()}

        result = LogitLensResult(
            idx=idx,
            is_success=False,
            child_concept=child_concept,
            parent_concept=parent_concept,
            child_token_id=child_token_id,
            parent_token_id=parent_token_id,
            layer_probs=layer_probs,
            layer_gaps=layer_gaps,
            layer_ranks=layer_ranks,
            layer_rank_gaps=layer_rank_gaps,
        )
        failure_results.append(result)

    # Print debug info
    if verbose and debug_info:
        print("\n" + "=" * 60)
        print("DEBUG INFO (first examples)")
        print("=" * 60)
        for info in debug_info:
            print(f"\nIdx {info['idx']} ({'SUCCESS' if info['is_success'] else 'FAILURE'}):")
            print(f"  Child: '{info['child_concept']}' -> token {info['child_token_id']} = '{info['child_token_decoded']}'")
            print(f"  Parent: '{info['parent_concept']}' -> token {info['parent_token_id']} = '{info['parent_token_decoded']}'")
            print(f"  Layer 40 top tokens: {info['top_tokens'].get(40, 'N/A')}")
            print(f"  Layer 40 ranks: child={info['ranks'].get(40, ('N/A',))[0]}, parent={info['ranks'].get(40, ('N/A',))[1]}")
            print(f"  Layer 40 log_probs: child={info['log_probs'].get(40, ('N/A',))[0]:.2f}, parent={info['log_probs'].get(40, ('N/A',))[1]:.2f}")

    # Aggregate results
    if verbose:
        print(f"\nAnalyzed {len(success_results)} successes, {len(failure_results)} failures")

    # Compute statistics
    stats = compute_statistics(success_results, failure_results, config.target_layers)

    return {
        "config": {
            "target_layers": config.target_layers,
            "n_per_group": config.n_per_group,
            "seed": config.seed,
        },
        "success_results": [vars(r) for r in success_results],
        "failure_results": [vars(r) for r in failure_results],
        "statistics": stats,
        "debug_info": debug_info,
    }


def compute_statistics(
    success_results: List[LogitLensResult],
    failure_results: List[LogitLensResult],
    target_layers: List[int],
) -> Dict:
    """Compute aggregate statistics from individual results."""

    stats = {
        "n_success": len(success_results),
        "n_failure": len(failure_results),
        "by_layer": {},
    }

    for layer in target_layers:
        # Successes - probabilities
        success_p_child = [r.layer_probs[layer][0] for r in success_results if layer in r.layer_probs]
        success_p_parent = [r.layer_probs[layer][1] for r in success_results if layer in r.layer_probs]
        success_gaps = [r.layer_gaps[layer] for r in success_results if layer in r.layer_gaps]

        # Successes - ranks
        success_child_ranks = [r.layer_ranks[layer][0] for r in success_results if r.layer_ranks and layer in r.layer_ranks]
        success_parent_ranks = [r.layer_ranks[layer][1] for r in success_results if r.layer_ranks and layer in r.layer_ranks]
        success_rank_gaps = [r.layer_rank_gaps[layer] for r in success_results if r.layer_rank_gaps and layer in r.layer_rank_gaps]

        # Failures - probabilities
        failure_p_child = [r.layer_probs[layer][0] for r in failure_results if layer in r.layer_probs]
        failure_p_parent = [r.layer_probs[layer][1] for r in failure_results if layer in r.layer_probs]
        failure_gaps = [r.layer_gaps[layer] for r in failure_results if layer in r.layer_gaps]

        # Failures - ranks
        failure_child_ranks = [r.layer_ranks[layer][0] for r in failure_results if r.layer_ranks and layer in r.layer_ranks]
        failure_parent_ranks = [r.layer_ranks[layer][1] for r in failure_results if r.layer_ranks and layer in r.layer_ranks]
        failure_rank_gaps = [r.layer_rank_gaps[layer] for r in failure_results if r.layer_rank_gaps and layer in r.layer_rank_gaps]

        stats["by_layer"][layer] = {
            "success": {
                "p_child_mean": float(np.mean(success_p_child)) if success_p_child else 0,
                "p_child_std": float(np.std(success_p_child)) if success_p_child else 0,
                "p_parent_mean": float(np.mean(success_p_parent)) if success_p_parent else 0,
                "p_parent_std": float(np.std(success_p_parent)) if success_p_parent else 0,
                "gap_mean": float(np.mean(success_gaps)) if success_gaps else 0,
                "gap_std": float(np.std(success_gaps)) if success_gaps else 0,
                # Ranks (lower = more likely)
                "child_rank_mean": float(np.mean(success_child_ranks)) if success_child_ranks else 0,
                "child_rank_std": float(np.std(success_child_ranks)) if success_child_ranks else 0,
                "parent_rank_mean": float(np.mean(success_parent_ranks)) if success_parent_ranks else 0,
                "parent_rank_std": float(np.std(success_parent_ranks)) if success_parent_ranks else 0,
                "rank_gap_mean": float(np.mean(success_rank_gaps)) if success_rank_gaps else 0,
                "rank_gap_std": float(np.std(success_rank_gaps)) if success_rank_gaps else 0,
            },
            "failure": {
                "p_child_mean": float(np.mean(failure_p_child)) if failure_p_child else 0,
                "p_child_std": float(np.std(failure_p_child)) if failure_p_child else 0,
                "p_parent_mean": float(np.mean(failure_p_parent)) if failure_p_parent else 0,
                "p_parent_std": float(np.std(failure_p_parent)) if failure_p_parent else 0,
                "gap_mean": float(np.mean(failure_gaps)) if failure_gaps else 0,
                "gap_std": float(np.std(failure_gaps)) if failure_gaps else 0,
                # Ranks
                "child_rank_mean": float(np.mean(failure_child_ranks)) if failure_child_ranks else 0,
                "child_rank_std": float(np.std(failure_child_ranks)) if failure_child_ranks else 0,
                "parent_rank_mean": float(np.mean(failure_parent_ranks)) if failure_parent_ranks else 0,
                "parent_rank_std": float(np.std(failure_parent_ranks)) if failure_parent_ranks else 0,
                "rank_gap_mean": float(np.mean(failure_rank_gaps)) if failure_rank_gaps else 0,
                "rank_gap_std": float(np.std(failure_rank_gaps)) if failure_rank_gaps else 0,
            },
        }

    # Find decision emergence and locking points using RANKS
    # Rank gap > 0 means child preferred (parent_rank > child_rank)
    success_rank_gaps = [stats["by_layer"][l]["success"]["rank_gap_mean"] for l in target_layers]
    failure_rank_gaps = [stats["by_layer"][l]["failure"]["rank_gap_mean"] for l in target_layers]

    # Decision emergence: first layer where rank gap is significantly positive (>100) or negative
    success_emergence = None
    failure_emergence = None
    for i, layer in enumerate(target_layers):
        if success_rank_gaps[i] > 100 and success_emergence is None:
            success_emergence = layer
        if failure_rank_gaps[i] < -100 and failure_emergence is None:
            failure_emergence = layer

    stats["decision_emergence"] = {
        "success": success_emergence,
        "failure": failure_emergence,
    }

    # Max rank gap achieved
    stats["max_rank_gap"] = {
        "success": max(success_rank_gaps) if success_rank_gaps else 0,
        "failure": min(failure_rank_gaps) if failure_rank_gaps else 0,
    }

    # Legacy probability-based metrics
    success_gaps = [stats["by_layer"][l]["success"]["gap_mean"] for l in target_layers]
    failure_gaps = [stats["by_layer"][l]["failure"]["gap_mean"] for l in target_layers]

    stats["max_gap"] = {
        "success": max(success_gaps) if success_gaps else 0,
        "failure": min(failure_gaps) if failure_gaps else 0,
    }

    return stats


def create_visualizations(results: Dict, output_dir: str, highlight_layers: List[int] = None):
    """Create and save visualizations."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = results["statistics"]
    layers = sorted(stats["by_layer"].keys())

    if highlight_layers is None:
        highlight_layers = [8, 12]

    # =====================
    # Plot 0: RANK Evolution (PRIMARY METRIC for rare tokens)
    # =====================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Successes
    ax = axes[0]
    success_child_ranks = [stats["by_layer"][l]["success"]["child_rank_mean"] for l in layers]
    success_parent_ranks = [stats["by_layer"][l]["success"]["parent_rank_mean"] for l in layers]

    ax.plot(layers, success_child_ranks, 'b-o', label='Child rank', linewidth=2)
    ax.plot(layers, success_parent_ranks, 'r-s', label='Parent rank', linewidth=2)

    for hl in highlight_layers:
        ax.axvline(x=hl, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Rank (lower = more likely)', fontsize=12)
    ax.set_title(f'H1 Successes (n={stats["n_success"]}): Child Should Win', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for ranks

    # Right: Failures
    ax = axes[1]
    failure_child_ranks = [stats["by_layer"][l]["failure"]["child_rank_mean"] for l in layers]
    failure_parent_ranks = [stats["by_layer"][l]["failure"]["parent_rank_mean"] for l in layers]

    ax.plot(layers, failure_child_ranks, 'b-o', label='Child rank', linewidth=2)
    ax.plot(layers, failure_parent_ranks, 'r-s', label='Parent rank', linewidth=2)

    for hl in highlight_layers:
        ax.axvline(x=hl, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Rank (lower = more likely)', fontsize=12)
    ax.set_title(f'H1 Failures (n={stats["n_failure"]}): Parent Wins (incorrectly)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path / 'rank_evolution.png', dpi=150)
    plt.savefig(output_path / 'rank_evolution.pdf')
    plt.close()

    # =====================
    # Plot 0b: Rank Gap Evolution
    # =====================
    fig, ax = plt.subplots(figsize=(10, 6))

    success_rank_gaps = [stats["by_layer"][l]["success"]["rank_gap_mean"] for l in layers]
    failure_rank_gaps = [stats["by_layer"][l]["failure"]["rank_gap_mean"] for l in layers]

    ax.plot(layers, success_rank_gaps, 'g-o', label='Successes (positive = child preferred)', linewidth=2)
    ax.plot(layers, failure_rank_gaps, 'm-s', label='Failures (negative = parent preferred)', linewidth=2)

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    for hl in highlight_layers:
        ax.axvline(x=hl, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Rank Gap (parent_rank - child_rank)', fontsize=12)
    ax.set_title('Rank Gap Evolution: When Does Child/Parent Preference Emerge?', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'rank_gap_evolution.png', dpi=150)
    plt.savefig(output_path / 'rank_gap_evolution.pdf')
    plt.close()

    # =====================
    # Plot 1: Probability Evolution
    # =====================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Successes
    ax = axes[0]
    success_p_child = [stats["by_layer"][l]["success"]["p_child_mean"] for l in layers]
    success_p_parent = [stats["by_layer"][l]["success"]["p_parent_mean"] for l in layers]
    success_p_child_std = [stats["by_layer"][l]["success"]["p_child_std"] for l in layers]
    success_p_parent_std = [stats["by_layer"][l]["success"]["p_parent_std"] for l in layers]

    ax.plot(layers, success_p_child, 'b-o', label='P(child)', linewidth=2)
    ax.fill_between(layers,
                   [p - s for p, s in zip(success_p_child, success_p_child_std)],
                   [p + s for p, s in zip(success_p_child, success_p_child_std)],
                   alpha=0.2, color='blue')

    ax.plot(layers, success_p_parent, 'r-s', label='P(parent)', linewidth=2)
    ax.fill_between(layers,
                   [p - s for p, s in zip(success_p_parent, success_p_parent_std)],
                   [p + s for p, s in zip(success_p_parent, success_p_parent_std)],
                   alpha=0.2, color='red')

    # Highlight key layers
    for hl in highlight_layers:
        ax.axvline(x=hl, color='gray', linestyle='--', alpha=0.5, label=f'Layer {hl}' if hl == highlight_layers[0] else '')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'H1 Successes (n={stats["n_success"]}): Child Output', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Right: Failures
    ax = axes[1]
    failure_p_child = [stats["by_layer"][l]["failure"]["p_child_mean"] for l in layers]
    failure_p_parent = [stats["by_layer"][l]["failure"]["p_parent_mean"] for l in layers]
    failure_p_child_std = [stats["by_layer"][l]["failure"]["p_child_std"] for l in layers]
    failure_p_parent_std = [stats["by_layer"][l]["failure"]["p_parent_std"] for l in layers]

    ax.plot(layers, failure_p_child, 'b-o', label='P(child)', linewidth=2)
    ax.fill_between(layers,
                   [p - s for p, s in zip(failure_p_child, failure_p_child_std)],
                   [p + s for p, s in zip(failure_p_child, failure_p_child_std)],
                   alpha=0.2, color='blue')

    ax.plot(layers, failure_p_parent, 'r-s', label='P(parent)', linewidth=2)
    ax.fill_between(layers,
                   [p - s for p, s in zip(failure_p_parent, failure_p_parent_std)],
                   [p + s for p, s in zip(failure_p_parent, failure_p_parent_std)],
                   alpha=0.2, color='red')

    # Highlight key layers
    for hl in highlight_layers:
        ax.axvline(x=hl, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'H1 Failures (n={stats["n_failure"]}): Parent Output', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path / 'probability_evolution.png', dpi=150)
    plt.savefig(output_path / 'probability_evolution.pdf')
    plt.close()

    # =====================
    # Plot 2: Gap Evolution
    # =====================
    fig, ax = plt.subplots(figsize=(10, 6))

    success_gaps = [stats["by_layer"][l]["success"]["gap_mean"] for l in layers]
    success_gaps_std = [stats["by_layer"][l]["success"]["gap_std"] for l in layers]
    failure_gaps = [stats["by_layer"][l]["failure"]["gap_mean"] for l in layers]
    failure_gaps_std = [stats["by_layer"][l]["failure"]["gap_std"] for l in layers]

    ax.plot(layers, success_gaps, 'g-o', label='Successes (child > parent)', linewidth=2)
    ax.fill_between(layers,
                   [g - s for g, s in zip(success_gaps, success_gaps_std)],
                   [g + s for g, s in zip(success_gaps, success_gaps_std)],
                   alpha=0.2, color='green')

    ax.plot(layers, failure_gaps, 'm-s', label='Failures (parent > child)', linewidth=2)
    ax.fill_between(layers,
                   [g - s for g, s in zip(failure_gaps, failure_gaps_std)],
                   [g + s for g, s in zip(failure_gaps, failure_gaps_std)],
                   alpha=0.2, color='magenta')

    # Decision threshold lines
    ax.axhline(y=0.1, color='gray', linestyle=':', alpha=0.5, label='Decision threshold (+/- 0.1)')
    ax.axhline(y=-0.1, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Highlight key layers
    for hl in highlight_layers:
        ax.axvline(x=hl, color='gray', linestyle='--', alpha=0.5, label=f'Layer {hl}' if hl == highlight_layers[0] else '')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Gap: P(child) - P(parent)', fontsize=12)
    ax.set_title('Decision Crystallization: When Does the Model Commit?', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'gap_evolution.png', dpi=150)
    plt.savefig(output_path / 'gap_evolution.pdf')
    plt.close()

    # =====================
    # Plot 3: Layer-by-Layer Comparison
    # =====================
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(layers))
    width = 0.35

    # Success group
    ax.bar(x - width/2, success_gaps, width, label='Successes', color='green', alpha=0.7)
    ax.bar(x + width/2, failure_gaps, width, label='Failures', color='magenta', alpha=0.7)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Gap: P(child) - P(parent)', fontsize=12)
    ax.set_title('Gap Comparison by Layer', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / 'gap_comparison.png', dpi=150)
    plt.savefig(output_path / 'gap_comparison.pdf')
    plt.close()

    print(f"Visualizations saved to {output_path}")

    return {
        "rank_evolution": str(output_path / 'rank_evolution.png'),
        "rank_gap_evolution": str(output_path / 'rank_gap_evolution.png'),
        "probability_evolution": str(output_path / 'probability_evolution.png'),
        "gap_evolution": str(output_path / 'gap_evolution.png'),
        "gap_comparison": str(output_path / 'gap_comparison.png'),
    }


def print_summary(results: Dict):
    """Print a summary of the logit lens analysis."""

    stats = results["statistics"]
    layers = sorted(stats["by_layer"].keys())

    print("\n" + "=" * 70)
    print("LOGIT LENS ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nSamples: {stats['n_success']} successes, {stats['n_failure']} failures")

    # PRIMARY METRIC: RANK EVOLUTION (better for rare tokens)
    print("\n" + "-" * 80)
    print("RANK EVOLUTION BY LAYER (lower = more likely)")
    print("-" * 80)
    print(f"{'Layer':<8} | {'Success Child':<15} | {'Success Parent':<15} | {'Failure Child':<15} | {'Failure Parent':<15}")
    print("-" * 80)

    for layer in layers:
        s = stats["by_layer"][layer]["success"]
        f = stats["by_layer"][layer]["failure"]
        print(f"{layer:<8} | {s['child_rank_mean']:>10.0f}     | {s['parent_rank_mean']:>10.0f}     | "
              f"{f['child_rank_mean']:>10.0f}     | {f['parent_rank_mean']:>10.0f}")

    print("\n" + "-" * 50)
    print("RANK GAP EVOLUTION (parent_rank - child_rank)")
    print("Positive = child preferred, Negative = parent preferred")
    print("-" * 50)
    print(f"{'Layer':<8} | {'Success Gap':<20} | {'Failure Gap':<20}")
    print("-" * 50)

    for layer in layers:
        s_gap = stats["by_layer"][layer]["success"]["rank_gap_mean"]
        f_gap = stats["by_layer"][layer]["failure"]["rank_gap_mean"]
        # Add visual indicator
        s_indicator = "child>>" if s_gap > 100 else "child>" if s_gap > 0 else "parent>" if s_gap < -100 else "~"
        f_indicator = "child>>" if f_gap > 100 else "child>" if f_gap > 0 else "parent>" if f_gap < -100 else "~"
        print(f"{layer:<8} | {s_gap:>+10.0f} {s_indicator:<8} | {f_gap:>+10.0f} {f_indicator:<8}")

    print("\n" + "-" * 50)
    print("KEY METRICS")
    print("-" * 50)

    emergence = stats.get("decision_emergence", {})
    print(f"Decision emergence (rank gap > 100):")
    print(f"  Successes: Layer {emergence.get('success', 'N/A')}")
    print(f"  Failures:  Layer {emergence.get('failure', 'N/A')}")

    max_rank_gap = stats.get("max_rank_gap", {})
    print(f"\nMax rank gap achieved:")
    print(f"  Successes: {max_rank_gap.get('success', 0):+.0f}")
    print(f"  Failures:  {max_rank_gap.get('failure', 0):+.0f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Check if rank gaps diverge early using RANKS (not probabilities)
    layer_8_success_rank_gap = stats["by_layer"].get(8, {}).get("success", {}).get("rank_gap_mean", 0)
    layer_8_failure_rank_gap = stats["by_layer"].get(8, {}).get("failure", {}).get("rank_gap_mean", 0)

    if layer_8_success_rank_gap > 100 and layer_8_failure_rank_gap < 0:
        print(f"""
EARLY DECISION CRYSTALLIZATION (by layer 8)

Layer 8 rank gaps:
  Successes: {layer_8_success_rank_gap:+.0f} (child ranked much higher than parent)
  Failures:  {layer_8_failure_rank_gap:+.0f} (parent ranked higher/similar to child)

The model shows clear child vs parent preference by layer 8.
This SUPPORTS the probing finding that decision is readable at layer 8.

Combined with activation patching (0% causal effect at L8), this means:
- The decision IS encoded early (readable by layer 8)
- But the encoding is DISTRIBUTED (no single causal locus)
- Layer 8 reflects an early "vote" that gets amplified later
""")
    elif layer_8_success_rank_gap > 0:
        print(f"""
GRADUAL DECISION EMERGENCE

Layer 8 rank gaps:
  Successes: {layer_8_success_rank_gap:+.0f}
  Failures:  {layer_8_failure_rank_gap:+.0f}

The model shows some child preference in successes by layer 8,
but the gap is not as clear-cut as expected.

The decision may emerge gradually across multiple layers.
""")
    else:
        print(f"""
LATE DECISION EMERGENCE

Layer 8 rank gaps:
  Successes: {layer_8_success_rank_gap:+.0f}
  Failures:  {layer_8_failure_rank_gap:+.0f}

The child/parent preference is not clearly established by layer 8.
The decision crystallizes in later layers.
""")


if __name__ == "__main__":
    # Local testing
    import sys

    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

    # Load data
    print("Loading data...")
    with open("benchmark/matched_pairs_set1_pure.pkl", "rb") as f:
        pairs = pickle.load(f)

    with open("benchmark/factorial_results/factorial_gemma29b.pkl", "rb") as f:
        results = pickle.load(f)

    h1_results = results['set1']['h1_results']

    # Run analysis
    config = LogitLensConfig(n_per_group=10)  # Small test

    results = run_logit_lens_analysis(model, tokenizer, pairs, h1_results, config)

    # Print summary
    print_summary(results)

    # Create visualizations
    create_visualizations(results, "logit_lens_results", config.highlight_layers)
