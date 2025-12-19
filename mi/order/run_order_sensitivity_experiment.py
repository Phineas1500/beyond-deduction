#!/usr/bin/env python3
"""
Axiom Order Sensitivity Experiment

Tests if LLMs rely on sentence position rather than logical structure.

Usage:
    # Generate pairs and run with reversed order
    python run_order_sensitivity_experiment.py \
        --model gemma2-9b \
        --base-url "https://your-endpoint/v1" \
        --n-pairs 200

    # Dry run (print prompts without calling API)
    python run_order_sensitivity_experiment.py --dry-run --n-pairs 5
"""

import os
import sys
import pickle
import argparse
import json
from random import seed
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
from scipy import stats

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from morphology import Morphology, Prop
from generate_matched_pairs import generate_matched_pairs
from evaluate import (
    parse_hypotheses_from_response,
    compute_strong_accuracy,
    compute_weak_accuracy,
)

# Use same seed as INABHYD paper
SEED = 62471893

# System prompt (matches paper exactly)
SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
Only output final hypotheses."""


def reverse_theories(theories_nl: str) -> str:
    """
    Reverse the order of sentences in a theories string.
    
    Example:
        Input:  "Amy is a dalpist. Each dalpist is a rompus. Bob is a dalpist."
        Output: "Bob is a dalpist. Each dalpist is a rompus. Amy is a dalpist."
    """
    # Remove trailing period, split by ". ", reverse, rejoin
    sentences = theories_nl.rstrip('.').split('. ')
    return '. '.join(reversed(sentences)) + '.'


def make_user_prompt(theories: str, observations: str) -> str:
    """Create user prompt matching paper format."""
    return f"Q: {theories} We observe that: {observations} Please come up with hypothesis to explain observations."


def get_openai_client(base_url: Optional[str] = None):
    """Create OpenAI client with optional custom base URL."""
    from openai import OpenAI
    
    if base_url:
        return OpenAI(
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "dummy-key")
        )
    return OpenAI()


def query_model(
    client,
    model_name: str,
    theories: str,
    observations: str,
    temperature: float = 0.0,
) -> str:
    """Query the model and return the response."""
    user_prompt = make_user_prompt(theories, observations)
    
    # Gemma 2 doesn't support system messages - include instructions in user message
    if 'gemma' in model_name.lower():
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        messages = [{"role": "user", "content": full_prompt}]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=256,
    )
    
    return response.choices[0].message.content



def run_experiment(
    pairs: List[Tuple[Dict, Dict]],
    model_name: str,
    base_url: Optional[str] = None,
    order: str = "original",
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Run experiment on pairs with specified order.
    
    Args:
        pairs: List of (h1, h2) matched pairs
        model_name: Model name for API
        base_url: API endpoint
        order: "original" or "reversed"
        dry_run: If True, just print prompts
        verbose: Print progress
        
    Returns:
        Dict with results
    """
    results = {
        'order': order,
        'model': model_name,
        'n_pairs': len(pairs),
        'h1_results': [],
        'h2_results': [],
        'timestamp': datetime.now().isoformat(),
    }
    
    if not dry_run:
        client = get_openai_client(base_url)
    
    for i, (h1, h2) in enumerate(pairs):
        if verbose:
            print(f"  Pair {i+1}/{len(pairs)}...", end='\r')
        
        # Get theories (reverse if needed)
        theories = h1['theories_nl']  # Same for h1 and h2
        if order == "reversed":
            theories = reverse_theories(theories)
        
        observations = h1['observations_nl']  # Same for both
        
        if dry_run:
            if i < 2:  # Only print first 2 examples
                print(f"\n--- Pair {i+1} ({order}) ---")
                print(f"Theories: {theories}")
                print(f"Observations: {observations}")
                print(f"H1 GT: {h1['gt_hypothesis_nl']}")
                print(f"H2 GT: {h2['gt_hypothesis_nl']}")
            continue
        
        # Query for H1
        try:
            h1_reply = query_model(client, model_name, theories, observations)
            h1_strong = compute_strong_accuracy(
                [h1['gt_hypothesis_nl']], 
                [h1_reply]
            )
            results['h1_results'].append({
                'reply': h1_reply,
                'gt': h1['gt_hypothesis_nl'],
                'strong': h1_strong,
                'seed': h1['seed'],
            })
        except Exception as e:
            print(f"\n  Error on H1 pair {i}: {e}")
            results['h1_results'].append({'error': str(e)})
        
        # Query for H2 (same prompt, different ground truth)
        try:
            h2_reply = query_model(client, model_name, theories, observations)
            h2_strong = compute_strong_accuracy(
                [h2['gt_hypothesis_nl']], 
                [h2_reply]
            )
            results['h2_results'].append({
                'reply': h2_reply,
                'gt': h2['gt_hypothesis_nl'],
                'strong': h2_strong,
                'seed': h2['seed'],
            })
        except Exception as e:
            print(f"\n  Error on H2 pair {i}: {e}")
            results['h2_results'].append({'error': str(e)})
    
    if verbose and not dry_run:
        print()  # Newline after progress
    
    # Compute aggregate metrics
    if not dry_run:
        h1_correct = [r['strong'] for r in results['h1_results'] if 'strong' in r]
        h2_correct = [r['strong'] for r in results['h2_results'] if 'strong' in r]
        
        results['h1_accuracy'] = sum(h1_correct) / len(h1_correct) if h1_correct else 0
        results['h2_accuracy'] = sum(h2_correct) / len(h2_correct) if h2_correct else 0
    
    return results


def compute_mcnemar_test(
    original_results: List[int],
    reversed_results: List[int],
) -> Tuple[float, float]:
    """
    Compute McNemar's test for paired binary data.
    
    Returns:
        (chi2_statistic, p_value)
    """
    # Build contingency table
    # b = originally correct, reversed wrong
    # c = originally wrong, reversed correct
    b = sum(1 for o, r in zip(original_results, reversed_results) if o == 1 and r == 0)
    c = sum(1 for o, r in zip(original_results, reversed_results) if o == 0 and r == 1)
    
    if b + c == 0:
        return 0.0, 1.0  # No difference
    
    # McNemar's chi-squared (with continuity correction)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return chi2, p_value


def generate_report(
    original_results: Dict,
    reversed_results: Dict,
    output_path: Optional[str] = None,
) -> str:
    """Generate Markdown report with comparison table."""
    
    model = original_results['model']
    
    # H1 comparison
    h1_orig = original_results.get('h1_accuracy', 0) * 100
    h1_rev = reversed_results.get('h1_accuracy', 0) * 100
    h1_delta = h1_rev - h1_orig
    
    # H2 comparison
    h2_orig = original_results.get('h2_accuracy', 0) * 100
    h2_rev = reversed_results.get('h2_accuracy', 0) * 100
    h2_delta = h2_rev - h1_orig
    
    # McNemar's test for H1
    h1_orig_list = [r.get('strong', 0) for r in original_results.get('h1_results', [])]
    h1_rev_list = [r.get('strong', 0) for r in reversed_results.get('h1_results', [])]
    
    if len(h1_orig_list) == len(h1_rev_list) and len(h1_orig_list) > 0:
        _, h1_pvalue = compute_mcnemar_test(h1_orig_list, h1_rev_list)
    else:
        h1_pvalue = None
    
    # Compute p-value string outside of f-string to avoid format specifier issues
    pvalue_str = f"{h1_pvalue:.4f}" if h1_pvalue is not None else "N/A"
    
    report = f"""# Axiom Order Sensitivity Results

**Model**: {model}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**N pairs**: {original_results.get('n_pairs', 'N/A')}

## Results

| Metric | Original | Reversed | Δ | p-value |
|--------|----------|----------|---|---------|
| H1 Accuracy | {h1_orig:.1f}% | {h1_rev:.1f}% | {h1_delta:+.1f}% | {pvalue_str} |
| H2 Accuracy | {h2_orig:.1f}% | {h2_rev:.1f}% | {h2_delta:+.1f}% | N/A |

## Interpretation

"""

    
    if h1_pvalue is not None and h1_pvalue < 0.05:
        report += f"**Significant difference detected** (p={h1_pvalue:.4f}). "
        if h1_delta < 0:
            report += "Reversing axiom order **decreased** accuracy, suggesting the model relies on positional heuristics.\n"
        else:
            report += "Reversing axiom order **increased** accuracy. Unusual result - investigate further.\n"
    else:
        report += "No significant difference detected. The model appears robust to axiom order.\n"
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Axiom Order Sensitivity Experiment')
    parser.add_argument('--model', type=str, default='gemma2-9b',
                        help='Model name for API')
    parser.add_argument('--base-url', type=str, default=None,
                        help='API base URL (e.g., https://your-endpoint/v1)')
    parser.add_argument('--n-pairs', type=int, default=200,
                        help='Number of matched pairs to test')
    parser.add_argument('--base-seed', type=int, default=SEED,
                        help='Base seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory for output files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print prompts without calling API')
    parser.add_argument('--load-pairs', type=str, default=None,
                        help='Path to existing matched pairs pickle')
    
    args = parser.parse_args()
    
    # Set seeds
    seed(args.base_seed)
    np.random.seed(args.base_seed)
    
    print("=" * 60)
    print("AXIOM ORDER SENSITIVITY EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"N pairs: {args.n_pairs}")
    print(f"Dry run: {args.dry_run}")
    
    # Load or generate pairs
    if args.load_pairs and Path(args.load_pairs).exists():
        print(f"\nLoading pairs from: {args.load_pairs}")
        with open(args.load_pairs, 'rb') as f:
            pairs = pickle.load(f)
        pairs = pairs[:args.n_pairs]
    else:
        print(f"\nGenerating {args.n_pairs} matched pairs...")
        pairs = generate_matched_pairs(
            n_pairs=args.n_pairs,
            base_seed=args.base_seed,
            include_negated=True
        )
    
    print(f"Using {len(pairs)} pairs")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run original order
    print(f"\n--- Running ORIGINAL order ---")
    original_results = run_experiment(
        pairs, args.model, args.base_url,
        order="original", dry_run=args.dry_run
    )
    
    # Run reversed order
    print(f"\n--- Running REVERSED order ---")
    reversed_results = run_experiment(
        pairs, args.model, args.base_url,
        order="reversed", dry_run=args.dry_run
    )
    
    if args.dry_run:
        print("\n[Dry run complete - no API calls made]")
        return
    
    # Save raw results
    model_safe = args.model.replace('/', '_')
    results_path = output_dir / f"order_sensitivity_{model_safe}.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump({
            'original': original_results,
            'reversed': reversed_results,
            'pairs': pairs,
        }, f)
    print(f"\nRaw results saved to: {results_path}")
    
    # Generate report
    report_path = output_dir / f"order_sensitivity_{model_safe}.md"
    report = generate_report(original_results, reversed_results, str(report_path))
    
    print("\n" + report)
    
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
