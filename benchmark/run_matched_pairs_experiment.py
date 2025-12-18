#!/usr/bin/env python3
"""
Run matched minimal pairs experiment for MI analysis.

Runs both H1 and H2 versions of each matched pair through the model,
preserving the pairing for later analysis.
"""

import os
import sys
import pickle
import argparse
import pathlib
import asyncio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")

from evaluate import (
    parse_hypotheses_from_response,
    compute_strong_accuracy,
    compute_weak_accuracy,
    wilson_confidence_interval
)

CONCURRENT_REQUESTS = 20
NO_SYSTEM_PROMPT = False  # Set via --no-system-prompt flag

# NL System prompt (matches INABHYD paper)
NL_SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
        Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
        You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
    .   Only output final hypotheses.
"""

# FOL System prompt
FOL_SYSTEM_PROMPT = """You are a logical reasoning system that performs abduction and induction in first-order logic.
Your job is to produce hypotheses in FOL format that explain observations given theories.
Each hypothesis should take one of these forms:
- predicate(constant) for ground atoms (e.g., dalpist(Amy), rainy(Amy))
- ∀x(P(x) → Q(x)) for universal rules (e.g., ∀x(dalpist(x) → rainy(x)))
Output only FOL hypotheses, one per line."""


def sanitize_model_name(model_name):
    return model_name.replace("-", "").replace("/", "_").replace(":", "_").lower()


def make_user_prompt_nl(example):
    """Create NL user prompt from matched pair example."""
    return f"Q: {example['theories_nl']} We observe that: {example['observations_nl']} Please come up with hypothesis to explain observations."


def make_user_prompt_fol(example):
    """Create FOL user prompt from matched pair example."""
    return f"Theories: {example['theories_fol']} Observations: {example['observations_fol']} Produce hypotheses to explain observations."


def parse_fol_hypothesis(response):
    """Parse FOL hypothesis from model response."""
    # Look for universal quantifier patterns
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        # Skip preamble
        if 'hypothes' in line.lower() or 'here are' in line.lower():
            continue
        # Look for universal quantifier
        if '∀' in line or 'forall' in line.lower():
            # Remove numbering
            import re
            clean = re.sub(r'^\d+\.\s*', '', line)
            return clean
    return ""


def normalize_fol(fol):
    """Normalize FOL string for comparison."""
    norm = fol.replace(" ", "").lower()
    norm = norm.replace("->", "→").replace("−>", "→")
    norm = norm.replace("forall", "∀")
    return norm


def check_fol_match(response, expected):
    """Check if expected FOL hypothesis is in response."""
    response_norm = normalize_fol(response)
    expected_norm = normalize_fol(expected)
    return expected_norm in response_norm


async def process_single_example(client, model_name, example, format_type, semaphore):
    """Process a single matched pair example."""
    async with semaphore:
        if format_type == 'nl':
            user_prompt = make_user_prompt_nl(example)
            system_prompt = NL_SYSTEM_PROMPT
            gt = example['gt_hypothesis_nl']
        else:  # fol
            user_prompt = make_user_prompt_fol(example)
            system_prompt = FOL_SYSTEM_PROMPT
            gt = example['gt_hypothesis_fol']

        # Build messages
        if NO_SYSTEM_PROMPT:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            messages = [{"role": "user", "content": combined_prompt}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

        try:
            completion = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=512
            )
            reply = completion.choices[0].message.content
        except Exception as e:
            print(f"  Error: {e}")
            reply = ""

        # Evaluate based on format
        if format_type == 'nl':
            pred_hyps = parse_hypotheses_from_response(reply)
            gt_hyps = [gt]  # Single hypothesis for matched pairs
            strong_acc = compute_strong_accuracy(pred_hyps, gt_hyps)
            # For weak accuracy, we need observations and theories
            weak_acc = compute_weak_accuracy(pred_hyps, gt_hyps,
                                             example['observations_nl'],
                                             example['theories_nl'])
        else:  # fol
            strong_acc = 1 if check_fol_match(reply, gt) else 0
            # For FOL, weak = strong for simplicity
            weak_acc = strong_acc

        return {
            'reply': reply,
            'gt': gt,
            'strong': strong_acc,
            'weak': weak_acc,
            'depth': example['depth'],
            'seed': example['seed'],
            'is_negated': example['is_negated']
        }


async def run_matched_pairs_experiment(model_name, format_type, base_url, pairs_file):
    """Run matched pairs experiment."""
    from openai import AsyncOpenAI

    global NO_SYSTEM_PROMPT

    client = AsyncOpenAI(
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=DEFAULT_API_KEY,
    )

    # Load matched pairs
    with open(pairs_file, 'rb') as f:
        pairs = pickle.load(f)

    print(f"\n{'='*60}")
    print(f"Matched Pairs Experiment: {model_name}")
    print(f"Format: {format_type.upper()}")
    print(f"Pairs: {len(pairs)}")
    print(f"No system prompt: {NO_SYSTEM_PROMPT}")
    print(f"{'='*60}")

    # Flatten pairs into list of examples with pair index
    all_examples = []
    for pair_idx, (h1, h2) in enumerate(pairs):
        h1['pair_idx'] = pair_idx
        h2['pair_idx'] = pair_idx
        all_examples.append(h1)
        all_examples.append(h2)

    print(f"  Processing {len(all_examples)} examples ({len(pairs)} pairs)...")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = [
        process_single_example(client, model_name, ex, format_type, semaphore)
        for ex in all_examples
    ]

    responses = await asyncio.gather(*tasks)

    # Reconstruct paired results
    h1_results = []
    h2_results = []
    paired_results = []

    for i in range(0, len(responses), 2):
        h1_resp = responses[i]
        h2_resp = responses[i + 1]

        h1_results.append(h1_resp)
        h2_results.append(h2_resp)

        paired_results.append({
            'pair_idx': i // 2,
            'seed': h1_resp['seed'],
            'is_negated': h1_resp['is_negated'],
            'h1_strong': h1_resp['strong'],
            'h1_weak': h1_resp['weak'],
            'h1_reply': h1_resp['reply'],
            'h1_gt': h1_resp['gt'],
            'h2_strong': h2_resp['strong'],
            'h2_weak': h2_resp['weak'],
            'h2_reply': h2_resp['reply'],
            'h2_gt': h2_resp['gt'],
        })

    # Compute aggregate metrics
    h1_strong = np.mean([r['strong'] for r in h1_results])
    h1_weak = np.mean([r['weak'] for r in h1_results])
    h2_strong = np.mean([r['strong'] for r in h2_results])
    h2_weak = np.mean([r['weak'] for r in h2_results])

    n = len(pairs)
    h1_strong_ci = wilson_confidence_interval(h1_strong, n)
    h2_strong_ci = wilson_confidence_interval(h2_strong, n)

    print(f"\n  Results:")
    print(f"    H1 Strong: {h1_strong:.3f} [{h1_strong_ci[0]:.3f}, {h1_strong_ci[1]:.3f}]")
    print(f"    H1 Weak:   {h1_weak:.3f}")
    print(f"    H2 Strong: {h2_strong:.3f} [{h2_strong_ci[0]:.3f}, {h2_strong_ci[1]:.3f}]")
    print(f"    H2 Weak:   {h2_weak:.3f}")

    # Analyze paired outcomes
    h1_success_h2_success = sum(1 for p in paired_results if p['h1_strong'] and p['h2_strong'])
    h1_success_h2_fail = sum(1 for p in paired_results if p['h1_strong'] and not p['h2_strong'])
    h1_fail_h2_success = sum(1 for p in paired_results if not p['h1_strong'] and p['h2_strong'])
    h1_fail_h2_fail = sum(1 for p in paired_results if not p['h1_strong'] and not p['h2_strong'])

    print(f"\n  Paired Outcomes:")
    print(f"    H1 success, H2 success: {h1_success_h2_success}")
    print(f"    H1 success, H2 fail:    {h1_success_h2_fail}  ← Key for MI (depth-sensitive)")
    print(f"    H1 fail, H2 success:    {h1_fail_h2_success}")
    print(f"    H1 fail, H2 fail:       {h1_fail_h2_fail}")

    # Negation breakdown
    negated_pairs = [p for p in paired_results if p['is_negated']]
    positive_pairs = [p for p in paired_results if not p['is_negated']]

    if negated_pairs:
        neg_h1 = np.mean([p['h1_strong'] for p in negated_pairs])
        neg_h2 = np.mean([p['h2_strong'] for p in negated_pairs])
        print(f"\n  Negated cases ({len(negated_pairs)} pairs):")
        print(f"    H1 Strong: {neg_h1:.3f}")
        print(f"    H2 Strong: {neg_h2:.3f}")

    if positive_pairs:
        pos_h1 = np.mean([p['h1_strong'] for p in positive_pairs])
        pos_h2 = np.mean([p['h2_strong'] for p in positive_pairs])
        print(f"\n  Positive cases ({len(positive_pairs)} pairs):")
        print(f"    H1 Strong: {pos_h1:.3f}")
        print(f"    H2 Strong: {pos_h2:.3f}")

    # Save results
    output_dir = pathlib.Path("matched_pairs_results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"matched_pairs_{format_type}_{sanitize_model_name(model_name)}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({
            'pairs': pairs,
            'h1_results': h1_results,
            'h2_results': h2_results,
            'paired_results': paired_results,
            'metrics': {
                'h1_strong': h1_strong,
                'h1_weak': h1_weak,
                'h2_strong': h2_strong,
                'h2_weak': h2_weak,
                'h1_strong_ci': h1_strong_ci,
                'h2_strong_ci': h2_strong_ci,
            },
            'model': model_name,
            'format': format_type,
        }, f)

    print(f"\n  Saved to {output_file}")

    return {
        'h1_strong': h1_strong,
        'h2_strong': h2_strong,
        'paired_results': paired_results
    }


def main():
    global NO_SYSTEM_PROMPT, CONCURRENT_REQUESTS

    parser = argparse.ArgumentParser(description='Run matched pairs experiment for MI analysis')
    parser.add_argument('--model', '-m', type=str, required=True, help='Model name')
    parser.add_argument('--base-url', '-b', type=str, required=True, help='API base URL')
    parser.add_argument('--format', '-f', type=str, default='nl', choices=['nl', 'fol'],
                        help='Format: nl (natural language) or fol (first-order logic)')
    parser.add_argument('--pairs-file', '-p', type=str, default='matched_pairs.pkl',
                        help='Path to matched pairs pickle file')
    parser.add_argument('--no-system-prompt', action='store_true',
                        help='Merge system prompt into user message (for Gemma 2)')
    parser.add_argument('--concurrent', '-c', type=int, default=20,
                        help='Number of concurrent requests')
    args = parser.parse_args()

    NO_SYSTEM_PROMPT = args.no_system_prompt
    CONCURRENT_REQUESTS = args.concurrent

    asyncio.run(run_matched_pairs_experiment(
        model_name=args.model,
        format_type=args.format,
        base_url=args.base_url,
        pairs_file=args.pairs_file
    ))


if __name__ == "__main__":
    main()
