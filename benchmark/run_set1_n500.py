#!/usr/bin/env python3
"""Run Set 1 experiment on the n=500 dataset."""

import os
import sys
import pickle
import asyncio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import AsyncOpenAI
from evaluate import parse_hypotheses_from_response, compute_strong_accuracy

CONCURRENT_REQUESTS = 20
MODEL_NAME = "gemma2-9b"
BASE_URL = "https://phineas1500--gemma2-9b-inference-serve.modal.run/v1"

NL_SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
        Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
        You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
    .   Only output final hypotheses.
"""


async def process_matched_pair(client, h1, h2, semaphore, pair_idx):
    """Process a matched pair."""
    async with semaphore:
        results = []
        for example in [h1, h2]:
            user_prompt = f"Q: {example['theories_nl']} We observe that: {example['observations_nl']} Please come up with hypothesis to explain observations."

            # Gemma doesn't support system prompts well, combine into user message
            combined = f"{NL_SYSTEM_PROMPT}\n\n{user_prompt}"
            messages = [{"role": "user", "content": combined}]

            try:
                completion = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0,
                    max_tokens=512
                )
                reply = completion.choices[0].message.content
            except Exception as e:
                print(f"  Error on pair {pair_idx}: {e}")
                reply = ""

            pred_hyps = parse_hypotheses_from_response(reply)
            gt_hyps = [example['gt_hypothesis_nl']]
            strong_acc = compute_strong_accuracy(pred_hyps, gt_hyps)

            results.append({
                'reply': reply,
                'gt': example['gt_hypothesis_nl'],
                'strong': strong_acc,
                'depth': example['depth'],
                'seed': example['seed'],
            })

        return results[0], results[1]


async def main():
    # Load the dataset - check for command line arg or default
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if not os.path.isabs(input_file):
            input_file = os.path.join(script_dir, input_file)
    else:
        input_file = os.path.join(script_dir, 'matched_pairs_set1_n500.pkl')

    with open(input_file, 'rb') as f:
        pairs = pickle.load(f)

    print(f"Loaded {len(pairs)} pairs from {os.path.basename(input_file)}")
    print(f"Running inference on Gemma 2 9B...")
    print(f"Concurrent requests: {CONCURRENT_REQUESTS}")
    print()

    client = AsyncOpenAI(base_url=BASE_URL, api_key="not-needed")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    tasks = [
        process_matched_pair(client, h1, h2, semaphore, i)
        for i, (h1, h2) in enumerate(pairs)
    ]

    # Process with progress
    results = []
    batch_size = 50
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
        print(f"  Processed {min(i+batch_size, len(tasks))}/{len(tasks)} pairs...")

    h1_results = [r[0] for r in results]
    h2_results = [r[1] for r in results]

    # Compute statistics
    h1_strong = np.mean([r['strong'] for r in h1_results])
    h2_strong = np.mean([r['strong'] for r in h2_results])

    print(f"\n{'='*60}")
    print(f"RESULTS (n={len(pairs)})")
    print(f"{'='*60}")
    print(f"H1 Strong Accuracy: {h1_strong:.1%}")
    print(f"H2 Strong Accuracy: {h2_strong:.1%}")

    # Stratify by behavior pattern
    h1_fail_h2_success = sum(1 for h1, h2 in zip(h1_results, h2_results)
                             if h1['strong'] == 0 and h2['strong'] == 1)
    h1_success_h2_fail = sum(1 for h1, h2 in zip(h1_results, h2_results)
                             if h1['strong'] == 1 and h2['strong'] == 0)
    h1_success_h2_success = sum(1 for h1, h2 in zip(h1_results, h2_results)
                                if h1['strong'] == 1 and h2['strong'] == 1)
    h1_fail_h2_fail = sum(1 for h1, h2 in zip(h1_results, h2_results)
                          if h1['strong'] == 0 and h2['strong'] == 0)

    print(f"\nBehavioral Stratification:")
    print(f"  h1_fail_h2_success (GOLD for MI): {h1_fail_h2_success} ({h1_fail_h2_success/len(pairs):.1%})")
    print(f"  h1_success_h2_fail (standard):    {h1_success_h2_fail} ({h1_success_h2_fail/len(pairs):.1%})")
    print(f"  h1_success_h2_success:            {h1_success_h2_success} ({h1_success_h2_success/len(pairs):.1%})")
    print(f"  h1_fail_h2_fail:                  {h1_fail_h2_fail} ({h1_fail_h2_fail/len(pairs):.1%})")

    # Find gold pair indices
    gold_indices = [i for i, (h1, h2) in enumerate(zip(h1_results, h2_results))
                    if h1['strong'] == 0 and h2['strong'] == 1]
    print(f"\nGold pair indices (parent output): {gold_indices[:20]}{'...' if len(gold_indices) > 20 else ''}")
    print(f"Total gold pairs: {len(gold_indices)}")

    # Save results
    output = {
        'h1_strong': h1_strong,
        'h2_strong': h2_strong,
        'h1_results': h1_results,
        'h2_results': h2_results,
        'pairs': pairs,
        'gold_indices': gold_indices,
        'stratification': {
            'h1_fail_h2_success': h1_fail_h2_success,
            'h1_success_h2_fail': h1_success_h2_fail,
            'h1_success_h2_success': h1_success_h2_success,
            'h1_fail_h2_fail': h1_fail_h2_fail,
        }
    }

    # Generate output filename based on input
    input_basename = os.path.basename(input_file).replace('matched_pairs_set1_', '').replace('.pkl', '')
    output_path = os.path.join(script_dir, f'factorial_results/factorial_gemma29b_{input_basename}.pkl')
    os.makedirs(os.path.join(script_dir, 'factorial_results'), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
