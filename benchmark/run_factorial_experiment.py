#!/usr/bin/env python3
"""
Run factorial experiment across all matched pair sets.

Set 1 (Pure): Baseline 2-hop with single child→parent path
Set 2 (Salience): Same structure + direct parent members (tests frequency heuristic)
Set 4 (INABHYD-style): Multiple children → parent + direct member (ceiling)

This tests whether salience/frequency alone explains INABHYD's multi-hop "reasoning".

Examples:
    # Test local model (existing behavior):
    python run_factorial_experiment.py --model gemma2-9b --base-url "http://localhost:8000/v1"

    # Test GPT-4o via OpenRouter:
    python run_factorial_experiment.py --provider openrouter --model gpt-4o --concurrent 10

    # Test Claude Opus 4.5 via OpenRouter:
    python run_factorial_experiment.py --provider openrouter --model claude-opus-4.5 --concurrent 10

    # Test with full model ID:
    python run_factorial_experiment.py --provider openrouter --model anthropic/claude-opus-4.5 --concurrent 10
"""

import os
import sys
import re
import pickle
import argparse
import pathlib
import asyncio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")

from evaluate import parse_hypotheses_from_response, compute_strong_accuracy, wilson_confidence_interval

CONCURRENT_REQUESTS = 20
NO_SYSTEM_PROMPT = False
IS_REASONING_MODEL = False  # Set to True for models that use <think> tags

# OpenRouter model aliases
OPENROUTER_MODELS = {
    'gpt-4o': 'openai/chatgpt-4o-latest',
    'llama-3-70b': 'meta-llama/llama-3-70b-instruct',
    'deepseek-chat': 'deepseek/deepseek-chat',
    'deepseek-r1-70b': 'deepseek/deepseek-r1-distill-llama-70b',
    'gpt-5': 'openai/gpt-5.2',
    'gemini-3-pro': 'google/gemini-3-pro-preview',
    'claude-opus-4.5': 'anthropic/claude-opus-4.5',
}

# Rough cost per 1M tokens (input + output) for cost estimation
OPENROUTER_COSTS = {
    'gpt-4o': 5.0,
    'claude-opus-4.5': 15.0,
    'gpt-5': 20.0,
    'gemini-3-pro': 3.0,
    'llama-3-70b': 0.8,
    'deepseek-chat': 0.5,
    'deepseek-r1-70b': 1.0,
}

# Models that use <think> tags for reasoning
REASONING_MODELS = ['r1', 'reasoning', 'deepseek-r1']


def resolve_model_name(model: str, provider: str) -> str:
    """Resolve CLI model alias to full model ID."""
    if provider == 'openrouter' and model in OPENROUTER_MODELS:
        return OPENROUTER_MODELS[model]
    return model  # Return as-is for custom model IDs or other providers


def is_reasoning_model(model_name: str) -> bool:
    """Check if model uses <think> tags for reasoning."""
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in REASONING_MODELS)


def strip_thinking(response: str) -> str:
    """Remove <think>...</think> blocks from reasoning model outputs."""
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()


def detect_refusal(reply):
    """
    Detect if the response contains refusal or uncertainty language.

    Returns:
        bool: True if refusal/uncertainty detected
    """
    refusal_patterns = [
        "i cannot determine",
        "cannot determine",
        "i can't determine",
        "can't determine",
        "the evidence conflicts",
        "conflicting evidence",
        "contradictory information",
        "contradictory evidence",
        "it's unclear",
        "it is unclear",
        "unclear whether",
        "i'm not sure",
        "i am not sure",
        "not enough information",
        "insufficient information",
        "this is ambiguous",
        "ambiguous evidence",
        "cannot conclude",
        "can't conclude",
        "no definitive",
        "impossible to determine",
        "both possibilities",
        "either could be",
    ]

    reply_lower = reply.lower()
    return any(pattern in reply_lower for pattern in refusal_patterns)


def analyze_conflict_response(reply, example):
    """
    Perform granular analysis of a Set 6 conflict response.

    Returns:
        dict with:
            - contains_correct: bool - Parent concept + correct property anywhere
            - contains_shortcut: bool - Parent concept + shortcut property anywhere
            - contains_child_conclusion: bool - Child concept + any property
            - is_refusal: bool - Detected refusal/uncertainty language
            - classification: str - 'correct_only', 'shortcut_only', 'hedges', 'refusal', 'neither'
    """
    reply_lower = reply.lower()

    parent = example['parent_concept_lower']
    child = example['child_concept'].lower()
    correct_prop = example['correct_property_name']
    shortcut_prop = example['shortcut_property_name']

    # Check for presence of different conclusions
    contains_correct = parent in reply_lower and correct_prop in reply_lower
    contains_shortcut = parent in reply_lower and shortcut_prop in reply_lower

    # Check if child concept is mentioned with either property
    contains_child_conclusion = child in reply_lower and (
        correct_prop in reply_lower or shortcut_prop in reply_lower
    )

    # Check for refusal
    is_refusal = detect_refusal(reply)

    # Determine classification
    if is_refusal and not contains_correct and not contains_shortcut:
        classification = 'refusal'
    elif contains_correct and contains_shortcut:
        classification = 'hedges'
    elif contains_correct and not contains_shortcut:
        classification = 'correct_only'
    elif contains_shortcut and not contains_correct:
        classification = 'shortcut_only'
    else:
        classification = 'neither'

    return {
        'contains_correct': contains_correct,
        'contains_shortcut': contains_shortcut,
        'contains_child_conclusion': contains_child_conclusion,
        'is_refusal': is_refusal,
        'classification': classification,
    }


def classify_conflict_response(reply, example):
    """
    Classify a response to a Set 6 conflict example (legacy function for backwards compatibility).

    Returns:
        'correct': Model output matches ground truth (used 2-hop reasoning)
        'shortcut': Model output matches shortcut answer (used conjunction detection)
        'other': Neither (confused/refused/wrong)
    """
    analysis = analyze_conflict_response(reply, example)

    # Map new classifications to legacy format
    if analysis['classification'] == 'correct_only':
        return 'correct'
    elif analysis['classification'] == 'shortcut_only':
        return 'shortcut'
    else:
        return 'other'

NL_SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
        Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
        You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
    .   Only output final hypotheses.
"""


def sanitize_model_name(model_name):
    return model_name.replace("-", "").replace("/", "_").replace(":", "_").lower()


async def process_conflict_example(client, model_name, example, semaphore):
    """Process a single Set 6 conflict example with granular analysis."""
    async with semaphore:
        user_prompt = f"Q: {example['theories_nl']} We observe that: {example['observations_nl']} Please come up with hypothesis to explain observations."

        if NO_SYSTEM_PROMPT:
            combined = f"{NL_SYSTEM_PROMPT}\n\n{user_prompt}"
            messages = [{"role": "user", "content": combined}]
        else:
            messages = [
                {"role": "system", "content": NL_SYSTEM_PROMPT},
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

        # Store original reply (with thinking) and cleaned reply
        original_reply = reply
        if IS_REASONING_MODEL and reply:
            reply = strip_thinking(reply)

        # Granular analysis (on cleaned reply)
        analysis = analyze_conflict_response(reply, example)

        # Legacy classification for backwards compatibility
        legacy_classification = classify_conflict_response(reply, example)

        return {
            'reply': reply,
            'ground_truth': example['ground_truth_nl'],
            'shortcut_answer': example['shortcut_answer_nl'],
            'seed': example['seed'],
            'correct_is_negated': example['correct_is_negated'],
            # Granular analysis fields
            'contains_correct': analysis['contains_correct'],
            'contains_shortcut': analysis['contains_shortcut'],
            'contains_child_conclusion': analysis['contains_child_conclusion'],
            'is_refusal': analysis['is_refusal'],
            'classification': analysis['classification'],
            # Legacy field
            'legacy_classification': legacy_classification,
        }


async def process_example(client, model_name, example, semaphore):
    """Process a single example."""
    async with semaphore:
        # Build user prompt
        user_prompt = f"Q: {example['theories_nl']} We observe that: {example['observations_nl']} Please come up with hypothesis to explain observations."

        if NO_SYSTEM_PROMPT:
            combined = f"{NL_SYSTEM_PROMPT}\n\n{user_prompt}"
            messages = [{"role": "user", "content": combined}]
        else:
            messages = [
                {"role": "system", "content": NL_SYSTEM_PROMPT},
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

        # Strip thinking tags for reasoning models
        if IS_REASONING_MODEL and reply:
            reply = strip_thinking(reply)

        # Evaluate
        pred_hyps = parse_hypotheses_from_response(reply)
        gt_hyps = [example['gt_hypothesis_nl']]
        strong_acc = compute_strong_accuracy(pred_hyps, gt_hyps)

        return {
            'reply': reply,
            'gt': example['gt_hypothesis_nl'],
            'strong': strong_acc,
            'seed': example['seed'],
            'is_negated': example['is_negated'],
            'parent_mentions': example.get('parent_mentions', 1),
        }


async def process_matched_pair(client, model_name, h1, h2, semaphore):
    """Process a matched pair (Set 1 only)."""
    async with semaphore:
        results = []
        for example in [h1, h2]:
            user_prompt = f"Q: {example['theories_nl']} We observe that: {example['observations_nl']} Please come up with hypothesis to explain observations."

            if NO_SYSTEM_PROMPT:
                combined = f"{NL_SYSTEM_PROMPT}\n\n{user_prompt}"
                messages = [{"role": "user", "content": combined}]
            else:
                messages = [
                    {"role": "system", "content": NL_SYSTEM_PROMPT},
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

            # Strip thinking tags for reasoning models
            if IS_REASONING_MODEL and reply:
                reply = strip_thinking(reply)

            pred_hyps = parse_hypotheses_from_response(reply)
            gt_hyps = [example['gt_hypothesis_nl']]
            strong_acc = compute_strong_accuracy(pred_hyps, gt_hyps)

            results.append({
                'reply': reply,
                'gt': example['gt_hypothesis_nl'],
                'strong': strong_acc,
                'depth': example['depth'],
            })

        return results[0], results[1]  # h1_result, h2_result


async def run_set1_experiment(client, model_name, pairs_file):
    """Run Set 1 (Pure) experiment with H1/H2 pairs."""
    with open(pairs_file, 'rb') as f:
        pairs = pickle.load(f)

    print(f"\n--- Set 1 (Pure Multi-hop): {len(pairs)} pairs ---")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = [process_matched_pair(client, model_name, h1, h2, semaphore) for h1, h2 in pairs]
    results = await asyncio.gather(*tasks)

    h1_results = [r[0] for r in results]
    h2_results = [r[1] for r in results]

    h1_strong = np.mean([r['strong'] for r in h1_results])
    h2_strong = np.mean([r['strong'] for r in h2_results])

    print(f"  H1 Strong: {h1_strong:.1%}")
    print(f"  H2 Strong: {h2_strong:.1%}")

    return {
        'h1_strong': h1_strong,
        'h2_strong': h2_strong,
        'h1_results': h1_results,
        'h2_results': h2_results,
        'pairs': pairs,
    }


async def run_single_set_experiment(client, model_name, examples_file, set_name):
    """Run experiment on a single set (Set 2 or Set 4)."""
    with open(examples_file, 'rb') as f:
        examples = pickle.load(f)

    print(f"\n--- {set_name}: {len(examples)} examples ---")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = [process_example(client, model_name, ex, semaphore) for ex in examples]
    results = await asyncio.gather(*tasks)

    strong = np.mean([r['strong'] for r in results])
    avg_mentions = np.mean([r['parent_mentions'] for r in results])

    print(f"  Strong: {strong:.1%}")
    print(f"  Avg parent mentions: {avg_mentions:.1f}")

    return {
        'strong': strong,
        'results': results,
        'examples': examples,
        'avg_parent_mentions': avg_mentions,
    }


async def run_conflict_experiment(client, model_name, examples_file):
    """Run Set 6 (Conflict Test) experiment with granular analysis."""
    with open(examples_file, 'rb') as f:
        examples = pickle.load(f)

    print(f"\n--- Set 6 (Conflict Test): {len(examples)} examples ---")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = [process_conflict_example(client, model_name, ex, semaphore) for ex in examples]
    results = await asyncio.gather(*tasks)

    n = len(results)

    # Presence counts (can overlap)
    contains_correct_count = sum(1 for r in results if r['contains_correct'])
    contains_shortcut_count = sum(1 for r in results if r['contains_shortcut'])
    contains_child_count = sum(1 for r in results if r['contains_child_conclusion'])
    refusal_detected_count = sum(1 for r in results if r['is_refusal'])

    # Classification counts (mutually exclusive)
    correct_only_count = sum(1 for r in results if r['classification'] == 'correct_only')
    shortcut_only_count = sum(1 for r in results if r['classification'] == 'shortcut_only')
    hedges_count = sum(1 for r in results if r['classification'] == 'hedges')
    refusal_count = sum(1 for r in results if r['classification'] == 'refusal')
    neither_count = sum(1 for r in results if r['classification'] == 'neither')

    # Compute rates
    contains_correct_rate = contains_correct_count / n
    contains_shortcut_rate = contains_shortcut_count / n
    contains_child_rate = contains_child_count / n
    correct_only_rate = correct_only_count / n
    shortcut_only_rate = shortcut_only_count / n
    hedge_rate = hedges_count / n
    refusal_rate = refusal_count / n
    neither_rate = neither_count / n

    # Print summary
    print(f"  Contains correct answer: {contains_correct_rate:.1%}")
    print(f"  Contains shortcut answer: {contains_shortcut_rate:.1%}")
    print(f"  Contains child conclusion: {contains_child_rate:.1%}")
    print(f"  Classification breakdown:")
    print(f"    Correct only (genuine 2-hop?): {correct_only_rate:.1%} ({correct_only_count}/{n})")
    print(f"    Shortcut only (pure shortcut): {shortcut_only_rate:.1%} ({shortcut_only_count}/{n})")
    print(f"    Hedges (outputs both): {hedge_rate:.1%} ({hedges_count}/{n})")
    print(f"    Refusal (expresses uncertainty): {refusal_rate:.1%} ({refusal_count}/{n})")
    print(f"    Neither: {neither_rate:.1%} ({neither_count}/{n})")

    # Legacy rates for backwards compatibility
    legacy_correct = sum(1 for r in results if r['legacy_classification'] == 'correct') / n
    legacy_shortcut = sum(1 for r in results if r['legacy_classification'] == 'shortcut') / n
    legacy_other = sum(1 for r in results if r['legacy_classification'] == 'other') / n

    # Build metrics dict
    metrics = {
        'contains_correct_rate': contains_correct_rate,
        'contains_shortcut_rate': contains_shortcut_rate,
        'contains_child_rate': contains_child_rate,
        'correct_only_rate': correct_only_rate,
        'shortcut_only_rate': shortcut_only_rate,
        'hedge_rate': hedge_rate,
        'refusal_rate': refusal_rate,
        'neither_rate': neither_rate,
    }

    return {
        # New granular metrics
        'metrics': metrics,
        'contains_correct_rate': contains_correct_rate,
        'contains_shortcut_rate': contains_shortcut_rate,
        'contains_child_rate': contains_child_rate,
        'correct_only_rate': correct_only_rate,
        'shortcut_only_rate': shortcut_only_rate,
        'hedge_rate': hedge_rate,
        'refusal_rate': refusal_rate,
        'neither_rate': neither_rate,
        # Legacy rates (for backwards compatibility)
        'correct_rate': legacy_correct,
        'shortcut_rate': legacy_shortcut,
        'other_rate': legacy_other,
        # Counts
        'correct_only_count': correct_only_count,
        'shortcut_only_count': shortcut_only_count,
        'hedges_count': hedges_count,
        'refusal_count': refusal_count,
        'neither_count': neither_count,
        # Raw data
        'results': results,
        'examples': examples,
    }


async def run_factorial_experiment(client, model_name, display_name=None):
    """Run full factorial experiment across all sets.

    Args:
        client: AsyncOpenAI client instance
        model_name: Full model ID to use for API calls
        display_name: Human-readable name for output (defaults to model_name)
    """
    if display_name is None:
        display_name = model_name

    print("=" * 70)
    print(f"FACTORIAL EXPERIMENT: {display_name}")
    if display_name != model_name:
        print(f"  (Model ID: {model_name})")
    print(f"Testing salience hypothesis for multi-hop reasoning")
    print("=" * 70)

    results = {}

    # Set 1: Pure multi-hop
    if os.path.exists('matched_pairs_set1_pure.pkl'):
        results['set1'] = await run_set1_experiment(client, model_name, 'matched_pairs_set1_pure.pkl')
    else:
        print("  Warning: matched_pairs_set1_pure.pkl not found")

    # Set 2: Salience test
    if os.path.exists('matched_pairs_set2_salience.pkl'):
        results['set2'] = await run_single_set_experiment(
            client, model_name, 'matched_pairs_set2_salience.pkl', 'Set 2 (Salience Test)')
    else:
        print("  Warning: matched_pairs_set2_salience.pkl not found")

    # Set 4: INABHYD-style
    if os.path.exists('matched_pairs_set4_inabhyd.pkl'):
        results['set4'] = await run_single_set_experiment(
            client, model_name, 'matched_pairs_set4_inabhyd.pkl', 'Set 4 (INABHYD-style)')
    else:
        print("  Warning: matched_pairs_set4_inabhyd.pkl not found")

    # Set 5: Evidential Path
    if os.path.exists('matched_pairs_set5_evidential.pkl'):
        results['set5'] = await run_single_set_experiment(
            client, model_name, 'matched_pairs_set5_evidential.pkl', 'Set 5 (Evidential Path)')
    else:
        print("  Warning: matched_pairs_set5_evidential.pkl not found")

    # Set 6: Conflict Test
    if os.path.exists('matched_pairs_set6_conflict.pkl'):
        results['set6'] = await run_conflict_experiment(
            client, model_name, 'matched_pairs_set6_conflict.pkl')
    else:
        print("  Warning: matched_pairs_set6_conflict.pkl not found")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Set':<25} {'Parent Mentions':<18} {'Strong Accuracy':<15}")
    print("-" * 60)

    if 'set1' in results:
        print(f"{'Set 1 (Pure) H1':<25} {'1x':<18} {results['set1']['h1_strong']:.1%}")
        print(f"{'Set 1 (Pure) H2':<25} {'1x':<18} {results['set1']['h2_strong']:.1%}")

    if 'set2' in results:
        mentions = results['set2']['avg_parent_mentions']
        print(f"{'Set 2 (Salience)':<25} {f'{mentions:.0f}x':<18} {results['set2']['strong']:.1%}")

    if 'set4' in results:
        mentions = results['set4']['avg_parent_mentions']
        print(f"{'Set 4 (INABHYD-style)':<25} {f'{mentions:.0f}x':<18} {results['set4']['strong']:.1%}")

    if 'set5' in results:
        mentions = results['set5']['avg_parent_mentions']
        print(f"{'Set 5 (Evidential Path)':<25} {f'{mentions:.0f}x':<18} {results['set5']['strong']:.1%}")

    # Set 6 has granular analysis
    if 'set6' in results:
        s6 = results['set6']
        print(f"\n{'Set 6 (Conflict Test)':<60}")
        print("-" * 60)
        print(f"  Response contains correct answer: {s6['contains_correct_rate']:.1%}")
        print(f"  Response contains shortcut answer: {s6['contains_shortcut_rate']:.1%}")
        print(f"  Response contains child conclusion: {s6['contains_child_rate']:.1%}")
        print(f"  Classification:")
        print(f"    Correct only (genuine 2-hop?): {s6['correct_only_rate']:.1%}")
        print(f"    Shortcut only (pure shortcut): {s6['shortcut_only_rate']:.1%}")
        print(f"    Hedges (outputs both): {s6['hedge_rate']:.1%}")
        print(f"    Refusal (uncertainty): {s6['refusal_rate']:.1%}")
        print(f"    Neither: {s6['neither_rate']:.1%}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if 'set1' in results and 'set2' in results:
        set1_h2 = results['set1']['h2_strong']
        set2 = results['set2']['strong']

        if set2 > set1_h2 + 0.1:
            print(f"\nSet 2 ({set2:.0%}) >> Set 1 H2 ({set1_h2:.0%})")
            print("→ SALIENCE ALONE IMPROVES PERFORMANCE")
            print("→ The model uses frequency/salience heuristics, not compositional reasoning")
        elif abs(set2 - set1_h2) <= 0.1:
            print(f"\nSet 2 ({set2:.0%}) ≈ Set 1 H2 ({set1_h2:.0%})")
            print("→ Salience alone doesn't help")
            print("→ Need to investigate other factors")
        else:
            print(f"\nSet 2 ({set2:.0%}) < Set 1 H2 ({set1_h2:.0%})")
            print("→ Unexpected: distractors hurt performance")

    if 'set2' in results and 'set4' in results:
        set2 = results['set2']['strong']
        set4 = results['set4']['strong']

        if abs(set2 - set4) <= 0.1:
            print(f"\nSet 2 ({set2:.0%}) ≈ Set 4 ({set4:.0%})")
            print("→ Salience alone explains INABHYD performance")
            print("→ Multiple paths don't add anything beyond frequency boost")
        elif set4 > set2 + 0.1:
            print(f"\nSet 4 ({set4:.0%}) > Set 2 ({set2:.0%})")
            print("→ Structure (multiple paths) adds something beyond salience")

    # Set 5 analysis: Test the evidential path hypothesis
    if 'set5' in results:
        set5 = results['set5']['strong']

        if 'set2' in results:
            set2 = results['set2']['strong']
            if set5 > set2 + 0.1:
                print(f"\nSet 5 ({set5:.0%}) >> Set 2 ({set2:.0%})")
                print("→ EVIDENTIAL PATH HYPOTHESIS CONFIRMED")
                print("→ Direct parent members WITH observed properties enable 1-hop shortcut")
                print("→ Salience without evidence is useless")
            elif abs(set5 - set2) <= 0.1:
                print(f"\nSet 5 ({set5:.0%}) ≈ Set 2 ({set2:.0%})")
                print("→ Unexpected: evidential path doesn't help")

        if 'set4' in results:
            set4 = results['set4']['strong']
            if abs(set5 - set4) <= 0.1:
                print(f"\nSet 5 ({set5:.0%}) ≈ Set 4 ({set4:.0%})")
                print("→ STRUCTURE DOESN'T MATTER - only evidential path matters")
                print("→ Model uses: (X is concept) ∧ (X has property) → concept has property")
                print("→ This is CONJUNCTION DETECTION, not compositional reasoning")
            elif set4 > set5 + 0.1:
                print(f"\nSet 4 ({set4:.0%}) > Set 5 ({set5:.0%})")
                print("→ Multiple convergent paths add something beyond single evidential path")

    # Set 6 analysis: The definitive shortcut test
    if 'set6' in results:
        s6 = results['set6']
        hedge_rate = s6['hedge_rate']
        correct_only = s6['correct_only_rate']
        shortcut_only = s6['shortcut_only_rate']
        refusal_rate = s6['refusal_rate']
        child_rate = s6['contains_child_rate']
        contains_correct = s6['contains_correct_rate']
        contains_shortcut = s6['contains_shortcut_rate']

        print(f"\n--- SET 6 CONFLICT TEST ANALYSIS ---")

        # Primary finding: hedging behavior
        if hedge_rate > 0.5:
            print(f"\nHigh hedge rate ({hedge_rate:.0%}): Model outputs BOTH correct and shortcut answers")
            print("→ PARALLEL 1-HOP INFERENCE: Model does independent 1-hop from each evidence")
            print("→ Model cannot resolve conflict through compositional reasoning")
            print("→ This is NOT genuine multi-hop - it's hedging via multiple 1-hop inferences")
        elif hedge_rate > 0.3:
            print(f"\nModerate hedge rate ({hedge_rate:.0%}): Model often outputs both answers")
            print("→ Model struggles to resolve conflicting evidence")

        # Shortcut vs correct comparison (when not hedging)
        if shortcut_only > correct_only + 0.1:
            print(f"\nShortcut only ({shortcut_only:.0%}) >> Correct only ({correct_only:.0%})")
            print("→ When forced to choose, model prefers 1-hop shortcut over 2-hop reasoning")
        elif correct_only > shortcut_only + 0.1:
            print(f"\nCorrect only ({correct_only:.0%}) >> Shortcut only ({shortcut_only:.0%})")
            print("→ SURPRISING: Model sometimes uses genuine 2-hop reasoning")
        elif abs(correct_only - shortcut_only) <= 0.1 and correct_only > 0.05:
            print(f"\nCorrect only ({correct_only:.0%}) ≈ Shortcut only ({shortcut_only:.0%})")
            print("→ No systematic preference when forced to choose")

        # Refusal detection (especially for frontier models)
        if refusal_rate > 0.1:
            print(f"\nRefusal rate ({refusal_rate:.0%}): Model explicitly expresses uncertainty")
            print("→ Model detects conflict but cannot resolve through reasoning")

        # Child concept analysis
        if child_rate > 0.8:
            print(f"\nChild conclusion rate ({child_rate:.0%}): Model consistently does child-level inference")
            print("→ Model generalizes from entity to child concept (1-hop)")
            print("→ But fails to propagate to parent concept (missing transitive step)")

    # Save results
    output_dir = pathlib.Path("factorial_results")
    output_dir.mkdir(exist_ok=True)

    # Use display_name for cleaner filenames
    output_file = output_dir / f"factorial_{sanitize_model_name(display_name)}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved to {output_file}")

    return results


def main():
    global NO_SYSTEM_PROMPT, CONCURRENT_REQUESTS, IS_REASONING_MODEL

    parser = argparse.ArgumentParser(
        description='Run factorial experiment on matched pair sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test local model (existing behavior):
  python run_factorial_experiment.py --model gemma2-9b --base-url "http://localhost:8000/v1"

  # Test GPT-4o via OpenRouter:
  python run_factorial_experiment.py --provider openrouter --model gpt-4o --concurrent 10

  # Test Claude Opus 4.5 via OpenRouter:
  python run_factorial_experiment.py --provider openrouter --model claude-opus-4.5 --concurrent 10

  # Test with full model ID:
  python run_factorial_experiment.py --provider openrouter --model anthropic/claude-opus-4.5 --concurrent 10

Available OpenRouter aliases:
  gpt-4o, llama-3-70b, deepseek-chat, deepseek-r1-70b, gpt-5, gemini-3-pro, claude-opus-4.5
        """
    )
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Model name or alias (e.g., gpt-4o, claude-opus-4.5, gemma2-9b)')
    parser.add_argument('--provider', type=str, default='openai',
                        choices=['openai', 'openrouter'],
                        help='API provider (openai for custom endpoints, openrouter for OpenRouter)')
    parser.add_argument('--base-url', '-b', type=str, default=None,
                        help='API base URL (required for openai provider, ignored for openrouter)')
    parser.add_argument('--no-system-prompt', action='store_true',
                        help='Merge system into user prompt (for Gemma 2 and similar)')
    parser.add_argument('--concurrent', '-c', type=int, default=None,
                        help='Concurrent requests (default: 20 for local, 5 for openrouter)')
    args = parser.parse_args()

    # Validate arguments
    if args.provider == 'openai' and not args.base_url:
        parser.error("--base-url is required when using --provider openai")

    # Set global flags
    NO_SYSTEM_PROMPT = args.no_system_prompt

    # Set up client based on provider
    from openai import AsyncOpenAI

    if args.provider == 'openrouter':
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            print("Error: OPENROUTER_API_KEY environment variable not set")
            print("Get your API key from https://openrouter.ai/keys")
            sys.exit(1)

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/inabhyd-benchmark",
                "X-Title": "INABHYD Reasoning Benchmark"
            }
        )

        # Resolve model alias to full ID
        model_id = resolve_model_name(args.model, 'openrouter')
        display_name = args.model  # Use alias for display

        # Set concurrency (lower default for OpenRouter)
        CONCURRENT_REQUESTS = args.concurrent if args.concurrent else 5

        if args.concurrent and args.concurrent > 10:
            print(f"Warning: High concurrency ({args.concurrent}) may hit OpenRouter rate limits. Consider --concurrent 5-10")

        # Print cost estimate
        if args.model in OPENROUTER_COSTS:
            cost_per_m = OPENROUTER_COSTS[args.model]
            # Rough estimate: ~500 tokens per example, 200 examples * 6 sets = 1200 examples
            est_tokens = 1200 * 500 / 1_000_000
            est_cost = est_tokens * cost_per_m
            print(f"Estimated cost for full run: ~${est_cost:.2f} (at ${cost_per_m}/1M tokens)")

    else:
        # Standard OpenAI-compatible endpoint
        client = AsyncOpenAI(
            base_url=args.base_url,
            api_key=os.environ.get('OPENAI_API_KEY', 'not-needed')
        )
        model_id = args.model
        display_name = args.model

        # Set concurrency (higher default for local)
        CONCURRENT_REQUESTS = args.concurrent if args.concurrent else 20

    # Check if this is a reasoning model (uses <think> tags)
    IS_REASONING_MODEL = is_reasoning_model(model_id)
    if IS_REASONING_MODEL:
        print(f"Note: {args.model} is a reasoning model - will strip <think> tags from responses")

    print(f"Provider: {args.provider}")
    print(f"Model ID: {model_id}")
    print(f"Concurrency: {CONCURRENT_REQUESTS}")
    print()

    asyncio.run(run_factorial_experiment(client, model_id, display_name))


if __name__ == "__main__":
    main()
