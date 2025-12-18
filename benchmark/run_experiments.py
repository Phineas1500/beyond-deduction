"""
Unified experiment runner for INABHYD replication.
Generates examples, queries LLM, and evaluates results.
Supports OpenAI API and OpenAI-compatible endpoints (e.g., vLLM on Modal).
"""

import os
import sys
import pickle
import argparse
import pathlib
from random import seed, shuffle, randint
from functools import reduce
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Default endpoint settings (can be overridden via environment variables)
# For Modal Gemma: set OPENAI_BASE_URL to your Modal endpoint
# e.g., export OPENAI_BASE_URL="https://your-workspace--gemma3-27b-inference-serve.modal.run/v1"
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)  # None = use OpenAI default
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")  # vLLM doesn't need a key

from ontology import Ontology, OntologyConfig, Difficulty
from evaluate import (
    parse_hypotheses_from_response,
    parse_ground_truth,
    compute_strong_accuracy,
    compute_weak_accuracy,
    compute_quality,
    wilson_confidence_interval
)

SEED = 62471893
NUM_EXAMPLES = 100

# System prompt exactly matching paper's generate.py (including typo "assitant")
SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
        Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
        You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
    .   Only output final hypotheses.
 """


def make_user_prompt(ontology):
    """Create user prompt exactly matching paper's generate.py format."""
    return "Q: " + ontology.theories + " We observe that: " + ontology.observations + " Please come up with hypothesis to explain observations."


def make_completion_prompt(system_prompt, user_prompt):
    """Create a completion-style prompt for base models (no chat template)."""
    return f"""{system_prompt}

{user_prompt}

A:"""


# Base models that need completions API instead of chat
BASE_MODELS = ['pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'pythia-12b']


def is_base_model(model_name):
    """Check if model is a base model (needs completions API)."""
    return any(base in model_name.lower() for base in BASE_MODELS)


def sanitize_model_name(model_name):
    """Convert model name to safe filename string."""
    # e.g., "gpt-4o" -> "gpt4o", "gemma3-27b" -> "gemma3_27b"
    return model_name.replace("-", "").replace("/", "_").replace(":", "_").lower()


def generate_single_example(config):
    """Generate a single reasoning example."""
    while True:
        try:
            ontology = Ontology(config)
            return ontology
        except Exception:
            pass


def get_openai_client(base_url=None):
    """Create OpenAI client with optional custom base URL for vLLM/Modal."""
    from openai import OpenAI

    if base_url or DEFAULT_BASE_URL:
        return OpenAI(
            base_url=base_url or DEFAULT_BASE_URL,
            api_key=DEFAULT_API_KEY,
        )
    return OpenAI()


def run_zero_shot_single_hypothesis(model_name, task_type, height, num_examples=NUM_EXAMPLES, base_url=None):
    """
    Run zero-shot single hypothesis experiment.
    task_type: 'property', 'membership', or 'ontology'
    """
    client = get_openai_client(base_url)

    # Seed both Python random and numpy random for reproducibility
    seed(SEED)
    np.random.seed(SEED)

    # Configure based on task type
    config = OntologyConfig(
        hops=height,
        recover_membership=(task_type == 'membership'),
        recover_ontology=(task_type == 'ontology'),
        recover_property=(task_type == 'property'),
        difficulty=Difficulty.SINGLE,
        mix_hops=False
    )

    print(f"\n{'='*60}")
    print(f"Zero-shot Single Hypothesis: {task_type}, Height {height}")
    print(f"{'='*60}")

    examples = []
    replies = []
    results = []

    for i in range(num_examples):
        if i % 10 == 0:
            print(f"  Processing example {i+1}/{num_examples}...")

        ontology = generate_single_example(config)
        examples.append(ontology)

        user_prompt = make_user_prompt(ontology)

        try:
            if is_base_model(model_name):
                # Use completions API for base models
                prompt = make_completion_prompt(SYSTEM_PROMPT, user_prompt)
                completion = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=256
                )
                reply = completion.choices[0].text.strip()
            else:
                # Use chat completions API for instruction-tuned models
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0
                )
                reply = completion.choices[0].message.content
        except Exception as e:
            print(f"  Error on example {i+1}: {e}")
            reply = ""

        replies.append(reply)

        # Evaluate
        pred_hyps = parse_hypotheses_from_response(reply)
        gt_hyps = parse_ground_truth(ontology.hypotheses)

        strong_acc = compute_strong_accuracy(pred_hyps, gt_hyps)
        weak_acc = compute_weak_accuracy(pred_hyps, gt_hyps, ontology.observations, ontology.theories)
        quality = compute_quality(pred_hyps, gt_hyps, ontology.observations, ontology.theories)

        if strong_acc == 1:
            weak_acc = 1
            quality = 1.0

        results.append({
            'strong': strong_acc,
            'weak': weak_acc,
            'quality': quality
        })

    # Compute aggregate metrics
    strong_mean = np.mean([r['strong'] for r in results])
    weak_mean = np.mean([r['weak'] for r in results])
    quality_mean = np.mean([r['quality'] for r in results])

    strong_ci = wilson_confidence_interval(strong_mean, num_examples)
    weak_ci = wilson_confidence_interval(weak_mean, num_examples)

    print(f"\n  Results:")
    print(f"    Strong Accuracy: {strong_mean:.3f} [{strong_ci[0]:.3f}, {strong_ci[1]:.3f}]")
    print(f"    Weak Accuracy: {weak_mean:.3f} [{weak_ci[0]:.3f}, {weak_ci[1]:.3f}]")
    print(f"    Quality: {quality_mean:.3f}")

    # Save results
    output_dir = pathlib.Path("results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"zeroshot_single_{task_type}_h{height}_{sanitize_model_name(model_name)}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({
            'examples': examples,
            'replies': replies,
            'results': results,
            'metrics': {
                'strong_accuracy': strong_mean,
                'weak_accuracy': weak_mean,
                'quality': quality_mean,
                'strong_ci': strong_ci,
                'weak_ci': weak_ci
            }
        }, f)

    print(f"  Saved to {output_file}")

    return {
        'strong_accuracy': strong_mean,
        'weak_accuracy': weak_mean,
        'quality': quality_mean
    }


def run_zero_shot_multiple_hypothesis(model_name, height, num_examples=NUM_EXAMPLES, base_url=None):
    """Run zero-shot multiple hypothesis experiment."""
    client = get_openai_client(base_url)

    # Seed both Python random and numpy random for reproducibility
    seed(SEED)
    np.random.seed(SEED)

    config = OntologyConfig(
        hops=height,
        recover_membership=True,
        recover_ontology=True,
        recover_property=True,
        difficulty=Difficulty.EASY,
        mix_hops=True
    )

    print(f"\n{'='*60}")
    print(f"Zero-shot Multiple Hypotheses: Height {height}")
    print(f"{'='*60}")

    examples = []
    replies = []
    results = []

    for i in range(num_examples):
        if i % 10 == 0:
            print(f"  Processing example {i+1}/{num_examples}...")

        ontology = generate_single_example(config)
        examples.append(ontology)

        user_prompt = make_user_prompt(ontology)

        try:
            if is_base_model(model_name):
                # Use completions API for base models
                prompt = make_completion_prompt(SYSTEM_PROMPT, user_prompt)
                completion = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=256
                )
                reply = completion.choices[0].text.strip()
            else:
                # Use chat completions API for instruction-tuned models
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0
                )
                reply = completion.choices[0].message.content
        except Exception as e:
            print(f"  Error on example {i+1}: {e}")
            reply = ""

        replies.append(reply)

        # Evaluate
        pred_hyps = parse_hypotheses_from_response(reply)
        gt_hyps = parse_ground_truth(ontology.hypotheses)

        strong_acc = compute_strong_accuracy(pred_hyps, gt_hyps)
        weak_acc = compute_weak_accuracy(pred_hyps, gt_hyps, ontology.observations, ontology.theories)
        quality = compute_quality(pred_hyps, gt_hyps, ontology.observations, ontology.theories)

        if strong_acc == 1:
            weak_acc = 1
            quality = 1.0

        results.append({
            'strong': strong_acc,
            'weak': weak_acc,
            'quality': quality
        })

    # Compute aggregate metrics
    strong_mean = np.mean([r['strong'] for r in results])
    weak_mean = np.mean([r['weak'] for r in results])
    quality_mean = np.mean([r['quality'] for r in results])

    strong_ci = wilson_confidence_interval(strong_mean, num_examples)
    weak_ci = wilson_confidence_interval(weak_mean, num_examples)

    print(f"\n  Results:")
    print(f"    Strong Accuracy: {strong_mean:.3f} [{strong_ci[0]:.3f}, {strong_ci[1]:.3f}]")
    print(f"    Weak Accuracy: {weak_mean:.3f} [{weak_ci[0]:.3f}, {weak_ci[1]:.3f}]")
    print(f"    Quality: {quality_mean:.3f}")

    # Save results
    output_dir = pathlib.Path("results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"zeroshot_multi_h{height}_{sanitize_model_name(model_name)}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({
            'examples': examples,
            'replies': replies,
            'results': results,
            'metrics': {
                'strong_accuracy': strong_mean,
                'weak_accuracy': weak_mean,
                'quality': quality_mean,
                'strong_ci': strong_ci,
                'weak_ci': weak_ci
            }
        }, f)

    print(f"  Saved to {output_file}")

    return {
        'strong_accuracy': strong_mean,
        'weak_accuracy': weak_mean,
        'quality': quality_mean
    }


def run_icl_experiment(model_name, height, num_shots=8, ood=False, num_examples=NUM_EXAMPLES, base_url=None):
    """Run in-context learning experiment."""
    client = get_openai_client(base_url)

    # Seed both Python random and numpy random for reproducibility
    seed(SEED)
    np.random.seed(SEED)

    test_config = OntologyConfig(
        hops=height,
        recover_membership=True,
        recover_ontology=True,
        recover_property=True,
        difficulty=Difficulty.EASY,
        mix_hops=True
    )

    # For OOD, demos use height-1 single hypothesis; for in-distribution, same config
    if ood:
        demo_configs = [
            OntologyConfig(1, recover_membership=True, difficulty=Difficulty.SINGLE),
            OntologyConfig(1, recover_ontology=True, difficulty=Difficulty.SINGLE),
            OntologyConfig(1, recover_property=True, difficulty=Difficulty.SINGLE),
        ]
    else:
        demo_configs = [test_config] * num_shots

    exp_type = "OOD" if ood else "In-Distribution"
    print(f"\n{'='*60}")
    print(f"{num_shots}-shot ICL ({exp_type}): Height {height}")
    print(f"{'='*60}")

    examples = []
    replies = []
    results = []

    for i in range(num_examples):
        if i % 10 == 0:
            print(f"  Processing example {i+1}/{num_examples}...")

        # Generate demo examples
        demos = []
        for j in range(num_shots):
            if ood:
                demo_config = demo_configs[j % len(demo_configs)]
            else:
                demo_config = test_config
            demo_ontology = generate_single_example(demo_config)
            demos.append(demo_ontology)

        # Generate test example
        test_ontology = generate_single_example(test_config)
        examples.append({'demos': demos, 'test': test_ontology})

        # Build prompt with demonstrations
        test_q = make_user_prompt(test_ontology)

        try:
            if is_base_model(model_name):
                # Build few-shot completion prompt for base models
                prompt_parts = [SYSTEM_PROMPT, ""]
                for demo in demos:
                    demo_q = make_user_prompt(demo)
                    demo_a = demo.CoT
                    prompt_parts.append(f"{demo_q}\n\nA: {demo_a}")
                    prompt_parts.append("")
                prompt_parts.append(f"{test_q}\n\nA:")
                prompt = "\n".join(prompt_parts)

                completion = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=256
                )
                reply = completion.choices[0].text.strip()
            else:
                # Use chat completions API for instruction-tuned models
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]

                for demo in demos:
                    demo_q = make_user_prompt(demo)
                    # Use CoT (Chain-of-Thought) as per paper Section 4.3:
                    # "demonstrations' ground truth hypotheses and CoTs"
                    demo_a = demo.CoT
                    messages.append({"role": "user", "content": demo_q})
                    messages.append({"role": "assistant", "content": demo_a})

                messages.append({"role": "user", "content": test_q})

                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0
                )
                reply = completion.choices[0].message.content
        except Exception as e:
            print(f"  Error on example {i+1}: {e}")
            reply = ""

        replies.append(reply)

        # Evaluate
        pred_hyps = parse_hypotheses_from_response(reply)
        gt_hyps = parse_ground_truth(test_ontology.hypotheses)

        strong_acc = compute_strong_accuracy(pred_hyps, gt_hyps)
        weak_acc = compute_weak_accuracy(pred_hyps, gt_hyps, test_ontology.observations, test_ontology.theories)
        quality = compute_quality(pred_hyps, gt_hyps, test_ontology.observations, test_ontology.theories)

        if strong_acc == 1:
            weak_acc = 1
            quality = 1.0

        results.append({
            'strong': strong_acc,
            'weak': weak_acc,
            'quality': quality
        })

    # Compute aggregate metrics
    strong_mean = np.mean([r['strong'] for r in results])
    weak_mean = np.mean([r['weak'] for r in results])
    quality_mean = np.mean([r['quality'] for r in results])

    strong_ci = wilson_confidence_interval(strong_mean, num_examples)
    weak_ci = wilson_confidence_interval(weak_mean, num_examples)

    print(f"\n  Results:")
    print(f"    Strong Accuracy: {strong_mean:.3f} [{strong_ci[0]:.3f}, {strong_ci[1]:.3f}]")
    print(f"    Weak Accuracy: {weak_mean:.3f} [{weak_ci[0]:.3f}, {weak_ci[1]:.3f}]")
    print(f"    Quality: {quality_mean:.3f}")

    # Save results
    output_dir = pathlib.Path("results")
    output_dir.mkdir(exist_ok=True)

    suffix = "ood" if ood else "id"
    output_file = output_dir / f"icl{num_shots}_{suffix}_h{height}_{sanitize_model_name(model_name)}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({
            'examples': examples,
            'replies': replies,
            'results': results,
            'metrics': {
                'strong_accuracy': strong_mean,
                'weak_accuracy': weak_mean,
                'quality': quality_mean,
                'strong_ci': strong_ci,
                'weak_ci': weak_ci
            }
        }, f)

    print(f"  Saved to {output_file}")

    return {
        'strong_accuracy': strong_mean,
        'weak_accuracy': weak_mean,
        'quality': quality_mean
    }


def run_all_experiments(model_name="gpt-4o", num_examples=NUM_EXAMPLES):
    """Run all experiments for GPT-4o replication."""
    all_results = {}

    # 1. Zero-shot single hypothesis (Figure 2)
    print("\n" + "="*70)
    print("RUNNING ZERO-SHOT SINGLE HYPOTHESIS EXPERIMENTS (Figure 2)")
    print("="*70)

    for task in ['property', 'membership', 'ontology']:
        all_results[f'zeroshot_single_{task}'] = {}
        for height in range(1, 5):
            key = f'h{height}'
            all_results[f'zeroshot_single_{task}'][key] = run_zero_shot_single_hypothesis(
                model_name, task, height, num_examples
            )

    # 2. Zero-shot multiple hypotheses (Figure 3 top)
    print("\n" + "="*70)
    print("RUNNING ZERO-SHOT MULTIPLE HYPOTHESES EXPERIMENTS (Figure 3 top)")
    print("="*70)

    all_results['zeroshot_multi'] = {}
    for height in range(1, 5):
        key = f'h{height}'
        all_results['zeroshot_multi'][key] = run_zero_shot_multiple_hypothesis(
            model_name, height, num_examples
        )

    # 3. 8-shot ICL in-distribution (Figure 3 middle)
    print("\n" + "="*70)
    print("RUNNING 8-SHOT ICL IN-DISTRIBUTION EXPERIMENTS (Figure 3 middle)")
    print("="*70)

    all_results['icl8_id'] = {}
    for height in range(1, 5):
        key = f'h{height}'
        all_results['icl8_id'][key] = run_icl_experiment(
            model_name, height, num_shots=8, ood=False, num_examples=num_examples
        )

    # 4. 8-shot ICL out-of-distribution (Figure 3 bottom)
    print("\n" + "="*70)
    print("RUNNING 8-SHOT ICL OUT-OF-DISTRIBUTION EXPERIMENTS (Figure 3 bottom)")
    print("="*70)

    all_results['icl8_ood'] = {}
    for height in range(1, 5):
        key = f'h{height}'
        all_results['icl8_ood'][key] = run_icl_experiment(
            model_name, height, num_shots=8, ood=True, num_examples=num_examples
        )

    # Save all results
    output_dir = pathlib.Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "all_results_gpt4o.pkl", 'wb') as f:
        pickle.dump(all_results, f)

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print_results_summary(all_results)

    return all_results


def print_results_summary(all_results):
    """Print a summary table of all results."""
    print("\n--- Zero-shot Single Hypothesis ---")
    print(f"{'Task':<12} {'H1':>12} {'H2':>12} {'H3':>12} {'H4':>12}")
    print("-" * 60)

    for task in ['property', 'membership', 'ontology']:
        key = f'zeroshot_single_{task}'
        if key in all_results:
            row = f"{task:<12}"
            for h in range(1, 5):
                hkey = f'h{h}'
                if hkey in all_results[key]:
                    weak = all_results[key][hkey]['weak_accuracy']
                    row += f" {weak:>11.3f}"
                else:
                    row += f" {'N/A':>11}"
            print(row)

    print("\n--- Zero-shot Multiple Hypotheses ---")
    print(f"{'Metric':<12} {'H1':>12} {'H2':>12} {'H3':>12} {'H4':>12}")
    print("-" * 60)

    if 'zeroshot_multi' in all_results:
        for metric in ['weak_accuracy', 'strong_accuracy', 'quality']:
            row = f"{metric:<12}"
            for h in range(1, 5):
                hkey = f'h{h}'
                if hkey in all_results['zeroshot_multi']:
                    val = all_results['zeroshot_multi'][hkey][metric]
                    row += f" {val:>11.3f}"
                else:
                    row += f" {'N/A':>11}"
            print(row)

    print("\n--- 8-shot ICL In-Distribution ---")
    print(f"{'Metric':<12} {'H1':>12} {'H2':>12} {'H3':>12} {'H4':>12}")
    print("-" * 60)

    if 'icl8_id' in all_results:
        for metric in ['weak_accuracy', 'strong_accuracy', 'quality']:
            row = f"{metric:<12}"
            for h in range(1, 5):
                hkey = f'h{h}'
                if hkey in all_results['icl8_id']:
                    val = all_results['icl8_id'][hkey][metric]
                    row += f" {val:>11.3f}"
                else:
                    row += f" {'N/A':>11}"
            print(row)

    print("\n--- 8-shot ICL Out-of-Distribution ---")
    print(f"{'Metric':<12} {'H1':>12} {'H2':>12} {'H3':>12} {'H4':>12}")
    print("-" * 60)

    if 'icl8_ood' in all_results:
        for metric in ['weak_accuracy', 'strong_accuracy', 'quality']:
            row = f"{metric:<12}"
            for h in range(1, 5):
                hkey = f'h{h}'
                if hkey in all_results['icl8_ood']:
                    val = all_results['icl8_ood'][hkey][metric]
                    row += f" {val:>11.3f}"
                else:
                    row += f" {'N/A':>11}"
            print(row)


def main():
    parser = argparse.ArgumentParser(description='Run INABHYD experiments')
    parser.add_argument('--model', '-m', type=str, default='gpt-4o',
                        help='Model to use (default: gpt-4o, or gemma3-27b for Modal)')
    parser.add_argument('--base-url', '-b', type=str, default=None,
                        help='Base URL for OpenAI-compatible API (e.g., Modal vLLM endpoint)')
    parser.add_argument('--num-examples', '-n', type=int, default=100,
                        help='Number of examples per configuration (default: 100)')
    parser.add_argument('--experiment', '-e', type=str, default='all',
                        choices=['all', 'zeroshot_single', 'zeroshot_multi', 'icl_id', 'icl_ood'],
                        help='Which experiment to run')
    parser.add_argument('--task', '-t', type=str, default='all',
                        choices=['all', 'property', 'membership', 'ontology'],
                        help='Task type for single hypothesis experiments')
    parser.add_argument('--height', type=int, default=0,
                        choices=[0, 1, 2, 3, 4],
                        help='Ontology tree height (0 for all heights)')
    args = parser.parse_args()

    if args.experiment == 'all':
        run_all_experiments(args.model, args.num_examples)
    elif args.experiment == 'zeroshot_single':
        tasks = ['property', 'membership', 'ontology'] if args.task == 'all' else [args.task]
        heights = range(1, 5) if args.height == 0 else [args.height]
        for task in tasks:
            for height in heights:
                run_zero_shot_single_hypothesis(args.model, task, height, args.num_examples, args.base_url)
    elif args.experiment == 'zeroshot_multi':
        heights = range(1, 5) if args.height == 0 else [args.height]
        for height in heights:
            run_zero_shot_multiple_hypothesis(args.model, height, args.num_examples, args.base_url)
    elif args.experiment == 'icl_id':
        heights = range(1, 5) if args.height == 0 else [args.height]
        for height in heights:
            run_icl_experiment(args.model, height, num_shots=8, ood=False, num_examples=args.num_examples, base_url=args.base_url)
    elif args.experiment == 'icl_ood':
        heights = range(1, 5) if args.height == 0 else [args.height]
        for height in heights:
            run_icl_experiment(args.model, height, num_shots=8, ood=True, num_examples=args.num_examples, base_url=args.base_url)


if __name__ == '__main__':
    main()
