#!/usr/bin/env python3
"""
Evaluate models on INABHYD-style isomorphic datasets.

This script evaluates models on both symbolic FOL and natural language formats,
computing strong/weak accuracy and parsimony metrics.

Usage:
    # Evaluate a HuggingFace model
    python evaluate_model.py --model Qwen/Qwen3-0.6B --data eval_data/2hop_property_single.jsonl

    # Evaluate your trained transformer
    python evaluate_model.py --model local:depth2_trained.pt --data eval_data/

    # Evaluate with specific formats
    python evaluate_model.py --model gpt-4o --format symbolic --data eval_data/
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import random

sys.path.insert(0, str(Path(__file__).parent))

from isomorphic.evaluation import ParsimonyEvaluator, normalize_hypothesis


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset."""
    samples = []
    with open(path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def format_symbolic_prompt(sample: Dict, few_shot: List[Dict] = None, use_cot: bool = False) -> str:
    """Format a symbolic FOL prompt for the model."""
    prompt = ""

    # Add few-shot examples if provided
    if few_shot:
        for ex in few_shot:
            prompt += "[WORLD_MODEL]\n"
            prompt += "\n".join(ex["symbolic"]["world_model"]) + "\n"
            prompt += "[OBSERVATIONS]\n"
            prompt += "\n".join(ex["symbolic"]["observations"]) + "\n"
            prompt += "[ANSWER]\n"
            if use_cot and "cot" in ex["symbolic"]:
                prompt += "\n".join(ex["symbolic"]["cot"]) + "\n"
                prompt += "[HYPOTHESIS]\n"
            prompt += "\n".join(ex["symbolic"]["hypotheses"]) + "\n\n"

    # Add test example
    prompt += "[WORLD_MODEL]\n"
    prompt += "\n".join(sample["symbolic"]["world_model"]) + "\n"
    prompt += "[OBSERVATIONS]\n"
    prompt += "\n".join(sample["symbolic"]["observations"]) + "\n"
    if use_cot:
        prompt += "[TASK]\nShow your reasoning step by step, then infer the hidden axiom(s).\n"
    prompt += "[ANSWER]\n"

    return prompt


def format_nl_prompt(sample: Dict, few_shot: List[Dict] = None, use_cot: bool = False) -> str:
    """Format a natural language prompt (INABHYD style)."""
    if use_cot:
        system = (
            "You are a helpful assistant that performs abduction and induction reasoning. "
            "Your job is to come up with hypotheses that explain observations with given theories. "
            "Each hypothesis should explain as many observations as possible. "
            "You can come up with multiple hypotheses and each hypothesis should take one line "
            "with the format A is B or A is not B. "
            "Show your reasoning step by step before giving the final answer."
        )
    else:
        system = (
            "You are a helpful assistant that performs abduction and induction reasoning. "
            "Your job is to come up with hypotheses that explain observations with given theories. "
            "Each hypothesis should explain as many observations as possible. "
            "You can come up with multiple hypotheses and each hypothesis should take one line "
            "with the format A is B or A is not B. Only output final hypotheses."
        )

    prompt = system + "\n\n"

    # Add few-shot examples if provided
    if few_shot:
        for ex in few_shot:
            prompt += f"Q: {ex['natural_language']['theories']} "
            prompt += f"We observe that: {ex['natural_language']['observations']} "
            prompt += "Please come up with hypothesis to explain observations.\n"
            if use_cot and "cot" in ex["natural_language"]:
                prompt += f"Reasoning: {ex['natural_language']['cot']}\n"
                prompt += f"Therefore: {ex['natural_language']['hypotheses']}\n\n"
            else:
                prompt += f"A: {ex['natural_language']['hypotheses']}\n\n"

    # Add test example
    prompt += f"Q: {sample['natural_language']['theories']} "
    prompt += f"We observe that: {sample['natural_language']['observations']} "
    prompt += "Please come up with hypothesis to explain observations.\n"
    if use_cot:
        prompt += "Reasoning:"
    else:
        prompt += "A:"

    return prompt


def evaluate_with_hf_model(
    model_name: str,
    samples: List[Dict],
    format_type: str = "both",
    few_shot_count: int = 0,
    max_samples: int = None,
    device: str = "auto",
    use_cot: bool = False
) -> Dict[str, Any]:
    """
    Evaluate using a HuggingFace model.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-0.6B")
        samples: List of test samples
        format_type: "symbolic", "nl", or "both"
        few_shot_count: Number of few-shot examples
        max_samples: Maximum samples to evaluate (None = all)
        device: Device to use ("auto", "cuda", "mps", "cpu")
        use_cot: Whether to use chain-of-thought prompting

    Returns:
        Dict with evaluation results
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)
    model.eval()

    print(f"Using device: {device}")

    # Limit samples if specified
    if max_samples:
        samples = samples[:max_samples]

    # Get few-shot examples from beginning (use different samples for test)
    few_shot_samples = samples[:few_shot_count] if few_shot_count > 0 else []
    test_samples = samples[few_shot_count:]

    results = {"symbolic": None, "nl": None}

    for fmt in ["symbolic", "nl"]:
        if format_type != "both" and format_type != fmt:
            continue

        print(f"\nEvaluating {fmt} format...")
        evaluator = ParsimonyEvaluator()
        predictions = []

        for i, sample in enumerate(test_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{len(test_samples)}")

            # Format prompt
            if fmt == "symbolic":
                prompt = format_symbolic_prompt(sample, few_shot_samples, use_cot=use_cot)
                ground_truth = sample["symbolic"]["hypotheses"]
            else:
                prompt = format_nl_prompt(sample, few_shot_samples, use_cot=use_cot)
                ground_truth = sample["natural_language"]["hypotheses"]

            # Generate (more tokens if CoT)
            max_tokens = 300 if use_cot else 100
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode response
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Extract hypothesis from CoT response if needed
            if use_cot:
                # Try to find the hypothesis after "Therefore:" or "[HYPOTHESIS]"
                if "Therefore:" in response:
                    response = response.split("Therefore:")[-1].strip()
                elif "[HYPOTHESIS]" in response:
                    response = response.split("[HYPOTHESIS]")[-1].strip()
                # Take first line of the hypothesis part
                response = response.strip().split("\n")[0]
            else:
                response = response.strip().split("\n")[0]  # Take first line

            predictions.append(response)

            # Evaluate
            gt_str = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
            evaluator.evaluate_sample(
                prediction=response,
                ground_truth=gt_str,
                metadata={
                    "sample_id": sample["id"],
                    "hops": sample["config"]["hops"],
                    "task_types": sample["config"]["task_types"],
                }
            )

        results[fmt] = {
            "metrics": evaluator.compute_metrics(),
            "predictions": predictions,
            "by_task": evaluator.compute_metrics_by_task_type(),
        }

    return results


def evaluate_directory(
    model_name: str,
    data_dir: str,
    format_type: str = "both",
    few_shot_count: int = 0,
    max_samples_per_file: int = None,
    use_cot: bool = False
) -> Dict[str, Dict]:
    """Evaluate model on all datasets in a directory."""
    results = {}

    # Find all JSONL files
    data_path = Path(data_dir)
    jsonl_files = list(data_path.glob("*.jsonl"))

    print(f"Found {len(jsonl_files)} dataset files in {data_dir}")

    for jsonl_file in sorted(jsonl_files):
        condition_name = jsonl_file.stem
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {condition_name}")
        print(f"{'=' * 60}")

        samples = load_dataset(str(jsonl_file))
        result = evaluate_with_hf_model(
            model_name=model_name,
            samples=samples,
            format_type=format_type,
            few_shot_count=few_shot_count,
            max_samples=max_samples_per_file,
            use_cot=use_cot
        )
        results[condition_name] = result

    return results


def print_results_table(results: Dict[str, Dict]):
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Group by hops
    by_hops = defaultdict(list)
    for condition, result in results.items():
        parts = condition.split("_")
        hops = parts[0]
        by_hops[hops].append((condition, result))

    # Print summary table
    print("\nStrong Accuracy by Condition:")
    print("-" * 60)
    print(f"{'Condition':<30} {'Symbolic':>12} {'NL':>12} {'Gap':>10}")
    print("-" * 60)

    for hops in sorted(by_hops.keys()):
        for condition, result in sorted(by_hops[hops]):
            sym_acc = result.get("symbolic", {}).get("metrics", {}).get("strong_accuracy", 0)
            nl_acc = result.get("nl", {}).get("metrics", {}).get("strong_accuracy", 0)
            gap = sym_acc - nl_acc if sym_acc and nl_acc else 0

            sym_str = f"{sym_acc:.1%}" if sym_acc else "N/A"
            nl_str = f"{nl_acc:.1%}" if nl_acc else "N/A"
            gap_str = f"+{gap:.1%}" if gap > 0 else f"{gap:.1%}"

            print(f"{condition:<30} {sym_str:>12} {nl_str:>12} {gap_str:>10}")

    print("-" * 60)

    # Print weak vs strong comparison
    print("\nParsimony Analysis (Strong vs Weak Accuracy):")
    print("-" * 60)
    print(f"{'Condition':<30} {'Strong':>10} {'Weak':>10} {'Gap':>10}")
    print("-" * 60)

    for hops in sorted(by_hops.keys()):
        for condition, result in sorted(by_hops[hops]):
            for fmt in ["symbolic", "nl"]:
                if fmt not in result or result[fmt] is None:
                    continue
                metrics = result[fmt].get("metrics", {})
                strong = metrics.get("strong_accuracy", 0)
                weak = metrics.get("weak_accuracy", 0)
                gap = weak - strong

                print(f"{condition} ({fmt[:3]}){'':<10} {strong:>10.1%} {weak:>10.1%} {gap:>10.1%}")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on INABHYD-style isomorphic datasets"
    )
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model name (HuggingFace model ID or 'local:path.pt')")
    parser.add_argument("--data", "-d", type=str, required=True,
                        help="Path to dataset file or directory")
    parser.add_argument("--format", "-f", type=str, default="both",
                        choices=["symbolic", "nl", "both"],
                        help="Format to evaluate (default: both)")
    parser.add_argument("--few-shot", type=int, default=0,
                        help="Number of few-shot examples")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples per dataset")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file for results (JSON)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to use")
    parser.add_argument("--cot", action="store_true",
                        help="Use chain-of-thought prompting")

    args = parser.parse_args()

    # Check if data is file or directory
    data_path = Path(args.data)

    if data_path.is_file():
        print(f"Evaluating on single file: {args.data}")
        samples = load_dataset(args.data)
        results = {
            data_path.stem: evaluate_with_hf_model(
                model_name=args.model,
                samples=samples,
                format_type=args.format,
                few_shot_count=args.few_shot,
                max_samples=args.max_samples,
                device=args.device,
                use_cot=args.cot
            )
        }
    elif data_path.is_dir():
        print(f"Evaluating on directory: {args.data}")
        results = evaluate_directory(
            model_name=args.model,
            data_dir=args.data,
            format_type=args.format,
            few_shot_count=args.few_shot,
            max_samples_per_file=args.max_samples,
            use_cot=args.cot
        )
    else:
        print(f"Error: {args.data} not found")
        return

    # Print results
    print_results_table(results)

    # Save results if output specified
    if args.output:
        # Convert to serializable format
        output_data = {
            "model": args.model,
            "format": args.format,
            "few_shot": args.few_shot,
            "cot": args.cot,
            "results": {}
        }
        for condition, result in results.items():
            output_data["results"][condition] = {
                fmt: {
                    "metrics": data["metrics"],
                    "by_task": data.get("by_task", {})
                }
                for fmt, data in result.items()
                if data is not None
            }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
