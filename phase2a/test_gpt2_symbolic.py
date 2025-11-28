"""
Test LLMs on symbolic ontology reasoning (same format as custom transformer).

This tests whether LLM performance on natural language was due to:
1. Not understanding the task structure, OR
2. Difficulty with natural language parsing

Supported models:
- GPT-2 family (gpt2, gpt2-medium, gpt2-large) - 1024 context
- GPT-Neo family (gpt-neo-125m, gpt-neo-1.3b) - 2048 context
- Pythia family (pythia-160m, pythia-410m) - 2048 context

Usage:
    python test_gpt2_symbolic.py --model gpt2 --num-samples 50
    python test_gpt2_symbolic.py --model gpt-neo-125m --num-samples 50
"""

import random
import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import deque
import torch
import argparse

# Import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Please install transformers: pip install transformers")
    exit(1)

# Import from the existing codebase
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Parent dir
from generate_symbolic_ontology import SymbolicOntologyGenerator


# =============================================================================
# Data Generation (reuse existing code)
# =============================================================================

def generate_symbolic_dataset(num_samples: int, tree_depth: int, seed: int = 42) -> List[Dict]:
    """Generate symbolic dataset filtered to depth_of_truth=0."""
    random.seed(seed)
    samples = []
    attempts = 0
    max_attempts = num_samples * 50

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1

        gen = SymbolicOntologyGenerator(
            depth=tree_depth,
            branching_factor=2,
            num_properties=5
        )

        # Use the generator's inductive task method
        sample = gen.generate_inductive_task()
        if sample and sample['metadata']['depth_of_truth'] == 0:
            sample['id'] = len(samples)
            samples.append(sample)

    return samples


# =============================================================================
# Few-shot Prompt Creation
# =============================================================================

def create_symbolic_few_shot_prompt(test_sample: Dict, few_shot_examples: List[Dict]) -> str:
    """Create a few-shot prompt using symbolic notation."""
    prompt = "Task: Given a world model and observations, infer the most general rule.\n\n"

    # Add few-shot examples
    for i, ex in enumerate(few_shot_examples):
        prompt += f"Example {i+1}:\n"
        prompt += f"{ex['input']}\n"
        prompt += f"[ANSWER]\n{ex['target']}\n\n"

    # Add test example (without answer)
    prompt += "Now solve:\n"
    prompt += f"{test_sample['input']}\n"
    prompt += "[ANSWER]\n"

    return prompt


# =============================================================================
# Model Configuration
# =============================================================================

MODEL_CONFIGS = {
    # GPT-2 family (1024 context)
    "gpt2": {"hf_name": "gpt2", "context_len": 1024},
    "gpt2-medium": {"hf_name": "gpt2-medium", "context_len": 1024},
    "gpt2-large": {"hf_name": "gpt2-large", "context_len": 1024},
    # GPT-Neo family (2048 context)
    "gpt-neo-125m": {"hf_name": "EleutherAI/gpt-neo-125M", "context_len": 2048},
    "gpt-neo-1.3b": {"hf_name": "EleutherAI/gpt-neo-1.3B", "context_len": 2048},
    # Pythia family (2048 context)
    "pythia-160m": {"hf_name": "EleutherAI/pythia-160m", "context_len": 2048},
    "pythia-410m": {"hf_name": "EleutherAI/pythia-410m", "context_len": 2048},
}


# =============================================================================
# LLM Evaluation
# =============================================================================

def generate_text(model, tokenizer, prompt: str, device: str, max_context: int = 1024, max_new_tokens: int = 30) -> str:
    """Generate text using greedy decoding."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Check if prompt is too long - truncate from beginning, leaving room for generation
    max_input = max_context - max_new_tokens - 10  # Leave buffer
    if input_ids.shape[1] > max_input:
        print(f"  [WARNING] Input truncated from {input_ids.shape[1]} to {max_input} tokens")
        input_ids = input_ids[:, -max_input:]

    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_tokens.append(next_token.item())

            # Stop at double newline or EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text


def evaluate_symbolic(model, tokenizer, test_samples: List[Dict],
                      few_shot_examples: List[Dict], device: str,
                      max_context: int = 1024, debug: bool = False) -> Dict:
    """Evaluate LLM on symbolic samples."""
    correct = 0
    total = 0
    results = []

    for i, sample in enumerate(test_samples):
        prompt = create_symbolic_few_shot_prompt(sample, few_shot_examples)

        # Generate - this now returns only the new tokens
        generated_answer = generate_text(model, tokenizer, prompt, device, max_context=max_context, max_new_tokens=30)
        generated_answer = generated_answer.strip().split('\n')[0]

        # Debug: show first few raw outputs
        if debug and i < 3:
            print(f"  DEBUG raw output: '{generated_answer[:100]}'")

        # Check if correct
        expected = sample['target'].strip()

        # Exact match or key components match
        is_correct = (
            expected == generated_answer or
            expected in generated_answer or
            (sample['metadata']['target_concept'] in generated_answer and
             sample['metadata']['target_property'] in generated_answer)
        )

        if is_correct:
            correct += 1
        total += 1

        results.append({
            'expected': expected,
            'generated': generated_answer[:80] if generated_answer else "(empty)",
            'correct': is_correct,
            'metadata': sample['metadata']
        })

    return {
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
        'results': results
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test LLMs on symbolic ontology tasks")
    parser.add_argument("--model", default="gpt2", choices=list(MODEL_CONFIGS.keys()),
                        help="Model to test (see MODEL_CONFIGS for options)")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-few-shot", type=int, default=2, help="Few-shot examples (keep small due to context)")
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Get model config
    model_config = MODEL_CONFIGS[args.model]
    hf_name = model_config["hf_name"]
    max_context = model_config["context_len"]

    # Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model using AutoModel/AutoTokenizer
    print(f"\nLoading {args.model} ({hf_name})...")
    print(f"Context length: {max_context} tokens")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForCausalLM.from_pretrained(hf_name).to(device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    all_results = {}

    print("\n" + "="*60)
    print(f"TESTING {args.model.upper()} ON SYMBOLIC ONTOLOGY REASONING")
    print("="*60)

    for depth in range(args.min_depth, args.max_depth + 1):
        hops = depth - 1
        print(f"\n{'='*60}")
        print(f"DEPTH-{depth} ({hops}-hop reasoning)")
        print(f"{'='*60}")

        # Generate test data
        print(f"Generating {args.num_samples} test samples...")
        test_samples = generate_symbolic_dataset(args.num_samples, depth, seed=args.seed + depth)

        if len(test_samples) == 0:
            print("ERROR: Could not generate samples")
            continue

        print(f"Generated {len(test_samples)} samples")

        # Generate few-shot examples (use depth-2 for simplicity)
        few_shot_samples = generate_symbolic_dataset(args.num_few_shot * 2, 2, seed=args.seed + 100)
        few_shot_examples = few_shot_samples[:args.num_few_shot]

        # Show sample
        sample = test_samples[0]
        print(f"\nSample input (first 200 chars):")
        print(f"  {sample['input'][:200]}...")
        print(f"\nExpected output: {sample['target']}")

        # Evaluate (debug=True for first depth to see raw output)
        print(f"\nEvaluating {len(test_samples)} samples...")
        result = evaluate_symbolic(model, tokenizer, test_samples, few_shot_examples, device,
                                   max_context=max_context, debug=(depth == args.min_depth))
        all_results[depth] = result

        # Show predictions
        print(f"\nSample predictions:")
        for r in result['results'][:3]:
            status = "✓" if r['correct'] else "✗"
            print(f"  [{status}] Expected: {r['expected']}")
            print(f"       Got: {r['generated']}")

        print(f"\n>>> DEPTH-{depth} ACCURACY: {result['accuracy']:.1%} ({result['correct']}/{result['total']}) <<<")

    # Summary
    print("\n" + "="*60)
    print(f"SUMMARY: {args.model.upper()} ON SYMBOLIC FORMAT")
    print("="*60)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params/1e6:.1f}M params, context: {max_context})")
    print()
    for depth, result in all_results.items():
        hops = depth - 1
        acc = result['accuracy']
        status = "PASS" if acc > 0.8 else "PARTIAL" if acc > 0.4 else "FAIL"
        print(f"Depth-{depth} ({hops}-hop): {acc:.1%} [{status}]")

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print("\nCustom Transformer (1.9M params, trained):")
    print("  Depth-2: 100.0%  |  Depth-3: 100.0%  |  Depth-4: 100.0%  |  Depth-5: 82.5%")

    print(f"\nGPT-2 on Natural Language (few-shot, 124M params):")
    print("  Depth-2: 42.0%   |  Depth-3: 16.0%   |  Depth-4: 6.0%    |  Depth-5: 8.0%")

    print(f"\nGPT-2 on Symbolic Format (few-shot, 124M params):")
    print("  Depth-2: 86.0%   |  Depth-3: 64.0%   |  Depth-4: 86.0%   |  Depth-5: 2.0% (context limit)")

    print(f"\n{args.model} on Symbolic Format (few-shot, {n_params/1e6:.1f}M params):")
    summary = "  "
    for depth in range(args.min_depth, args.max_depth + 1):
        if depth in all_results:
            summary += f"Depth-{depth}: {all_results[depth]['accuracy']:.1%}   |  "
    print(summary.rstrip("  |  "))


if __name__ == "__main__":
    main()
