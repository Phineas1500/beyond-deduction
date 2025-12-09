#!/usr/bin/env python3
"""
Generate a complete evaluation suite for INABHYD-style experiments.

This creates datasets matching the INABHYD paper's experimental conditions,
enabling systematic comparison of:
1. Symbolic FOL vs Natural Language performance
2. Performance across hops (1-4)
3. Performance across task types
4. Performance across difficulty levels

Usage:
    python generate_evaluation_suite.py --output-dir eval_data --samples-per-condition 100
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from itertools import product

import sys
sys.path.insert(0, str(Path(__file__).parent))

from isomorphic.core import OntologyConfig, TaskType
from isomorphic.difficulty import Difficulty
from generate_isomorphic_dataset import generate_dataset, save_dataset


# INABHYD experimental conditions
HOPS = [1, 2, 3, 4]
TASK_TYPES = {
    "property": [TaskType.INFER_PROPERTY],
    "membership": [TaskType.INFER_MEMBERSHIP],
    "subtype": [TaskType.INFER_SUBTYPE],
    "all": [TaskType.INFER_PROPERTY, TaskType.INFER_MEMBERSHIP, TaskType.INFER_SUBTYPE],
}
DIFFICULTIES = ["single", "easy"]  # Start with simpler conditions
# DIFFICULTIES = ["single", "easy", "medium", "hard"]  # Full INABHYD


def generate_evaluation_suite(
    output_dir: str,
    samples_per_condition: int = 100,
    seed: int = 42,
    include_hard: bool = False
) -> Dict[str, str]:
    """
    Generate complete evaluation suite.

    Returns dict mapping condition names to file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    difficulties = ["single", "easy", "medium", "hard"] if include_hard else ["single", "easy"]
    generated_files = {}

    total_conditions = len(HOPS) * len(TASK_TYPES) * len(difficulties)
    current = 0

    for hops in HOPS:
        for task_name, task_types in TASK_TYPES.items():
            for difficulty in difficulties:
                current += 1
                condition_name = f"{hops}hop_{task_name}_{difficulty}"

                print(f"[{current}/{total_conditions}] Generating {condition_name}...")

                # Skip invalid combinations
                if difficulty == "single" and len(task_types) > 1:
                    print(f"  Skipping (SINGLE difficulty requires single task type)")
                    continue

                diff_enum = Difficulty[difficulty.upper()]

                dataset = generate_dataset(
                    num_samples=samples_per_condition,
                    hops=hops,
                    task_types=task_types,
                    difficulty=diff_enum,
                    mix_hops=False,
                    seed=seed + current * 1000,
                    verbose=False
                )

                if dataset:
                    output_path = os.path.join(output_dir, f"{condition_name}.jsonl")
                    save_dataset(dataset, output_path, "jsonl")
                    generated_files[condition_name] = output_path
                    print(f"  Generated {len(dataset)} samples -> {output_path}")
                else:
                    print(f"  WARNING: No samples generated")

    return generated_files


def create_evaluation_manifest(output_dir: str, generated_files: Dict[str, str]):
    """Create a manifest file describing all generated datasets."""
    manifest = {
        "description": "INABHYD-style evaluation suite with isomorphic symbolic/NL pairs",
        "conditions": {},
    }

    for condition_name, file_path in generated_files.items():
        parts = condition_name.split("_")
        hops = int(parts[0].replace("hop", ""))
        task = parts[1]
        difficulty = parts[2]

        # Count samples
        with open(file_path, 'r') as f:
            num_samples = sum(1 for _ in f)

        manifest["conditions"][condition_name] = {
            "file": file_path,
            "hops": hops,
            "task_type": task,
            "difficulty": difficulty,
            "num_samples": num_samples,
        }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {manifest_path}")
    return manifest_path


def print_evaluation_plan():
    """Print recommended evaluation plan."""
    print("""
================================================================================
RECOMMENDED EVALUATION PLAN (based on INABHYD paper)
================================================================================

1. PRIMARY COMPARISON: Symbolic vs Natural Language
   ------------------------------------------------
   Your key finding: Models perform better on symbolic FOL than NL.

   For each model, compare:
   - Strong accuracy on symbolic format
   - Strong accuracy on NL format
   - Gap between formats (symbolic - NL)

   Present as: Table with rows=models, columns=format, cells=accuracy

2. SCALING WITH HOPS (Multi-hop Reasoning)
   ---------------------------------------
   Test if performance degrades with more reasoning hops.

   For each format (symbolic, NL):
   - Plot accuracy vs hops (1, 2, 3, 4)
   - Identify the "hop limit" where accuracy drops

   Present as: Line plot, x=hops, y=accuracy, lines=models

3. TASK TYPE COMPARISON
   --------------------
   Test which reasoning type is hardest.

   Compare accuracy on:
   - infer_property (inductive: induce property rule)
   - infer_membership (abductive: infer entity type)
   - infer_subtype (inductive: induce subtype relation)

   Present as: Grouped bar chart by task type

4. DIFFICULTY SCALING
   ------------------
   Test robustness to ambiguity.

   Compare SINGLE vs EASY vs MEDIUM vs HARD:
   - SINGLE: Exactly one hidden axiom (easiest)
   - HARD: Multiple hidden axioms with 30% probability

   Present as: Table with difficulty columns

5. PARSIMONY (Occam's Razor)
   -------------------------
   The core INABHYD finding: LLMs don't follow Occam's Razor.

   Report:
   - Strong accuracy (exact parsimonious match)
   - Weak accuracy (any valid hypothesis)
   - Gap = Weak - Strong (measures parsimony failure)

   If Strong << Weak: Model finds valid answers but not most general ones.

================================================================================
SUGGESTED TABLE FORMAT (like INABHYD paper)
================================================================================

Table 1: Symbolic vs NL Performance (Strong Accuracy %)
| Model          | Symbolic       | Natural Lang   | Gap    |
|----------------|----------------|----------------|--------|
|                | 1h  2h  3h  4h | 1h  2h  3h  4h |        |
| Your Model     | 98  95  88  72 | 45  32  18  10 | +52.5  |
| Qwen3-0.6B     | .. .. .. ..    | .. .. .. ..    | ..     |
| GPT-4o         | .. .. .. ..    | .. .. .. ..    | ..     |

Table 2: Performance by Task Type (2-hop, SINGLE difficulty)
| Model          | infer_property | infer_membership | infer_subtype |
|----------------|----------------|------------------|---------------|
| ...            | ...            | ...              | ...           |

Table 3: Parsimony Analysis (4-hop)
| Model          | Strong Acc | Weak Acc | Parsimony Gap |
|----------------|------------|----------|---------------|
| ...            | ...        | ...      | ...           |

================================================================================
""")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation suite for INABHYD-style experiments"
    )
    parser.add_argument("--output-dir", "-o", type=str, default="eval_data",
                        help="Output directory for generated datasets")
    parser.add_argument("--samples-per-condition", "-n", type=int, default=100,
                        help="Number of samples per experimental condition")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--include-hard", action="store_true",
                        help="Include MEDIUM and HARD difficulty levels")
    parser.add_argument("--plan-only", action="store_true",
                        help="Only print evaluation plan, don't generate data")

    args = parser.parse_args()

    print_evaluation_plan()

    if args.plan_only:
        return

    print("\n" + "=" * 60)
    print("GENERATING EVALUATION SUITE")
    print("=" * 60 + "\n")

    generated_files = generate_evaluation_suite(
        output_dir=args.output_dir,
        samples_per_condition=args.samples_per_condition,
        seed=args.seed,
        include_hard=args.include_hard
    )

    create_evaluation_manifest(args.output_dir, generated_files)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: Generated {len(generated_files)} datasets")
    print(f"{'=' * 60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Samples per condition: {args.samples_per_condition}")
    print(f"\nTo evaluate a model, load each dataset and compare:")
    print(f"  - sample['symbolic']['hypotheses'] (symbolic format)")
    print(f"  - sample['natural_language']['hypotheses'] (NL format)")


if __name__ == "__main__":
    main()
