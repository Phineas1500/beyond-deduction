#!/usr/bin/env python3
"""
Generate INABHYD-complete isomorphic datasets.

Produces parallel symbolic FOL and natural language versions
of the same reasoning tasks, with identical logical structure.

Usage:
    # Single task type (inductive reasoning)
    python generate_isomorphic_dataset.py --task infer_property --hops 2 --samples 500

    # All task types with multiple hypotheses
    python generate_isomorphic_dataset.py --task all --hops 3 --difficulty easy --mix-hops

    # Full INABHYD replication
    python generate_isomorphic_dataset.py --task all --hops 4 --difficulty hard --mix-hops

Output format (JSONL):
{
  "id": 0,
  "config": {"hops": 2, "difficulty": "EASY", "task_types": ["INFER_PROPERTY"]},
  "symbolic": {"world_model": [...], "observations": [...], "hypotheses": [...], "mapping": {...}},
  "natural_language": {"theories": "...", "observations": "...", "hypotheses": "..."},
  "metadata": {"depths": [0], "num_hypotheses": 1, ...}
}
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from isomorphic.core import Ontology, OntologyConfig, TaskType
from isomorphic.difficulty import Difficulty
from isomorphic.symbolic_renderer import SymbolicRenderer
from isomorphic.nl_renderer import NLRenderer


def generate_sample(
    config: OntologyConfig,
    sample_id: int,
    seed: int,
    include_cot: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Generate a single isomorphic sample.

    Args:
        config: Ontology configuration
        sample_id: Unique sample ID
        seed: Random seed for this sample
        include_cot: Whether to include chain-of-thought reasoning

    Returns:
        Sample dict or None if generation failed
    """
    try:
        # Create ontology
        ontology = Ontology(config, seed=seed)

        # Check we have hypotheses (something to infer)
        hypotheses = ontology.get_hypotheses()
        if not hypotheses:
            return None

        # Render both formats with same seed for consistency
        random.seed(seed)
        symbolic_renderer = SymbolicRenderer()
        symbolic = symbolic_renderer.render_full(ontology)

        random.seed(seed)
        nl_renderer = NLRenderer()
        nl = nl_renderer.render_full(ontology)

        # Get CoT if requested
        symbolic_cot = None
        nl_cot = None
        if include_cot:
            symbolic_cot = symbolic_renderer.render_cot(ontology)
            nl_cot = nl_renderer.render_cot(ontology)

        # Get metadata
        task_types = [h["task_type"].name for h in hypotheses]
        depths = [h["depth"] for h in hypotheses]
        observations = ontology.get_observations()
        theories = ontology.get_visible_theories()

        # Build sample dict
        sample = {
            "id": sample_id,
            "config": {
                "hops": config.hops,
                "difficulty": config.difficulty.name,
                "task_types": task_types,
                "mix_hops": config.mix_hops,
                "recover_property": config.recover_property,
                "recover_membership": config.recover_membership,
                "recover_ontology": config.recover_ontology,
            },
            "symbolic": {
                "world_model": symbolic["theories"],
                "observations": symbolic["observations"],
                "hypotheses": symbolic["hypotheses"],
                "mapping": symbolic["mapping"]
            },
            "natural_language": {
                "theories": nl["theories"],
                "observations": nl["observations"],
                "hypotheses": nl["hypotheses"]
            },
            "metadata": {
                "depths": depths,
                "num_hypotheses": len(hypotheses),
                "num_observations": len(observations),
                "num_theories": len(theories),
                "seed": seed
            }
        }

        # Add CoT if requested
        if include_cot:
            sample["symbolic"]["cot"] = symbolic_cot
            sample["natural_language"]["cot"] = nl_cot

        return sample
    except Exception as e:
        print(f"Warning: Failed to generate sample {sample_id} (seed={seed}): {e}")
        return None


def generate_dataset(
    num_samples: int,
    hops: int,
    task_types: List[TaskType],
    difficulty: Difficulty,
    mix_hops: bool,
    seed: int,
    verbose: bool = True,
    include_cot: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate a dataset of isomorphic samples.

    Args:
        num_samples: Number of samples to generate
        hops: Tree depth / reasoning hops
        task_types: List of enabled task types
        difficulty: Difficulty level
        mix_hops: Whether to allow hiding at any level
        seed: Base random seed
        verbose: Print progress
        include_cot: Whether to include chain-of-thought reasoning

    Returns:
        List of sample dicts
    """
    random.seed(seed)

    config = OntologyConfig(
        hops=hops,
        recover_property=TaskType.INFER_PROPERTY in task_types,
        recover_membership=TaskType.INFER_MEMBERSHIP in task_types,
        recover_ontology=TaskType.INFER_SUBTYPE in task_types,
        difficulty=difficulty,
        mix_hops=mix_hops
    )

    samples = []
    attempts = 0
    max_attempts = num_samples * 10

    while len(samples) < num_samples and attempts < max_attempts:
        sample_seed = seed + attempts
        sample = generate_sample(
            config,
            sample_id=len(samples),
            seed=sample_seed,
            include_cot=include_cot
        )

        if sample is not None:
            samples.append(sample)
            if verbose and len(samples) % 100 == 0:
                print(f"  Generated {len(samples)}/{num_samples} samples...")

        attempts += 1

    if len(samples) < num_samples:
        print(f"Warning: Only generated {len(samples)}/{num_samples} samples after {attempts} attempts")

    return samples


def save_dataset(samples: List[Dict[str, Any]], output_path: str, format: str = "jsonl"):
    """Save dataset to file."""
    if format == "jsonl":
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
    else:  # json
        with open(output_path, 'w') as f:
            json.dump(samples, f, indent=2)


def print_sample(sample: Dict[str, Any], max_items: int = 3):
    """Print a sample for inspection."""
    print("\n" + "=" * 60)
    print(f"SAMPLE {sample['id']}")
    print("=" * 60)

    print(f"\n[CONFIG]")
    print(f"  Hops: {sample['config']['hops']}")
    print(f"  Difficulty: {sample['config']['difficulty']}")
    print(f"  Task types: {sample['config']['task_types']}")
    print(f"  Mix hops: {sample['config']['mix_hops']}")

    print(f"\n[SYMBOLIC FORMAT]")
    print("  World Model:")
    for t in sample["symbolic"]["world_model"][:max_items]:
        print(f"    {t}")
    if len(sample["symbolic"]["world_model"]) > max_items:
        print(f"    ... ({len(sample['symbolic']['world_model'])} total)")
    print("  Observations:")
    for o in sample["symbolic"]["observations"][:max_items]:
        print(f"    {o}")
    if len(sample["symbolic"]["observations"]) > max_items:
        print(f"    ... ({len(sample['symbolic']['observations'])} total)")
    print("  Hypotheses:")
    for h in sample["symbolic"]["hypotheses"]:
        print(f"    {h}")

    print(f"\n[NATURAL LANGUAGE FORMAT]")
    theories = sample["natural_language"]["theories"]
    print(f"  Theories: {theories[:200]}{'...' if len(theories) > 200 else ''}")
    obs = sample["natural_language"]["observations"]
    print(f"  Observations: {obs[:200]}{'...' if len(obs) > 200 else ''}")
    print(f"  Hypotheses: {sample['natural_language']['hypotheses']}")
    if "cot" in sample["natural_language"]:
        cot = sample["natural_language"]["cot"]
        print(f"  CoT: {cot[:300]}{'...' if len(cot) > 300 else ''}")

    print(f"\n[MAPPING (partial)]")
    concepts = sample["symbolic"]["mapping"]["concepts"]
    entities = sample["symbolic"]["mapping"]["entities"]
    print(f"  Concepts: {dict(list(concepts.items())[:3])}...")
    print(f"  Entities: {dict(list(entities.items())[:3])}...")

    print(f"\n[METADATA]")
    print(f"  Depths: {sample['metadata']['depths']}")
    print(f"  Num hypotheses: {sample['metadata']['num_hypotheses']}")
    print(f"  Num observations: {sample['metadata']['num_observations']}")
    print(f"  Num theories: {sample['metadata']['num_theories']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate INABHYD-complete isomorphic datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate inductive reasoning samples (infer property)
  python generate_isomorphic_dataset.py --task infer_property --hops 2 --samples 500

  # Generate all task types
  python generate_isomorphic_dataset.py --task all --hops 3 --difficulty medium

  # Generate with multiple hypotheses (harder)
  python generate_isomorphic_dataset.py --task all --hops 3 --difficulty hard --mix-hops
        """
    )

    parser.add_argument("--samples", "-n", type=int, default=500,
                        help="Number of samples to generate (default: 500)")
    parser.add_argument("--hops", type=int, default=2,
                        help="Number of reasoning hops / tree depth (default: 2)")
    parser.add_argument("--task", "-t", type=str, default="infer_property",
                        choices=["infer_property", "infer_membership",
                                 "infer_subtype", "all"],
                        help="Task type to generate (default: infer_property)")
    parser.add_argument("--difficulty", "-d", type=str, default="single",
                        choices=["single", "easy", "medium", "hard"],
                        help="Difficulty level (default: single)")
    parser.add_argument("--mix-hops", action="store_true",
                        help="Allow hiding axioms at any tree level")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", "-o", type=str,
                        default=None,
                        help="Output file path (auto-generated if not specified)")
    parser.add_argument("--format", type=str, default="jsonl",
                        choices=["jsonl", "json"],
                        help="Output format (default: jsonl)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--show-sample", action="store_true",
                        help="Show a sample after generation")
    parser.add_argument("--cot", action="store_true",
                        help="Include chain-of-thought reasoning in samples")

    args = parser.parse_args()

    # Parse task types
    if args.task == "all":
        task_types = [TaskType.INFER_PROPERTY, TaskType.INFER_MEMBERSHIP,
                      TaskType.INFER_SUBTYPE]
    elif args.task == "infer_property":
        task_types = [TaskType.INFER_PROPERTY]
    elif args.task == "infer_membership":
        task_types = [TaskType.INFER_MEMBERSHIP]
    elif args.task == "infer_subtype":
        task_types = [TaskType.INFER_SUBTYPE]
    else:
        task_types = [TaskType[args.task.upper()]]

    # Parse difficulty
    difficulty = Difficulty[args.difficulty.upper()]

    # Generate output filename if not specified
    if args.output is None:
        task_str = args.task.replace("_", "")
        args.output = f"isomorphic_{task_str}_{args.hops}hop_{args.difficulty}.{args.format}"

    # Print configuration
    if not args.quiet:
        print(f"Generating {args.samples} isomorphic samples...")
        print(f"  Hops: {args.hops}")
        print(f"  Tasks: {[t.name for t in task_types]}")
        print(f"  Difficulty: {difficulty.name}")
        print(f"  Mix hops: {args.mix_hops}")
        print(f"  Include CoT: {args.cot}")
        print(f"  Seed: {args.seed}")
        print(f"  Output: {args.output}")
        print()

    # Generate dataset
    dataset = generate_dataset(
        num_samples=args.samples,
        hops=args.hops,
        task_types=task_types,
        difficulty=difficulty,
        mix_hops=args.mix_hops,
        seed=args.seed,
        verbose=not args.quiet,
        include_cot=args.cot
    )

    # Save
    save_dataset(dataset, args.output, args.format)

    if not args.quiet:
        print(f"\nGenerated {len(dataset)} samples. Saved to {args.output}")

    # Show sample if requested
    if args.show_sample and dataset:
        print_sample(dataset[0])

    # Print summary statistics
    if not args.quiet and dataset:
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)

        # Task type distribution
        from collections import Counter
        task_counts = Counter()
        for sample in dataset:
            for tt in sample["config"]["task_types"]:
                task_counts[tt] += 1
        print(f"\nTask type distribution:")
        for tt, count in sorted(task_counts.items()):
            print(f"  {tt}: {count}")

        # Depth distribution
        depth_counts = Counter()
        for sample in dataset:
            for d in sample["metadata"]["depths"]:
                depth_counts[d] += 1
        print(f"\nHypothesis depth distribution:")
        for d, count in sorted(depth_counts.items()):
            print(f"  Depth {d}: {count}")

        # Hypothesis counts
        hyp_counts = Counter(sample["metadata"]["num_hypotheses"] for sample in dataset)
        print(f"\nHypotheses per sample:")
        for n, count in sorted(hyp_counts.items()):
            print(f"  {n} hypothesis{'es' if n > 1 else ''}: {count} samples")


if __name__ == "__main__":
    main()
