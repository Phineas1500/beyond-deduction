"""
Attention Analysis Only - Uses pre-computed factorial results.

This script runs ONLY the attention analysis step, using existing
pairs and factorial results files that are mounted from local disk.

Usage:
    # Use original 200-pair data
    modal run mi/attention_only_modal.py

    # Use custom data files
    modal run mi/attention_only_modal.py \
        --pairs benchmark/matched_pairs_set1_1000.pkl \
        --results benchmark/factorial_results/factorial_gemma29b_1000.pkl

Author: Ram (CS 577 Project)
"""

import modal
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Image with dependencies
pipeline_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "scipy==1.14.1",
        "numpy==1.26.4",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
    )
    .add_local_dir(PROJECT_ROOT / "mi", remote_path="/root/beyond-deduction/mi")
    .add_local_dir(PROJECT_ROOT / "benchmark", remote_path="/root/beyond-deduction/benchmark")
)

app = modal.App("attention-only")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("attention-results", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MINUTES = 60


@app.function(
    image=pipeline_image,
    gpu="H100",
    timeout=180 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/results": results_vol,
    },
    secrets=[hf_secret],
)
def run_attention_analysis(
    pairs_path: str,
    results_path: str,
    layers: str = "4,6,8,10,12,14,16,18,20",
    max_per_category: int = None,
) -> dict:
    """Run attention analysis on pre-computed factorial results."""
    import sys
    import os
    import pickle
    import torch

    sys.path.insert(0, "/root/beyond-deduction")
    sys.path.insert(0, "/root/beyond-deduction/benchmark")
    os.chdir("/root/beyond-deduction")

    print("=" * 60)
    print("ATTENTION ANALYSIS (using pre-computed results)")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Pairs: {pairs_path}")
    print(f"Results: {results_path}")

    # Load data
    with open(pairs_path, "rb") as f:
        pairs = pickle.load(f)
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    print(f"Loaded {len(pairs)} pairs")

    # Verify Conservation Law
    h1_results = results['set1']['h1_results']
    h2_results = results['set1']['h2_results']

    h1_acc = sum(r['strong'] for r in h1_results) / len(h1_results)
    h2_acc = sum(r['strong'] for r in h2_results) / len(h2_results)

    print(f"\nConservation Law Check:")
    print(f"  H1 accuracy: {h1_acc:.1%}")
    print(f"  H2 accuracy: {h2_acc:.1%}")
    print(f"  Sum: {h1_acc + h2_acc:.1%}")

    if abs(h1_acc + h2_acc - 1.0) > 0.15:
        print("  WARNING: Conservation Law not satisfied - labels may be incorrect!")

    # Parse layers
    focus_layers = [int(l) for l in layers.split(",")]
    print(f"\nFocus layers: {focus_layers}")

    # Create probing examples
    from dataclasses import dataclass
    from typing import List, Optional

    @dataclass
    class ProbingExample:
        idx: int
        h1_or_h2: str
        seed: int
        prompt: str
        theories_nl: str
        observations_nl: str
        child_concept: str
        root_concept: str
        entities: List[str]
        property_name: str
        property_family: str
        is_negated: bool
        ground_truth: str
        depth: int
        model_correct: Optional[bool] = None
        model_response: Optional[str] = None

    system_prompt = """You are a helpful assitant that performs abduction and induction reasoning.
Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
Only output final hypotheses."""

    def format_prompt(theories_nl, observations_nl):
        user_prompt = f"Q: {theories_nl} We observe that: {observations_nl} Please come up with hypothesis to explain observations."
        return f"{system_prompt}\n\n{user_prompt}"

    examples = []

    for idx, (h1, h2) in enumerate(pairs):
        for example, h_type, result_list in [(h1, "h1", h1_results), (h2, "h2", h2_results)]:
            prompt = format_prompt(example['theories_nl'], example['observations_nl'])
            result = result_list[idx] if idx < len(result_list) else None

            prop = example['property']
            if hasattr(prop, 'name'):
                property_name = prop.name
                property_family = prop.family if hasattr(prop, 'family') else ""
            else:
                property_name = str(prop)
                property_family = ""

            probing_ex = ProbingExample(
                idx=idx,
                h1_or_h2=h_type,
                seed=example['seed'],
                prompt=prompt,
                theories_nl=example['theories_nl'],
                observations_nl=example['observations_nl'],
                child_concept=example['child_concept'],
                root_concept=example['root_concept'],
                entities=example['entities'],
                property_name=property_name,
                property_family=property_family,
                is_negated=example.get('is_negated', False),
                ground_truth=example['gt_hypothesis_nl'],
                depth=example['depth'],
                model_correct=bool(result['strong']) if result else None,
                model_response=result.get('reply', '') if result else None
            )
            examples.append(probing_ex)

    print(f"Created {len(examples)} probing examples")

    # Stratify
    categories = {
        'h1_fail_h2_success': [],
        'h1_success_h2_fail': [],
        'h1_success_h2_success': [],
        'h1_fail_h2_fail': [],
    }

    for i in range(len(h1_results)):
        h1_success = h1_results[i]['strong'] == 1
        h2_success = h2_results[i]['strong'] == 1

        if not h1_success and h2_success:
            categories['h1_fail_h2_success'].append(i)
        elif h1_success and not h2_success:
            categories['h1_success_h2_fail'].append(i)
        elif h1_success and h2_success:
            categories['h1_success_h2_success'].append(i)
        else:
            categories['h1_fail_h2_fail'].append(i)

    print("\nBehavioral Stratification:")
    for cat, indices in categories.items():
        print(f"  {cat}: {len(indices)}")

    # Import and run attention analysis
    from mi.model_wrapper import Gemma2Wrapper
    from mi.attention_analysis import AttentionAnalysisConfig, run_attention_analysis as run_analysis

    config = AttentionAnalysisConfig(
        focus_layers=focus_layers,
        output_dir="/root/results",
    )

    print("\nLoading Gemma 2 9B IT...")
    model = Gemma2Wrapper(use_4bit=False)

    print("\nRunning attention analysis...")
    analysis_results = run_analysis(
        model=model,
        examples=examples,
        categories=categories,
        config=config,
        max_per_category=max_per_category,
        verbose=True,
    )

    results_vol.commit()

    # Build summary
    summary = {
        "n_pairs": len(pairs),
        "h1_accuracy": h1_acc,
        "h2_accuracy": h2_acc,
        "conservation_sum": h1_acc + h2_acc,
        "n_h1_failures": len(categories['h1_fail_h2_success']) + len(categories['h1_fail_h2_fail']),
        "n_gold_pairs": len(categories['h1_fail_h2_success']),
        "n_analyzed_failures": analysis_results.get("n_failures", 0),
        "n_analyzed_successes": analysis_results.get("n_successes", 0),
    }

    if analysis_results.get("comparison", {}).get("aggregate"):
        agg = analysis_results["comparison"]["aggregate"]
        summary["aggregate"] = {
            "failure_mean": float(agg["failure_mean"]),
            "success_mean": float(agg["success_mean"]),
            "difference": float(agg["difference"]),
            "p_value": float(agg["p_value"]),
        }

    if analysis_results.get("comparison", {}).get("by_layer"):
        summary["by_layer"] = {}
        for layer, data in analysis_results["comparison"]["by_layer"].items():
            summary["by_layer"][layer] = {
                "difference": float(data["difference"]),
                "p_value": float(data["p_value"]),
                "cohens_d": float(data["cohens_d"]),
            }

    if analysis_results.get("region_comparison"):
        summary["region_comparison"] = {}
        for region, data in analysis_results["region_comparison"].items():
            summary["region_comparison"][region] = {
                "failure_mean": float(data["failure_mean"]),
                "success_mean": float(data["success_mean"]),
                "difference": float(data["difference"]),
                "p_value": float(data["p_value"]),
            }

    return summary


@app.function(
    image=pipeline_image,
    volumes={"/root/results": results_vol},
)
def download_results() -> dict:
    """Download all result files."""
    import os

    files = {}
    results_dir = "/root/results"
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            filepath = os.path.join(results_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, "rb") as f:
                    files[filename] = f.read()
    return files


@app.local_entrypoint()
def main(
    pairs: str = "benchmark/matched_pairs_set1_pure.pkl",
    results: str = "benchmark/factorial_results/factorial_gemma29b.pkl",
    layers: str = "4,6,8,10,12,14,16,18,20",
    max_attention: int = None,
    download: bool = True,
):
    """
    Run attention analysis on pre-computed factorial results.

    Args:
        pairs: Path to matched pairs pickle (relative to project root)
        results: Path to factorial results pickle (relative to project root)
        layers: Comma-separated layers to analyze
        max_attention: Max examples per category for attention analysis
        download: Download results after completion
    """
    import json

    # Convert to absolute paths for Modal
    pairs_path = f"/root/beyond-deduction/{pairs}"
    results_path = f"/root/beyond-deduction/{results}"

    print("=" * 60)
    print("ATTENTION ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Pairs: {pairs}")
    print(f"Results: {results}")
    print(f"Layers: {layers}")

    # Run analysis
    summary = run_attention_analysis.remote(
        pairs_path=pairs_path,
        results_path=results_path,
        layers=layers,
        max_per_category=max_attention,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    # Interpret
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"Conservation Law: H1={summary['h1_accuracy']:.1%} + H2={summary['h2_accuracy']:.1%} = {summary['conservation_sum']:.1%}")

    if abs(summary['conservation_sum'] - 1.0) < 0.1:
        print("  ✓ Conservation Law holds - labels are correct")
    else:
        print("  ✗ Conservation Law violated - check evaluation")

    if "aggregate" in summary:
        agg = summary["aggregate"]
        print(f"\nSubsumption attention: diff={agg['difference']:+.4f}, p={agg['p_value']:.4f}")

    if "region_comparison" in summary:
        print("\nRegion comparison:")
        for region, data in summary["region_comparison"].items():
            sig = "***" if data["p_value"] < 0.001 else "**" if data["p_value"] < 0.01 else "*" if data["p_value"] < 0.05 else ""
            print(f"  {region}: diff={data['difference']:+.4f}, p={data['p_value']:.4f} {sig}")

    # Download
    if download:
        print("\nDownloading results...")
        files = download_results.remote()

        output_dir = Path(__file__).parent / "attention_results"
        output_dir.mkdir(exist_ok=True)

        for filename, data in files.items():
            filepath = output_dir / filename
            with open(filepath, "wb") as f:
                f.write(data)
            print(f"  Saved: {filepath}")


if __name__ == "__main__":
    print("Run with: modal run mi/attention_only_modal.py")
    print("Custom data: modal run mi/attention_only_modal.py --pairs PATH --results PATH")
