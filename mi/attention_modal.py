"""
Modal deployment for Attention Analysis on H100.

Runs the attention analysis pipeline on cloud GPU and returns results.

Usage:
    # Quick test (5 examples per category)
    modal run mi/attention_modal.py --test

    # Full analysis (all examples)
    modal run mi/attention_modal.py

    # Custom layers
    modal run mi/attention_modal.py --layers 4,6,8,10,12

Author: Ram (CS 577 Project)
"""

import modal
from pathlib import Path

# Image with PyTorch and transformers
attention_image = (
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
)

# Modal app
app = modal.App("attention-analysis")

# HuggingFace cache volume (reuse existing)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Results volume to persist outputs
results_vol = modal.Volume.from_name("attention-results", create_if_missing=True)

# HuggingFace secret for Gemma 2 access
hf_secret = modal.Secret.from_name("huggingface-secret")

# Mount local code
local_code = modal.Mount.from_local_dir(
    Path(__file__).parent.parent,
    remote_path="/root/beyond-deduction",
    condition=lambda path: (
        # Include MI code and benchmark data
        "mi/" in path or
        "benchmark/" in path or
        path.endswith(".py") or
        path.endswith(".pkl")
    ) and not (
        # Exclude large/unnecessary files
        "__pycache__" in path or
        ".git" in path or
        "archive/" in path or
        ".pt" in path
    ),
)

MINUTES = 60


@app.function(
    image=attention_image,
    gpu="H100",
    timeout=60 * MINUTES,
    mounts=[local_code],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/results": results_vol,
    },
    secrets=[hf_secret],
)
def run_attention_analysis(
    test_mode: bool = False,
    max_per_category: int = None,
    layers: str = "4,6,8,10,12,14,16,18,20",
) -> dict:
    """
    Run attention analysis on H100 GPU.

    Args:
        test_mode: If True, run with 5 examples per category
        max_per_category: Override max examples per category
        layers: Comma-separated list of layers to analyze

    Returns:
        Dict with analysis results summary
    """
    import sys
    import os
    import torch
    import pickle

    # Add code to path
    sys.path.insert(0, "/root/beyond-deduction")
    os.chdir("/root/beyond-deduction")

    print("=" * 60)
    print("ATTENTION ANALYSIS ON H100")
    print("=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Parse layers
    focus_layers = [int(l) for l in layers.split(",")]
    print(f"Focus layers: {focus_layers}")

    # Set max examples
    if test_mode:
        max_examples = 5
        print("TEST MODE: 5 examples per category")
    elif max_per_category:
        max_examples = max_per_category
    else:
        max_examples = None  # All examples
        print("FULL MODE: All examples")

    # Import MI modules
    from mi.data_loader import create_probing_dataset, stratify_gemma_results
    from mi.model_wrapper import Gemma2Wrapper
    from mi.attention_analysis import AttentionAnalysisConfig, run_attention_analysis as run_analysis

    # Load data
    print("\nLoading data...")
    examples = create_probing_dataset()
    categories = stratify_gemma_results()

    # Configure analysis
    config = AttentionAnalysisConfig(
        focus_layers=focus_layers,
        output_dir="/root/results",
    )

    # Load model
    print("\nLoading Gemma 2 9B IT...")
    model = Gemma2Wrapper(use_4bit=False)  # Full precision on H100

    # Run analysis
    print("\nRunning attention analysis...")
    results = run_analysis(
        model=model,
        examples=examples,
        categories=categories,
        config=config,
        max_per_category=max_examples,
        verbose=True,
    )

    # Commit results to volume
    results_vol.commit()

    # Prepare summary for return
    summary = {
        "n_failures": results.get("n_failures", 0),
        "n_successes": results.get("n_successes", 0),
        "focus_layers": focus_layers,
    }

    if results.get("comparison", {}).get("aggregate"):
        agg = results["comparison"]["aggregate"]
        summary["aggregate"] = {
            "failure_mean": float(agg["failure_mean"]),
            "success_mean": float(agg["success_mean"]),
            "difference": float(agg["difference"]),
            "p_value": float(agg["p_value"]),
        }

    if results.get("comparison", {}).get("by_layer"):
        summary["by_layer"] = {}
        for layer, data in results["comparison"]["by_layer"].items():
            summary["by_layer"][layer] = {
                "difference": float(data["difference"]),
                "p_value": float(data["p_value"]),
                "cohens_d": float(data["cohens_d"]),
            }

    if results.get("layer_kv_analysis", {}).get("top_differences"):
        summary["top_components"] = [
            {
                "layer": item["layer"],
                "kv_group": item["kv_group"],
                "difference": float(item["difference"]),
                "p_value": float(item["p_value"]),
            }
            for item in results["layer_kv_analysis"]["top_differences"][:5]
        ]

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return summary


@app.function(
    image=attention_image,
    volumes={"/root/results": results_vol},
)
def download_results() -> dict:
    """Download all result files (pickle and PNGs)."""
    import os

    results_dir = "/root/results"
    files = {}

    for filename in os.listdir(results_dir):
        filepath = os.path.join(results_dir, filename)
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                files[filename] = f.read()

    return files


@app.function(
    image=attention_image,
    volumes={"/root/results": results_vol},
)
def list_results() -> list:
    """List files in results volume."""
    import os

    results_dir = "/root/results"
    if os.path.exists(results_dir):
        return os.listdir(results_dir)
    return []


@app.local_entrypoint()
def main(
    test: bool = False,
    max: int = None,
    layers: str = "4,6,8,10,12,14,16,18,20",
    download: bool = True,
):
    """
    Run attention analysis on H100 and optionally download results.

    Args:
        test: Run in test mode (5 examples per category)
        max: Maximum examples per category
        layers: Comma-separated list of layers to analyze
        download: Download results after completion
    """
    import json

    print("Starting attention analysis on Modal H100...")

    # Run analysis
    summary = run_attention_analysis.remote(
        test_mode=test,
        max_per_category=max,
        layers=layers,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    # Interpret results
    if "aggregate" in summary:
        agg = summary["aggregate"]
        print("\n" + "=" * 60)
        print("INTERPRETATION")
        print("=" * 60)
        print(f"Subsumption attention difference: {agg['difference']:+.4f}")
        print(f"Statistical significance: p={agg['p_value']:.4f}")

        if agg["p_value"] < 0.05:
            if agg["difference"] > 0:
                print("\n>>> H3 SUPPORTED: Failures attend MORE to subsumption")
                print("    This suggests attention to 'X is a Y' triggers over-generalization")
            else:
                print("\n>>> H3 REJECTED: Successes attend MORE to subsumption")
                print("    Attention to subsumption may PREVENT over-generalization")
        else:
            print("\n>>> H3 INCONCLUSIVE: No significant difference")
            print("    Consider: SAE features, later layers, or other mechanisms")

    # Download results
    if download:
        print("\nDownloading results...")
        files = download_results.remote()

        # Save locally
        output_dir = Path(__file__).parent / "attention_results"
        output_dir.mkdir(exist_ok=True)

        for filename, data in files.items():
            filepath = output_dir / filename
            with open(filepath, "wb") as f:
                f.write(data)
            print(f"  Saved: {filepath}")

        print(f"\nAll results saved to: {output_dir}/")
        print("Files downloaded:")
        for filename in files.keys():
            print(f"  - {filename}")


if __name__ == "__main__":
    print("Run with: modal run mi/attention_modal.py")
    print("Test mode: modal run mi/attention_modal.py --test")
    print("Full mode: modal run mi/attention_modal.py")
