"""
Modal deployment for Logit Lens analysis.

Runs logit lens on H100 GPU to track how P(child) vs P(parent) evolves
across layers in Gemma 2 9B IT.

Usage:
    modal run mi/logit_lens_modal.py
    modal run mi/logit_lens_modal.py --n-per-group 50
    modal run --detach mi/logit_lens_modal.py  # Background
"""

import modal
import os
from pathlib import Path

# Modal app
app = modal.App("logit-lens-gemma2")

# Project path
PROJECT_DIR = Path(__file__).parent.parent

# Image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "numpy==1.26.4",
        "scipy==1.14.1",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
    )
    .add_local_dir(PROJECT_DIR / "mi", remote_path="/root/beyond-deduction/mi")
    .add_local_dir(PROJECT_DIR / "benchmark", remote_path="/root/beyond-deduction/benchmark")
)

# Volumes
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("logit-lens-results", create_if_missing=True)

# Secrets
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,  # 1 hour
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/results": results_vol,
    },
    secrets=[hf_secret],
)
def run_logit_lens(
    pairs_path: str = "benchmark/matched_pairs_set1_pure.pkl",
    results_path: str = "benchmark/factorial_results/factorial_gemma29b.pkl",
    n_per_group: int = 25,
    layers: str = "0,4,8,12,16,20,24,28,32,36,40",
):
    """Run logit lens analysis on Modal."""
    import sys
    import torch
    import pickle

    # Setup paths
    sys.path.insert(0, "/root/beyond-deduction")
    sys.path.insert(0, "/root/beyond-deduction/benchmark")

    # Parse layers
    target_layers = [int(x) for x in layers.split(",")]

    print("=" * 60)
    print("LOGIT LENS ANALYSIS")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Pairs: {pairs_path}")
    print(f"Results: {results_path}")
    print(f"N per group: {n_per_group}")
    print(f"Target layers: {target_layers}")
    print("=" * 60)

    # Import logit lens module
    from mi.logit_lens import (
        LogitLensConfig,
        run_logit_lens_analysis,
        create_visualizations,
        print_summary,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model
    print("\nLoading Gemma 2 9B IT...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Required for Gemma 2
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    print("Model loaded successfully")

    # Load data
    print("\nLoading data...")
    with open(f"/root/beyond-deduction/{pairs_path}", "rb") as f:
        pairs = pickle.load(f)

    with open(f"/root/beyond-deduction/{results_path}", "rb") as f:
        factorial_results = pickle.load(f)

    h1_results = factorial_results['set1']['h1_results']
    print(f"Loaded {len(pairs)} pairs")

    # Conservation Law check
    h1_acc = sum(1 for r in h1_results if r['strong'] == 1) / len(h1_results)
    h2_results = factorial_results['set1']['h2_results']
    h2_acc = sum(1 for r in h2_results if r['strong'] == 1) / len(h2_results)

    print(f"\nConservation Law Check:")
    print(f"  H1 accuracy: {h1_acc*100:.1f}%")
    print(f"  H2 accuracy: {h2_acc*100:.1f}%")
    print(f"  Sum: {(h1_acc + h2_acc)*100:.1f}%")

    # Create config
    config = LogitLensConfig(
        target_layers=target_layers,
        n_per_group=n_per_group,
    )

    # Run analysis
    print("\nRunning logit lens analysis...")
    results = run_logit_lens_analysis(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        h1_results=h1_results,
        config=config,
        verbose=True,
    )

    # Print summary
    print_summary(results)

    # Create visualizations
    print("\nCreating visualizations...")
    viz_paths = create_visualizations(
        results,
        output_dir="/root/results",
        highlight_layers=[8, 12],
    )

    # Save full results
    with open("/root/results/logit_lens_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Results saved to /root/results/logit_lens_results.pkl")

    results_vol.commit()

    # Build summary for return
    stats = results["statistics"]
    summary = {
        "n_success": stats["n_success"],
        "n_failure": stats["n_failure"],
        "target_layers": target_layers,
        "conservation_law": {
            "h1_accuracy": h1_acc,
            "h2_accuracy": h2_acc,
            "sum": h1_acc + h2_acc,
        },
        "decision_emergence": stats.get("decision_emergence", {}),
        "max_rank_gap": stats.get("max_rank_gap", {}),
        # Rank-based metrics (primary for rare tokens)
        "layer_8_rank_gaps": {
            "success": stats["by_layer"].get(8, {}).get("success", {}).get("rank_gap_mean", 0),
            "failure": stats["by_layer"].get(8, {}).get("failure", {}).get("rank_gap_mean", 0),
        },
        "layer_12_rank_gaps": {
            "success": stats["by_layer"].get(12, {}).get("success", {}).get("rank_gap_mean", 0),
            "failure": stats["by_layer"].get(12, {}).get("failure", {}).get("rank_gap_mean", 0),
        },
        "layer_40_rank_gaps": {
            "success": stats["by_layer"].get(40, {}).get("success", {}).get("rank_gap_mean", 0),
            "failure": stats["by_layer"].get(40, {}).get("failure", {}).get("rank_gap_mean", 0),
        },
    }

    return summary


@app.function(
    image=image,
    volumes={"/root/results": results_vol},
)
def download_results():
    """Download results from Modal volume."""
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
    n_per_group: int = 25,
    layers: str = "0,4,8,12,16,20,24,28,32,36,40",
):
    """Run logit lens analysis."""
    import json

    print("=" * 60)
    print("LOGIT LENS PIPELINE")
    print("=" * 60)
    print(f"Pairs: {pairs}")
    print(f"Results: {results}")
    print(f"N per group: {n_per_group}")
    print(f"Layers: {layers}")
    print("=" * 60)

    # Run analysis
    summary = run_logit_lens.remote(
        pairs_path=pairs,
        results_path=results,
        n_per_group=n_per_group,
        layers=layers,
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    # Print interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    layer8_success_rank_gap = summary.get("layer_8_rank_gaps", {}).get("success", 0)
    layer8_failure_rank_gap = summary.get("layer_8_rank_gaps", {}).get("failure", 0)
    layer40_success_rank_gap = summary.get("layer_40_rank_gaps", {}).get("success", 0)
    layer40_failure_rank_gap = summary.get("layer_40_rank_gaps", {}).get("failure", 0)

    print(f"""
RANK-BASED ANALYSIS (lower rank = more likely)

Layer 8 rank gaps (parent_rank - child_rank):
  Successes: {layer8_success_rank_gap:+.0f} {'(child preferred)' if layer8_success_rank_gap > 0 else '(parent preferred)'}
  Failures:  {layer8_failure_rank_gap:+.0f} {'(child preferred)' if layer8_failure_rank_gap > 0 else '(parent preferred)'}

Layer 40 rank gaps (final layer):
  Successes: {layer40_success_rank_gap:+.0f} {'(child preferred)' if layer40_success_rank_gap > 0 else '(parent preferred)'}
  Failures:  {layer40_failure_rank_gap:+.0f} {'(child preferred)' if layer40_failure_rank_gap > 0 else '(parent preferred)'}
""")

    # Both groups often show child preference at final layer
    # because we're measuring at input end, not at generation time
    if layer40_success_rank_gap > 0 and layer40_failure_rank_gap > 0:
        print("""
KEY INSIGHT: DECISION IS NOT VISIBLE AT INPUT END

Both successes AND failures prefer child token at input's last position.
This is expected because:
1. We measure at the last position of INPUT, not during generation
2. The first output token is "Each"/"All", not the concept itself
3. The concept decision happens DURING generation, not at input encoding

This SUPPORTS the distributed nature of the decision:
- The input encoding doesn't directly encode "output child" vs "output parent"
- The decision emerges through the generation process
- Consistent with activation patching showing no single causal locus
""")
    elif layer40_success_rank_gap > layer40_failure_rank_gap:
        print("""
SUCCESS/FAILURE DIVERGENCE

Successes show stronger child preference than failures at layer 40.
The gap between groups may encode information used during generation.
""")
    else:
        print("""
FURTHER INVESTIGATION NEEDED

The rank gaps show an unexpected pattern.
Consider analyzing at different positions or during generation.
""")

    # Download results
    print("\nDownloading results...")
    files = download_results.remote()

    local_path = PROJECT_DIR / "mi" / "logit_lens_results"
    local_path.mkdir(exist_ok=True)

    for filename, data in files.items():
        filepath = local_path / filename
        with open(filepath, "wb") as f:
            f.write(data)
        print(f"  Saved: {filepath}")

    print(f"\nAll results saved to {local_path}")
