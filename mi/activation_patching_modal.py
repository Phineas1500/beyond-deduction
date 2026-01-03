"""
Modal deployment for Activation Patching experiments.

Runs causal intervention experiments on H100 GPU:
1. Layer 8 Residual Stream Patching (multiple layers)
2. Layer 12 Attention Output Patching
3. Bidirectional Patching

Usage:
    modal run mi/activation_patching_modal.py
    modal run mi/activation_patching_modal.py --n-pairs 100
    modal run --detach mi/activation_patching_modal.py  # Background
"""

import modal
import os
from pathlib import Path

# Modal app
app = modal.App("activation-patching-gemma2")

# Project path
PROJECT_DIR = Path(__file__).parent.parent

# Image with dependencies (include local dirs in image)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "numpy==1.26.4",
        "scipy==1.14.1",
        "matplotlib==3.9.2",  # Required by attention_analysis (imported via mi.__init__)
        "seaborn==0.13.2",
    )
    .add_local_dir(PROJECT_DIR / "mi", remote_path="/root/beyond-deduction/mi")
    .add_local_dir(PROJECT_DIR / "benchmark", remote_path="/root/beyond-deduction/benchmark")
)

# Volumes
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("attention-results", create_if_missing=True)

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
def run_patching(
    pairs_path: str = "benchmark/matched_pairs_set1_pure.pkl",
    results_path: str = "benchmark/factorial_results/factorial_gemma29b.pkl",
    n_pairs: int = 50,
    residual_layers: str = "8,12,16,20",
    attention_layer: int = 12,
):
    """Run activation patching experiments on Modal."""
    import sys
    import torch
    import pickle

    # Setup paths
    sys.path.insert(0, "/root/beyond-deduction")
    sys.path.insert(0, "/root/beyond-deduction/benchmark")

    # Parse layers
    layers = [int(x) for x in residual_layers.split(",")]

    print("="*60)
    print("ACTIVATION PATCHING EXPERIMENTS")
    print("="*60)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Pairs: {pairs_path}")
    print(f"Results: {results_path}")
    print(f"N pairs: {n_pairs}")
    print(f"Residual layers: {layers}")
    print(f"Attention layer: {attention_layer}")
    print("="*60)

    # Import patching module
    from mi.activation_patching import (
        PatchingConfig,
        run_all_experiments,
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

    # Create config
    config = PatchingConfig(
        residual_layers=layers,
        attention_layer=attention_layer,
        n_pairs=n_pairs,
    )

    # Run experiments
    results = run_all_experiments(
        model=model,
        tokenizer=tokenizer,
        pairs_path=f"/root/beyond-deduction/{pairs_path}",
        results_path=f"/root/beyond-deduction/{results_path}",
        config=config,
        output_dir="/root/results",
    )

    # Return summary
    summary = {
        "n_pairs": n_pairs,
        "residual_layers": layers,
        "attention_layer": attention_layer,
    }

    if "residual_patching" in results:
        summary["residual_patching"] = results["residual_patching"]["summary"]

    if "attention_patching" in results:
        summary["attention_patching"] = results["attention_patching"]["summary"]

    if "bidirectional_patching" in results:
        s2f = results["bidirectional_patching"]["success_to_failure"]
        f2s = results["bidirectional_patching"]["failure_to_success"]
        summary["bidirectional"] = {
            "success_to_failure_rate": s2f["flip_rate"],
            "failure_to_success_rate": f2s["flip_rate"],
        }

    # Save summary
    results_vol.commit()

    return summary


@app.function(
    image=image,
    volumes={"/root/results": results_vol},
)
def download_results():
    """Download results from Modal volume."""
    import pickle
    from pathlib import Path

    results_path = Path("/root/results/activation_patching_results.pkl")

    if results_path.exists():
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        return results
    else:
        return {"error": "Results file not found"}


@app.local_entrypoint()
def main(
    pairs: str = "benchmark/matched_pairs_set1_pure.pkl",
    results: str = "benchmark/factorial_results/factorial_gemma29b.pkl",
    n_pairs: int = 50,
    layers: str = "8,12,16,20",
    attn_layer: int = 12,
):
    """Run activation patching experiments."""
    import json

    print("="*60)
    print("ACTIVATION PATCHING PIPELINE")
    print("="*60)
    print(f"Pairs: {pairs}")
    print(f"Results: {results}")
    print(f"N pairs: {n_pairs}")
    print(f"Residual layers: {layers}")
    print(f"Attention layer: {attn_layer}")
    print("="*60)

    # Run patching
    summary = run_patching.remote(
        pairs_path=pairs,
        results_path=results,
        n_pairs=n_pairs,
        residual_layers=layers,
        attention_layer=attn_layer,
    )

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(json.dumps(summary, indent=2))

    # Print interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if "residual_patching" in summary:
        layer8_data = summary["residual_patching"].get("8", {})
        layer8_rate = layer8_data.get("flip_rate", 0)

        if layer8_rate > 0.3:
            print(f"""
✓ CAUSAL EVIDENCE FOUND!

Layer 8 flip rate: {layer8_rate*100:.1f}%

This means patching SUCCESS activations into FAILURE examples
causes the model to output CHILD instead of PARENT in {layer8_rate*100:.1f}%
of cases.

INTERPRETATION:
- Layer 8 residual stream CAUSALLY controls the output
- The probing result (94% accuracy) reflects genuine causal structure
- The decision IS made by layer 8, not just "readable" there

This directly addresses the reviewer's main concern about causal evidence.
""")
        elif layer8_rate > 0.1:
            print(f"""
⚠ PARTIAL CAUSAL EFFECT

Layer 8 flip rate: {layer8_rate*100:.1f}%

INTERPRETATION:
- Layer 8 has some causal influence
- But the effect is not complete - other components also matter
- The decision is distributed across multiple layers
""")
        else:
            print(f"""
✗ NO CAUSAL EFFECT

Layer 8 flip rate: {layer8_rate*100:.1f}%

INTERPRETATION:
- Layer 8 is correlational, not causal
- Probing detected information that exists but isn't used
- "The speedometer, not the steering wheel"
- Decision is made elsewhere despite being readable at layer 8
""")

    # Download results to local
    print("\nDownloading results...")
    full_results = download_results.remote()

    if "error" not in full_results:
        import pickle
        local_path = PROJECT_DIR / "mi" / "patching_results"
        local_path.mkdir(exist_ok=True)

        with open(local_path / "activation_patching_results.pkl", 'wb') as f:
            pickle.dump(full_results, f)
        print(f"Results saved to: {local_path / 'activation_patching_results.pkl'}")
    else:
        print(f"Warning: {full_results['error']}")
