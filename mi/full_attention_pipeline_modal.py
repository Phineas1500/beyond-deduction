"""
Full Attention Analysis Pipeline on Modal H100.

This script does everything:
1. Generates a large dataset (1000 pairs)
2. Runs Gemma 2 9B inference to get model outputs
3. Runs attention analysis to test H3

Usage:
    # Full pipeline (1000 pairs, ~90 min total)
    modal run mi/full_attention_pipeline_modal.py

    # Quick test (100 pairs, ~15 min)
    modal run mi/full_attention_pipeline_modal.py --n-pairs 100

    # Just generate data (no Modal GPU needed)
    modal run mi/full_attention_pipeline_modal.py --generate-only

Author: Ram (CS 577 Project)
"""

import modal
from pathlib import Path

# Get project root for adding local files
PROJECT_ROOT = Path(__file__).parent.parent

# Image with all dependencies + local code
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
        "aiohttp==3.11.10",
    )
    # Add local code to image
    .add_local_dir(PROJECT_ROOT / "mi", remote_path="/root/beyond-deduction/mi")
    .add_local_dir(PROJECT_ROOT / "benchmark", remote_path="/root/beyond-deduction/benchmark")
)

# Modal app
app = modal.App("attention-pipeline")

# Volumes
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("attention-results", create_if_missing=True)
data_vol = modal.Volume.from_name("attention-data", create_if_missing=True)

# Secrets
hf_secret = modal.Secret.from_name("huggingface-secret")

MINUTES = 300


# =============================================================================
# STEP 1: Generate Large Dataset
# =============================================================================

@app.function(
    image=pipeline_image,
    volumes={"/root/data": data_vol},
    timeout=10 * MINUTES,
)
def generate_large_dataset(n_pairs: int = 1000, base_seed: int = 42) -> str:
    """Generate a large dataset of matched pairs."""
    import sys
    import os
    import pickle

    sys.path.insert(0, "/root/beyond-deduction/benchmark")
    os.chdir("/root/beyond-deduction/benchmark")

    from generate_matched_pairs import generate_matched_pairs

    print(f"Generating {n_pairs} matched pairs...")
    pairs = generate_matched_pairs(n_pairs, base_seed, include_negated=True)

    # Save to volume
    output_path = f"/root/data/matched_pairs_large_{n_pairs}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(pairs, f)

    print(f"Saved {len(pairs)} pairs to {output_path}")
    data_vol.commit()

    return output_path


# =============================================================================
# STEP 2: Run Factorial Experiment (Model Inference)
# =============================================================================

@app.function(
    image=pipeline_image,
    gpu="H100",
    timeout=120 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/data": data_vol,
    },
    secrets=[hf_secret],
)
def run_factorial_inference(pairs_path: str, batch_size: int = 8) -> str:
    """Run Gemma 2 9B inference on all pairs to get model outputs."""
    import sys
    import os
    import pickle
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Add paths BEFORE loading pickle (needed for morphology module)
    sys.path.insert(0, "/root/beyond-deduction")
    sys.path.insert(0, "/root/beyond-deduction/benchmark")
    os.chdir("/root/beyond-deduction")

    print("=" * 60)
    print("STEP 2: FACTORIAL INFERENCE")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load pairs (requires morphology module for Prop objects)
    with open(pairs_path, "rb") as f:
        pairs = pickle.load(f)
    print(f"Loaded {len(pairs)} pairs")

    # Load model
    print("\nLoading Gemma 2 9B IT...")
    model_id = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    # System prompt (from run_factorial_experiment.py)
    system_prompt = """You are a helpful assitant that performs abduction and induction reasoning.
Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
Only output final hypotheses."""

    def format_prompt(theories_nl, observations_nl):
        user_prompt = f"Q: {theories_nl} We observe that: {observations_nl} Please come up with hypothesis to explain observations."
        # Use chat template for Gemma 2 IT
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

    # Import proper evaluation from benchmark
    sys.path.insert(0, "/root/beyond-deduction/benchmark")
    from evaluate import parse_hypotheses_from_response, compute_strong_accuracy, parse_ground_truth

    # Run inference
    h1_results = []
    h2_results = []

    print(f"\nRunning inference on {len(pairs)} pairs...")
    for i, (h1, h2) in enumerate(pairs):
        if i % 50 == 0:
            print(f"  Processing pair {i+1}/{len(pairs)}")

        # H1
        prompt_h1 = format_prompt(h1['theories_nl'], h1['observations_nl'])
        reply_h1 = generate_response(prompt_h1)
        pred_hyps_h1 = parse_hypotheses_from_response(reply_h1)
        gt_hyps_h1 = parse_ground_truth(h1['gt_hypothesis_nl'])
        strong_h1 = compute_strong_accuracy(pred_hyps_h1, gt_hyps_h1)

        h1_results.append({
            'reply': reply_h1,
            'gt': h1['gt_hypothesis_nl'],
            'strong': strong_h1,
            'depth': h1['depth'],
        })

        # H2
        prompt_h2 = format_prompt(h2['theories_nl'], h2['observations_nl'])
        reply_h2 = generate_response(prompt_h2)
        pred_hyps_h2 = parse_hypotheses_from_response(reply_h2)
        gt_hyps_h2 = parse_ground_truth(h2['gt_hypothesis_nl'])
        strong_h2 = compute_strong_accuracy(pred_hyps_h2, gt_hyps_h2)

        h2_results.append({
            'reply': reply_h2,
            'gt': h2['gt_hypothesis_nl'],
            'strong': strong_h2,
            'depth': h2['depth'],
        })

        # Memory cleanup
        if i % 100 == 0:
            torch.cuda.empty_cache()

    # Compute accuracies
    h1_acc = sum(r['strong'] for r in h1_results) / len(h1_results)
    h2_acc = sum(r['strong'] for r in h2_results) / len(h2_results)

    print(f"\nResults:")
    print(f"  H1 accuracy: {h1_acc:.1%}")
    print(f"  H2 accuracy: {h2_acc:.1%}")
    print(f"  Sum: {h1_acc + h2_acc:.1%}")

    # Stratify
    n_h1_fail_h2_success = sum(1 for i in range(len(h1_results))
                               if h1_results[i]['strong'] == 0 and h2_results[i]['strong'] == 1)
    print(f"  Gold pairs (H1 fail + H2 success): {n_h1_fail_h2_success}")

    # Save results
    results = {
        'set1': {
            'h1_strong': h1_acc,
            'h2_strong': h2_acc,
            'h1_results': h1_results,
            'h2_results': h2_results,
            'pairs': pairs,
        }
    }

    output_path = pairs_path.replace("matched_pairs", "factorial_results")
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved results to {output_path}")
    data_vol.commit()

    return output_path


# =============================================================================
# STEP 3: Run Attention Analysis
# =============================================================================

@app.function(
    image=pipeline_image,
    gpu="H100",
    timeout=120 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/data": data_vol,
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
    """Run attention analysis on the factorial results."""
    import sys
    import os
    import pickle
    import torch

    # Add paths BEFORE loading pickle (needed for morphology module)
    sys.path.insert(0, "/root/beyond-deduction")
    sys.path.insert(0, "/root/beyond-deduction/benchmark")
    os.chdir("/root/beyond-deduction")

    print("=" * 60)
    print("STEP 3: ATTENTION ANALYSIS")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    with open(pairs_path, "rb") as f:
        pairs = pickle.load(f)

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    print(f"Loaded {len(pairs)} pairs")

    # Parse layers
    focus_layers = [int(l) for l in layers.split(",")]
    print(f"Focus layers: {focus_layers}")

    # Create probing examples manually (since paths differ)
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
    h1_results = results['set1']['h1_results']
    h2_results = results['set1']['h2_results']

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

    print("\nLoading Gemma 2 9B IT for attention extraction...")
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

    return summary


# =============================================================================
# DOWNLOAD RESULTS
# =============================================================================

@app.function(
    image=pipeline_image,
    volumes={"/root/results": results_vol, "/root/data": data_vol},
)
def download_all_results() -> dict:
    """Download all result files."""
    import os

    files = {}

    # Results
    results_dir = "/root/results"
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            filepath = os.path.join(results_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, "rb") as f:
                    files[f"results/{filename}"] = f.read()

    # Data (factorial results)
    data_dir = "/root/data"
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if "factorial" in filename:
                filepath = os.path.join(data_dir, filename)
                if os.path.isfile(filepath):
                    with open(filepath, "rb") as f:
                        files[f"data/{filename}"] = f.read()

    return files


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main(
    n_pairs: int = 1000,
    layers: str = "4,6,8,10,12,14,16,18,20",
    max_attention: int = None,
    generate_only: bool = False,
    skip_generate: bool = False,
    skip_inference: bool = False,
    use_original_data: bool = False,
    download: bool = True,
):
    """
    Run the full attention analysis pipeline.

    Args:
        n_pairs: Number of pairs to generate (default: 1000)
        layers: Comma-separated layers to analyze
        max_attention: Max examples per category for attention analysis
        generate_only: Only generate data, skip GPU steps
        skip_generate: Skip data generation, use existing
        skip_inference: Skip inference, use existing factorial results
        use_original_data: Use original 200-pair dataset with correct labels
        download: Download results after completion
    """
    import json

    print("=" * 60)
    print("FULL ATTENTION ANALYSIS PIPELINE")
    print("=" * 60)

    # Option to use original data with correct labels
    if use_original_data:
        print("Using ORIGINAL 200-pair dataset with correct labels")
        pairs_path = "/root/beyond-deduction/benchmark/matched_pairs_set1_pure.pkl"
        results_path = "/root/beyond-deduction/benchmark/factorial_results/factorial_gemma29b.pkl"
        print(f"  Pairs: {pairs_path}")
        print(f"  Results: {results_path}")
    else:
        print(f"n_pairs: {n_pairs}")
        print(f"layers: {layers}")

        # Step 1: Generate data
        if not skip_generate:
            print("\n[STEP 1] Generating dataset...")
            pairs_path = generate_large_dataset.remote(n_pairs=n_pairs)
            print(f"  Generated: {pairs_path}")
        else:
            pairs_path = f"/root/data/matched_pairs_large_{n_pairs}.pkl"
            print(f"\n[STEP 1] Skipped - using existing: {pairs_path}")

        if generate_only:
            print("\n--generate-only specified, stopping here.")
            return

        # Step 2: Run inference
        if not skip_inference:
            print("\n[STEP 2] Running factorial inference...")
            results_path = run_factorial_inference.remote(pairs_path)
            print(f"  Results: {results_path}")
        else:
            results_path = pairs_path.replace("matched_pairs", "factorial_results")
            print(f"\n[STEP 2] Skipped - using existing: {results_path}")

    # Step 3: Run attention analysis
    print("\n[STEP 3] Running attention analysis...")
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
    if "aggregate" in summary:
        agg = summary["aggregate"]
        print("\n" + "=" * 60)
        print("INTERPRETATION")
        print("=" * 60)
        print(f"Dataset: {summary['n_pairs']} pairs")
        print(f"H1 failures analyzed: {summary['n_analyzed_failures']}")
        print(f"Gold pairs: {summary['n_gold_pairs']}")
        print(f"\nSubsumption attention difference: {agg['difference']:+.4f}")
        print(f"Statistical significance: p={agg['p_value']:.4f}")

        if agg["p_value"] < 0.05:
            if agg["difference"] > 0:
                print("\n>>> H3 SUPPORTED: Failures attend MORE to subsumption")
            else:
                print("\n>>> H3 REJECTED: Successes attend MORE to subsumption")
        else:
            print("\n>>> H3 INCONCLUSIVE: No significant difference")

    # Download
    if download:
        print("\n[DOWNLOAD] Fetching results...")
        files = download_all_results.remote()

        output_dir = Path(__file__).parent / "attention_results"
        output_dir.mkdir(exist_ok=True)

        for filepath, data in files.items():
            local_path = output_dir / filepath
            local_path.parent.mkdir(exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(data)
            print(f"  Saved: {local_path}")

        print(f"\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    print("Run with: modal run mi/full_attention_pipeline_modal.py")
    print("Quick test: modal run mi/full_attention_pipeline_modal.py --n-pairs 100")
