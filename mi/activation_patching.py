"""
Activation Patching for Causal Analysis of Over-Generalization in Gemma 2 9B
=============================================================================

Tests causal role of Layer 8 residual stream and Layer 12 KV Group 1 in
determining whether model outputs parent vs child concept.

Experiments:
1. Layer 8 Residual Stream Patching
   - Patch activations from SUCCESS (child output) into FAILURE (parent output)
   - If output flips to child: Layer 8 is causally involved

2. Layer 12 KV1 Attention Output Patching
   - Patch attention outputs from specific KV group
   - Tests if attention pattern (found significant in attention analysis) is causal

Key insight from prior work:
- Probing: Decision decodable at layer 8 (94% accuracy)
- Attention: Layer 12 KV1 shows p=0.0037 difference
- But: These are correlational. Patching tests causation.
"""

import sys
import os
import torch
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from tqdm import tqdm
import random
from collections import defaultdict


@dataclass
class PatchingConfig:
    """Configuration for activation patching experiments."""

    # Target layers for patching
    residual_layers: List[int] = field(default_factory=lambda: [8, 12, 16, 20])
    attention_layer: int = 12
    attention_kv_group: int = 1  # From attention analysis (p=0.0037)

    # Number of examples to test
    n_pairs: int = 50  # Number of (source, target) pairs to test

    # Generation settings
    max_new_tokens: int = 100
    temperature: float = 0.0  # Greedy for reproducibility

    # Random seed
    seed: int = 42


@dataclass
class PatchingResult:
    """Result from a single patching experiment."""

    source_idx: int
    target_idx: int

    # Original outputs
    source_output: str
    target_output: str
    source_level: str  # 'parent', 'child', 'both', 'neither'
    target_level: str

    # Patched outputs
    patched_output: str
    patched_level: str

    # Did patching flip the output?
    flipped: bool

    # Concepts for reference
    parent_concept: str
    child_concept: str


SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
Only output final hypotheses."""


def format_prompt(theories_nl: str, observations_nl: str, tokenizer=None) -> str:
    """Format prompt for Gemma 2 IT with proper chat template."""
    user_prompt = f"Q: {theories_nl} We observe that: {observations_nl} Please come up with hypothesis to explain observations."

    # Combine system prompt and user message (Gemma 2 doesn't support system role)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    if tokenizer is not None:
        # Use chat template
        messages = [{"role": "user", "content": full_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return full_prompt


def check_output_level(generated_text: str, parent_concept: str, child_concept: str) -> str:
    """
    Check if generated text contains parent and/or child concept.
    Returns: 'parent', 'child', 'both', or 'neither'
    """
    gen_lower = generated_text.lower()
    has_parent = parent_concept.lower() in gen_lower
    has_child = child_concept.lower() in gen_lower

    if has_parent and has_child:
        return 'both'
    elif has_parent:
        return 'parent'
    elif has_child:
        return 'child'
    else:
        return 'neither'


def generate_with_hook(
    model,
    tokenizer,
    prompt: str,
    hook_fn: Optional[Callable] = None,
    hook_layer: int = 8,
    hook_type: str = "residual",  # "residual" or "attention"
    max_new_tokens: int = 100,
) -> str:
    """
    Generate text with optional intervention hook.

    Args:
        model: Gemma 2 model
        tokenizer: Tokenizer
        prompt: Input prompt
        hook_fn: Function to apply during forward pass
        hook_layer: Which layer to hook
        hook_type: "residual" (full layer output) or "attention" (attention output only)
        max_new_tokens: Max tokens to generate

    Returns:
        Generated text (excluding prompt)
    """
    tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = tokens.input_ids.shape[1]

    handles = []

    if hook_fn is not None:
        if hook_type == "residual":
            # Hook the full layer output (residual stream after layer)
            def hook(module, input, output):
                # output is tuple: (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    modified = hook_fn(output[0])
                    return (modified,) + output[1:]
                else:
                    return hook_fn(output)

            handle = model.model.layers[hook_layer].register_forward_hook(hook)
            handles.append(handle)

        elif hook_type == "attention":
            # Hook the self-attention module output
            def hook(module, input, output):
                # Attention module output is (attn_output, attn_weights, past_key_value)
                if isinstance(output, tuple):
                    modified = hook_fn(output[0])
                    return (modified,) + output[1:]
                else:
                    return hook_fn(output)

            handle = model.model.layers[hook_layer].self_attn.register_forward_hook(hook)
            handles.append(handle)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                tokens.input_ids,
                attention_mask=tokens.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy
                pad_token_id=tokenizer.pad_token_id,
            )
        # Decode only the generated part
        generated_ids = output_ids[0, prompt_len:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()

    return generated


def capture_activations(
    model,
    tokenizer,
    prompt: str,
    layers: List[int],
    capture_attention: bool = False,
) -> Dict[int, torch.Tensor]:
    """
    Run forward pass and capture activations at specified layers.

    Returns:
        Dict mapping layer index to activation tensor [1, seq_len, hidden_size]
    """
    tokens = tokenizer(prompt, return_tensors="pt").to(model.device)

    activations = {}
    attention_outputs = {}
    handles = []

    for layer in layers:
        # Capture residual stream (layer output)
        def make_residual_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[layer_idx] = output[0].detach().clone()
                else:
                    activations[layer_idx] = output.detach().clone()
                return output
            return hook

        handle = model.model.layers[layer].register_forward_hook(make_residual_hook(layer))
        handles.append(handle)

        if capture_attention:
            # Capture attention output
            def make_attn_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        attention_outputs[layer_idx] = output[0].detach().clone()
                    else:
                        attention_outputs[layer_idx] = output.detach().clone()
                    return output
                return hook

            handle = model.model.layers[layer].self_attn.register_forward_hook(make_attn_hook(layer))
            handles.append(handle)

    try:
        with torch.no_grad():
            model(**tokens)
    finally:
        for h in handles:
            h.remove()

    if capture_attention:
        return activations, attention_outputs
    return activations


def run_residual_patching_experiment(
    model,
    tokenizer,
    pairs: List,
    h1_results: List,
    config: PatchingConfig,
) -> Dict:
    """
    Experiment 1: Residual Stream Patching at Layer 8

    Patch from SUCCESS examples (output child) into FAILURE examples (output parent).
    If patching causes failures to output child, Layer 8 is causally involved.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: RESIDUAL STREAM PATCHING")
    print("="*70)

    # Identify success and failure indices
    success_indices = [i for i, r in enumerate(h1_results) if r['strong'] == 1]
    failure_indices = [i for i, r in enumerate(h1_results) if r['strong'] == 0]

    print(f"H1 Successes (child output): {len(success_indices)}")
    print(f"H1 Failures (parent output): {len(failure_indices)}")

    if len(failure_indices) == 0:
        print("No failures to patch into!")
        return {"error": "no_failures"}

    # Sample pairs
    random.seed(config.seed)
    n_to_test = min(config.n_pairs, len(failure_indices), len(success_indices))

    sampled_failures = random.sample(failure_indices, n_to_test)
    sampled_successes = random.sample(success_indices, n_to_test)

    results_by_layer = {layer: [] for layer in config.residual_layers}

    for layer in config.residual_layers:
        print(f"\n--- Layer {layer} ---")

        for i, (fail_idx, succ_idx) in enumerate(tqdm(
            zip(sampled_failures, sampled_successes),
            total=n_to_test,
            desc=f"Layer {layer}"
        )):
            h1_fail, _ = pairs[fail_idx]
            h1_succ, _ = pairs[succ_idx]

            # Get prompts
            fail_prompt = format_prompt(
                h1_fail['theories_nl'],
                h1_fail['observations_nl'],
                tokenizer
            )
            succ_prompt = format_prompt(
                h1_succ['theories_nl'],
                h1_succ['observations_nl'],
                tokenizer
            )

            # Concepts (from failure example for evaluation)
            parent_concept = h1_fail['root_concept']
            child_concept = h1_fail['child_concept']

            # 1. Get baseline failure output
            fail_output = generate_with_hook(model, tokenizer, fail_prompt)
            fail_level = check_output_level(fail_output, parent_concept, child_concept)

            # 2. Get baseline success output
            succ_output = generate_with_hook(model, tokenizer, succ_prompt)
            succ_level = check_output_level(succ_output, parent_concept, child_concept)

            # 3. Capture success activations at this layer
            succ_activations = capture_activations(
                model, tokenizer, succ_prompt, [layer]
            )
            source_activation = succ_activations[layer]  # [1, seq_len, hidden_size]

            # 4. Create patching hook - replace last token position
            def patch_hook(hidden_states):
                modified = hidden_states.clone()
                # Patch the last token position with source activation
                # Need to handle different sequence lengths
                modified[:, -1, :] = source_activation[:, -1, :].to(modified.device)
                return modified

            # 5. Generate with patching
            patched_output = generate_with_hook(
                model, tokenizer, fail_prompt,
                hook_fn=patch_hook,
                hook_layer=layer,
                hook_type="residual"
            )
            patched_level = check_output_level(patched_output, parent_concept, child_concept)

            # 6. Record result
            flipped = (fail_level == 'parent' and patched_level == 'child')

            result = PatchingResult(
                source_idx=succ_idx,
                target_idx=fail_idx,
                source_output=succ_output[:200],
                target_output=fail_output[:200],
                source_level=succ_level,
                target_level=fail_level,
                patched_output=patched_output[:200],
                patched_level=patched_level,
                flipped=flipped,
                parent_concept=parent_concept,
                child_concept=child_concept,
            )
            results_by_layer[layer].append(result)

        # Summary for this layer
        layer_results = results_by_layer[layer]
        n_flipped = sum(1 for r in layer_results if r.flipped)
        n_valid = sum(1 for r in layer_results if r.target_level == 'parent')

        print(f"  Flipped (parent→child): {n_flipped}/{n_valid} ({100*n_flipped/max(1,n_valid):.1f}%)")

    return {
        "experiment": "residual_patching",
        "layers": config.residual_layers,
        "n_pairs": n_to_test,
        "results_by_layer": {
            layer: [vars(r) for r in results]
            for layer, results in results_by_layer.items()
        },
        "summary": {
            layer: {
                "n_flipped": sum(1 for r in results if r.flipped),
                "n_valid": sum(1 for r in results if r.target_level == 'parent'),
                "flip_rate": sum(1 for r in results if r.flipped) / max(1, sum(1 for r in results if r.target_level == 'parent'))
            }
            for layer, results in results_by_layer.items()
        }
    }


def run_attention_patching_experiment(
    model,
    tokenizer,
    pairs: List,
    h1_results: List,
    config: PatchingConfig,
) -> Dict:
    """
    Experiment 2: Attention Output Patching at Layer 12

    Patch attention outputs from SUCCESS into FAILURE at the specific layer.
    Tests if the attention mechanism (not just residual stream) is causal.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: ATTENTION OUTPUT PATCHING (Layer 12)")
    print("="*70)

    layer = config.attention_layer

    # Identify success and failure indices
    success_indices = [i for i, r in enumerate(h1_results) if r['strong'] == 1]
    failure_indices = [i for i, r in enumerate(h1_results) if r['strong'] == 0]

    print(f"H1 Successes: {len(success_indices)}")
    print(f"H1 Failures: {len(failure_indices)}")
    print(f"Target layer: {layer}")

    # Sample pairs
    random.seed(config.seed)
    n_to_test = min(config.n_pairs, len(failure_indices), len(success_indices))

    sampled_failures = random.sample(failure_indices, n_to_test)
    sampled_successes = random.sample(success_indices, n_to_test)

    results = []

    for i, (fail_idx, succ_idx) in enumerate(tqdm(
        zip(sampled_failures, sampled_successes),
        total=n_to_test,
        desc="Attention patching"
    )):
        h1_fail, _ = pairs[fail_idx]
        h1_succ, _ = pairs[succ_idx]

        # Get prompts
        fail_prompt = format_prompt(
            h1_fail['theories_nl'],
            h1_fail['observations_nl'],
            tokenizer
        )
        succ_prompt = format_prompt(
            h1_succ['theories_nl'],
            h1_succ['observations_nl'],
            tokenizer
        )

        # Concepts
        parent_concept = h1_fail['root_concept']
        child_concept = h1_fail['child_concept']

        # 1. Baseline outputs
        fail_output = generate_with_hook(model, tokenizer, fail_prompt)
        fail_level = check_output_level(fail_output, parent_concept, child_concept)

        succ_output = generate_with_hook(model, tokenizer, succ_prompt)
        succ_level = check_output_level(succ_output, parent_concept, child_concept)

        # 2. Capture success attention outputs
        _, succ_attn = capture_activations(
            model, tokenizer, succ_prompt, [layer], capture_attention=True
        )
        source_attn = succ_attn[layer]

        # 3. Create patching hook for attention output
        def attn_patch_hook(attn_output):
            modified = attn_output.clone()
            # Patch last token position
            modified[:, -1, :] = source_attn[:, -1, :].to(modified.device)
            return modified

        # 4. Generate with attention patching
        patched_output = generate_with_hook(
            model, tokenizer, fail_prompt,
            hook_fn=attn_patch_hook,
            hook_layer=layer,
            hook_type="attention"
        )
        patched_level = check_output_level(patched_output, parent_concept, child_concept)

        # 5. Record result
        flipped = (fail_level == 'parent' and patched_level == 'child')

        result = PatchingResult(
            source_idx=succ_idx,
            target_idx=fail_idx,
            source_output=succ_output[:200],
            target_output=fail_output[:200],
            source_level=succ_level,
            target_level=fail_level,
            patched_output=patched_output[:200],
            patched_level=patched_level,
            flipped=flipped,
            parent_concept=parent_concept,
            child_concept=child_concept,
        )
        results.append(result)

    # Summary
    n_flipped = sum(1 for r in results if r.flipped)
    n_valid = sum(1 for r in results if r.target_level == 'parent')

    print(f"\nResults:")
    print(f"  Flipped (parent→child): {n_flipped}/{n_valid} ({100*n_flipped/max(1,n_valid):.1f}%)")

    return {
        "experiment": "attention_patching",
        "layer": layer,
        "n_pairs": n_to_test,
        "results": [vars(r) for r in results],
        "summary": {
            "n_flipped": n_flipped,
            "n_valid": n_valid,
            "flip_rate": n_flipped / max(1, n_valid)
        }
    }


def run_bidirectional_patching(
    model,
    tokenizer,
    pairs: List,
    h1_results: List,
    config: PatchingConfig,
) -> Dict:
    """
    Experiment 3: Bidirectional Patching at Layer 8

    Test both directions:
    - SUCCESS → FAILURE: Can we make failures output child?
    - FAILURE → SUCCESS: Can we make successes output parent?

    If both work, Layer 8 has full causal control.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: BIDIRECTIONAL PATCHING (Layer 8)")
    print("="*70)

    layer = 8

    # Identify success and failure indices
    success_indices = [i for i, r in enumerate(h1_results) if r['strong'] == 1]
    failure_indices = [i for i, r in enumerate(h1_results) if r['strong'] == 0]

    print(f"H1 Successes: {len(success_indices)}")
    print(f"H1 Failures: {len(failure_indices)}")

    # Sample pairs
    random.seed(config.seed)
    n_to_test = min(config.n_pairs, len(failure_indices), len(success_indices))

    sampled_failures = random.sample(failure_indices, n_to_test)
    sampled_successes = random.sample(success_indices, n_to_test)

    # Direction 1: SUCCESS → FAILURE (flip to child)
    print("\n--- Direction 1: SUCCESS → FAILURE (flip to child) ---")
    s2f_results = []

    for fail_idx, succ_idx in tqdm(zip(sampled_failures, sampled_successes), total=n_to_test):
        h1_fail, _ = pairs[fail_idx]
        h1_succ, _ = pairs[succ_idx]

        fail_prompt = format_prompt(h1_fail['theories_nl'], h1_fail['observations_nl'], tokenizer)
        succ_prompt = format_prompt(h1_succ['theories_nl'], h1_succ['observations_nl'], tokenizer)

        parent_concept = h1_fail['root_concept']
        child_concept = h1_fail['child_concept']

        # Baseline
        fail_output = generate_with_hook(model, tokenizer, fail_prompt)
        fail_level = check_output_level(fail_output, parent_concept, child_concept)

        # Capture success activation
        succ_activations = capture_activations(model, tokenizer, succ_prompt, [layer])
        source_activation = succ_activations[layer]

        # Patch
        def patch_hook(hidden_states):
            modified = hidden_states.clone()
            modified[:, -1, :] = source_activation[:, -1, :].to(modified.device)
            return modified

        patched_output = generate_with_hook(
            model, tokenizer, fail_prompt,
            hook_fn=patch_hook, hook_layer=layer, hook_type="residual"
        )
        patched_level = check_output_level(patched_output, parent_concept, child_concept)

        flipped = (fail_level == 'parent' and patched_level == 'child')
        s2f_results.append({
            'fail_idx': fail_idx,
            'succ_idx': succ_idx,
            'original_level': fail_level,
            'patched_level': patched_level,
            'flipped': flipped
        })

    s2f_flipped = sum(1 for r in s2f_results if r['flipped'])
    s2f_valid = sum(1 for r in s2f_results if r['original_level'] == 'parent')
    print(f"  SUCCESS → FAILURE flip rate: {s2f_flipped}/{s2f_valid} ({100*s2f_flipped/max(1,s2f_valid):.1f}%)")

    # Direction 2: FAILURE → SUCCESS (flip to parent)
    print("\n--- Direction 2: FAILURE → SUCCESS (flip to parent) ---")
    f2s_results = []

    for fail_idx, succ_idx in tqdm(zip(sampled_failures, sampled_successes), total=n_to_test):
        h1_fail, _ = pairs[fail_idx]
        h1_succ, _ = pairs[succ_idx]

        fail_prompt = format_prompt(h1_fail['theories_nl'], h1_fail['observations_nl'], tokenizer)
        succ_prompt = format_prompt(h1_succ['theories_nl'], h1_succ['observations_nl'], tokenizer)

        parent_concept = h1_succ['root_concept']
        child_concept = h1_succ['child_concept']

        # Baseline
        succ_output = generate_with_hook(model, tokenizer, succ_prompt)
        succ_level = check_output_level(succ_output, parent_concept, child_concept)

        # Capture failure activation
        fail_activations = capture_activations(model, tokenizer, fail_prompt, [layer])
        source_activation = fail_activations[layer]

        # Patch
        def patch_hook(hidden_states):
            modified = hidden_states.clone()
            modified[:, -1, :] = source_activation[:, -1, :].to(modified.device)
            return modified

        patched_output = generate_with_hook(
            model, tokenizer, succ_prompt,
            hook_fn=patch_hook, hook_layer=layer, hook_type="residual"
        )
        patched_level = check_output_level(patched_output, parent_concept, child_concept)

        flipped = (succ_level == 'child' and patched_level == 'parent')
        f2s_results.append({
            'fail_idx': fail_idx,
            'succ_idx': succ_idx,
            'original_level': succ_level,
            'patched_level': patched_level,
            'flipped': flipped
        })

    f2s_flipped = sum(1 for r in f2s_results if r['flipped'])
    f2s_valid = sum(1 for r in f2s_results if r['original_level'] == 'child')
    print(f"  FAILURE → SUCCESS flip rate: {f2s_flipped}/{f2s_valid} ({100*f2s_flipped/max(1,f2s_valid):.1f}%)")

    return {
        "experiment": "bidirectional_patching",
        "layer": layer,
        "n_pairs": n_to_test,
        "success_to_failure": {
            "results": s2f_results,
            "n_flipped": s2f_flipped,
            "n_valid": s2f_valid,
            "flip_rate": s2f_flipped / max(1, s2f_valid)
        },
        "failure_to_success": {
            "results": f2s_results,
            "n_flipped": f2s_flipped,
            "n_valid": f2s_valid,
            "flip_rate": f2s_flipped / max(1, f2s_valid)
        }
    }


def print_summary(all_results: Dict):
    """Print summary of all patching experiments."""

    print("\n" + "="*70)
    print("ACTIVATION PATCHING SUMMARY")
    print("="*70)

    # Experiment 1: Residual patching by layer
    if "residual_patching" in all_results:
        exp1 = all_results["residual_patching"]
        print("\n1. RESIDUAL STREAM PATCHING (SUCCESS → FAILURE)")
        print("-"*50)
        print(f"{'Layer':<10} | {'Flipped':<15} | {'Rate':<10}")
        print("-"*50)
        for layer, summary in exp1["summary"].items():
            rate = summary["flip_rate"]
            print(f"{layer:<10} | {summary['n_flipped']}/{summary['n_valid']:<12} | {rate*100:>5.1f}%")

    # Experiment 2: Attention patching
    if "attention_patching" in all_results:
        exp2 = all_results["attention_patching"]
        print(f"\n2. ATTENTION OUTPUT PATCHING (Layer {exp2['layer']})")
        print("-"*50)
        summary = exp2["summary"]
        print(f"Flipped: {summary['n_flipped']}/{summary['n_valid']} ({summary['flip_rate']*100:.1f}%)")

    # Experiment 3: Bidirectional
    if "bidirectional_patching" in all_results:
        exp3 = all_results["bidirectional_patching"]
        print(f"\n3. BIDIRECTIONAL PATCHING (Layer {exp3['layer']})")
        print("-"*50)
        s2f = exp3["success_to_failure"]
        f2s = exp3["failure_to_success"]
        print(f"SUCCESS → FAILURE: {s2f['n_flipped']}/{s2f['n_valid']} ({s2f['flip_rate']*100:.1f}%)")
        print(f"FAILURE → SUCCESS: {f2s['n_flipped']}/{f2s['n_valid']} ({f2s['flip_rate']*100:.1f}%)")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if "residual_patching" in all_results:
        layer8_rate = all_results["residual_patching"]["summary"].get(8, {}).get("flip_rate", 0)

        if layer8_rate > 0.3:
            print(f"""
Layer 8 shows {layer8_rate*100:.1f}% causal effect!
→ CAUSAL EVIDENCE: Layer 8 residual stream causally controls output
→ The probing result (94% accuracy) reflects genuine causal structure
→ The decision IS made at layer 8, not just "readable" there
""")
        elif layer8_rate > 0.1:
            print(f"""
Layer 8 shows modest {layer8_rate*100:.1f}% causal effect.
→ PARTIAL CAUSATION: Layer 8 has some causal role
→ But other layers/components also contribute
→ Decision is distributed, not localized to layer 8
""")
        else:
            print(f"""
Layer 8 shows minimal {layer8_rate*100:.1f}% causal effect.
→ NO CAUSAL ROLE: Layer 8 is correlational, not causal
→ Probing detected "the speedometer, not the steering wheel"
→ Decision is made elsewhere despite being readable at layer 8
""")


def run_all_experiments(
    model,
    tokenizer,
    pairs_path: str,
    results_path: str,
    config: PatchingConfig,
    output_dir: str = "patching_results",
) -> Dict:
    """Run all activation patching experiments."""

    # Load data
    print("Loading data...")

    # Add benchmark to path for morphology import
    benchmark_path = str(Path(pairs_path).parent)
    if benchmark_path not in sys.path:
        sys.path.insert(0, benchmark_path)

    with open(pairs_path, 'rb') as f:
        pairs = pickle.load(f)

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    h1_results = results['set1']['h1_results']

    print(f"Loaded {len(pairs)} pairs")
    print(f"H1 results: {len(h1_results)}")

    # Conservation Law check
    h1_acc = sum(1 for r in h1_results if r['strong'] == 1) / len(h1_results)
    h2_results = results['set1']['h2_results']
    h2_acc = sum(1 for r in h2_results if r['strong'] == 1) / len(h2_results)

    print(f"\nConservation Law Check:")
    print(f"  H1 accuracy: {h1_acc*100:.1f}%")
    print(f"  H2 accuracy: {h2_acc*100:.1f}%")
    print(f"  Sum: {(h1_acc + h2_acc)*100:.1f}%")

    all_results = {}

    # Experiment 1: Residual patching
    all_results["residual_patching"] = run_residual_patching_experiment(
        model, tokenizer, pairs, h1_results, config
    )

    # Experiment 2: Attention patching
    all_results["attention_patching"] = run_attention_patching_experiment(
        model, tokenizer, pairs, h1_results, config
    )

    # Experiment 3: Bidirectional
    all_results["bidirectional_patching"] = run_bidirectional_patching(
        model, tokenizer, pairs, h1_results, config
    )

    # Print summary
    print_summary(all_results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / "activation_patching_results.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    # Local testing
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

    config = PatchingConfig(n_pairs=20)  # Small test

    run_all_experiments(
        model,
        tokenizer,
        pairs_path="benchmark/matched_pairs_set1_pure.pkl",
        results_path="benchmark/factorial_results/factorial_gemma29b.pkl",
        config=config,
    )
