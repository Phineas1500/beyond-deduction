# Modal deployment for Pythia 160M with vLLM
# OpenAI-compatible API with greedy decoding
# Note: Pythia is a base model (not instruction-tuned), uses completions API

import json
from typing import Any

import aiohttp
import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# Pythia 160M deduped - small base model, no special access needed
MODEL_NAME = "EleutherAI/pythia-160m-deduped"
MODEL_REVISION = "main"

# Modal volumes for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Fast boot for quicker cold starts
FAST_BOOT = True

app = modal.App("pythia-160m-inference")

N_GPU = 1  # Pythia 160M easily fits on any GPU
MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu="T4",  # T4 is more than enough for 160M params
    scaledown_window=15 * MINUTES,
    timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        "pythia-160m",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # Pythia 160M settings
        "--max-model-len", "2048",  # Pythia context length
        "--dtype", "float16",  # FP16 for T4
    ]

    # Fast boot vs optimized inference trade-off
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # Tensor parallelism for multi-GPU (not needed for single GPU)
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print("Starting vLLM with command:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES):
    """Test the Pythia 160M endpoint using completions API (base model)."""
    url = serve.get_web_url()

    print(f"Pythia 160M server URL: {url}")

    # For base models, format prompt as plain text (no chat template)
    # Using same structure as INABHYD but as completion
    prompt = """You are a helpful assitant that performs abduction and induction reasoning.
Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
Only output final hypotheses.

Q: Jerry is a dalpist. Dalpists are brown. We observe that: Jerry is brown. Please come up with hypothesis to explain observations.

A:"""

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            up = resp.status == 200
        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"\nSending INABHYD test prompt (completions API)...")
        print(f"Prompt:\n{prompt}\n")

        # Use completions API for base model with greedy decoding
        payload = {
            "prompt": prompt,
            "model": "pythia-160m",
            "temperature": 0,  # Greedy decoding
            "max_tokens": 256,
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}

        async with session.post("/v1/completions", json=payload, headers=headers, timeout=2 * MINUTES) as resp:
            resp.raise_for_status()
            result = await resp.json()
            response = result["choices"][0]["text"]
            print(f"Pythia 160M response:\n{response}")


if __name__ == "__main__":
    print("Deploy with: modal deploy pythia_modal.py")
    print("Test with: modal run pythia_modal.py")
