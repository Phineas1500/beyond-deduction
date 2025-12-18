# Modal deployment for Llama 3.2 1B Instruct with vLLM
# OpenAI-compatible API with greedy decoding
# Small instruction-tuned model for MI research

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

# Llama 3.2 1B Instruct - instruction-tuned model
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "main"

# Modal volumes for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# HuggingFace secret for gated model access (Llama requires license acceptance)
hf_secret = modal.Secret.from_name("huggingface-secret")

# Fast boot for quicker cold starts
FAST_BOOT = True

app = modal.App("llama32-1b-inference")

N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu="L4",  # L4 is sufficient for 1B model
    scaledown_window=15 * MINUTES,
    timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[hf_secret],  # Required for gated model access
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
        "llama32-1b",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # Llama 3.2 1B settings
        "--max-model-len", "8192",
        "--dtype", "bfloat16",
    ]

    # Fast boot vs optimized inference trade-off
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # Tensor parallelism
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print("Starting vLLM with command:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES):
    """Test the Llama 3.2 1B endpoint."""
    import aiohttp

    url = serve.get_web_url()
    print(f"Llama 3.2 1B server URL: {url}")

    # Test with FOL-style prompt
    system_prompt = """You are a logical reasoning system that performs abduction and induction in first-order logic.
Your job is to produce hypotheses in FOL format that explain observations given theories.
Each hypothesis should take one of these forms:
- predicate(constant) for ground atoms (e.g., dalpist(Amy), rainy(Amy))
- forall x(P(x) -> Q(x)) for universal rules (e.g., forall x(dalpist(x) -> rainy(x)))
Output only FOL hypotheses, one per line."""

    user_prompt = "Theories: dalpist(Amy). dalpist(Jerry). dalpist(Pamela). Observations: rainy(Amy). rainy(Jerry). rainy(Pamela). Produce hypotheses to explain observations."

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            up = resp.status == 200
        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"\nSending FOL test prompt...")
        print(f"User: {user_prompt}\n")

        # Use chat completions API (instruction-tuned model)
        # Llama 3.2 supports system messages
        payload = {
            "model": "llama32-1b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0,
            "max_tokens": 256,
        }

        headers = {"Content-Type": "application/json"}

        async with session.post("/v1/chat/completions", json=payload, headers=headers, timeout=2 * MINUTES) as resp:
            resp.raise_for_status()
            result = await resp.json()
            response = result["choices"][0]["message"]["content"]
            print(f"Llama 3.2 1B response:\n{response}")


if __name__ == "__main__":
    print("Deploy with: modal deploy llama32_1b_modal.py")
    print("Test with: modal run llama32_1b_modal.py")
