# Modal deployment for Gemma 3 27B IT with vLLM
# OpenAI-compatible API with greedy decoding
#
# Key Gemma 3 differences from Gemma 2:
# - 128k context (vs 8k) for 4B+ models
# - 5:1 local:global attention ratio (vs 1:1)
# - Sliding window of 1024 tokens (vs 4096)
# - Multimodal support (text + vision)

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

# Gemma 3 27B instruction-tuned model
MODEL_NAME = "google/gemma-3-27b-it"
MODEL_REVISION = "main"  # Use latest

# Modal volumes for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Fast boot for quicker cold starts
FAST_BOOT = True

app = modal.App("gemma3-27b-inference")

N_GPU = 1  # Gemma 3 27B fits on single H100 (80GB)
MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],  # For gated model access
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
        "gemma3-27b",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # Greedy decoding settings (temperature=0 enforced at request time)
        "--max-model-len", "8192",  # Reasonable context length
        "--dtype", "bfloat16",  # BF16 for H100
    ]

    # Fast boot vs optimized inference trade-off
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # Tensor parallelism for multi-GPU (not needed for single H100)
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print("Starting vLLM with command:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES):
    """Test the Gemma 3 27B endpoint."""
    url = serve.get_web_url()

    print(f"Gemma 3 27B server URL: {url}")

    # Test with INABHYD-style prompt (no system message for Gemma)
    instructions = """You are a helpful assistant that performs abduction and induction reasoning.
Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
Only output final hypotheses."""

    user_prompt = "Q: Jerry is a dalpist. Dalpists are brown. We observe that: Jerry is brown. Please come up with hypothesis to explain observations."

    # Combine into single user message (Gemma doesn't support system prompts)
    combined = f"{instructions}\n\n{user_prompt}"
    messages = [{"role": "user", "content": combined}]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            up = resp.status == 200
        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"\nSending INABHYD test prompt...")
        print(f"User: {user_prompt}\n")

        # Use greedy decoding (temperature=0)
        payload = {
            "messages": messages,
            "model": "gemma3-27b",
            "temperature": 0,  # Greedy decoding
            "max_tokens": 256,
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}

        async with session.post("/v1/chat/completions", json=payload, headers=headers, timeout=2 * MINUTES) as resp:
            resp.raise_for_status()
            result = await resp.json()
            response = result["choices"][0]["message"]["content"]
            print(f"Gemma 3 27B response:\n{response}")


if __name__ == "__main__":
    print("Deploy with: modal deploy gemma3_27b_modal.py")
    print("Test with: modal run gemma3_27b_modal.py")
