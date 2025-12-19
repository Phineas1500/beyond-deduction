"""
Model loading for Gemma 2 9B IT.

Uses bfloat16 for memory efficiency (~18GB on RTX 4090).
Critical: Uses eager attention (never flash_attention_2) per research plan.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List


class Gemma2Wrapper:
    """Wrapper for Gemma 2 9B IT for activation extraction."""

    # Gemma 2 9B architecture constants
    N_LAYERS = 42
    HIDDEN_SIZE = 3584
    N_QUERY_HEADS = 16
    N_KV_HEADS = 8  # GQA: 2 query heads share each KV head

    # Target layers for probing (focus around layer 20 - optimal probe layer)
    TARGET_LAYERS = [8, 15, 20, 25, 30, 35, 40]

    def __init__(
        self,
        model_id: str = "google/gemma-2-9b-it",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_4bit: bool = False,  # Set True if memory constrained
    ):
        """
        Initialize Gemma 2 9B IT with HuggingFace.

        Args:
            model_id: HuggingFace model ID
            device: Device to load model on
            dtype: Model dtype (bfloat16 recommended for memory)
            use_4bit: Use 4-bit quantization for memory-constrained setups
        """
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.use_4bit = use_4bit

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load model with HuggingFace
        print(f"Loading {model_id}...")
        print(f"  dtype: {dtype}")
        print(f"  attn_implementation: eager (required for Gemma 2)")
        print(f"  4-bit quantization: {use_4bit}")

        if use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                quantization_config=quantization_config,
                attn_implementation="eager",  # CRITICAL: Never flash_attention_2
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation="eager",  # CRITICAL: Never flash_attention_2
            )

        print(f"Model loaded successfully")

    @property
    def n_layers(self) -> int:
        """Number of transformer layers."""
        return self.N_LAYERS

    @property
    def hidden_size(self) -> int:
        """Hidden dimension size."""
        return self.HIDDEN_SIZE

    @property
    def target_layers(self) -> List[int]:
        """Layers to target for probing."""
        return self.TARGET_LAYERS

    def get_layer_module(self, layer_idx: int):
        """
        Get the decoder layer module for a given index.

        Args:
            layer_idx: Layer index (0 to 41)

        Returns:
            The layer module
        """
        return self.model.model.layers[layer_idx]

    def tokenize(self, text: str, return_tensors: str = "pt") -> dict:
        """
        Tokenize input text.

        Args:
            text: Input text string
            return_tensors: Tensor format ("pt" for PyTorch)

        Returns:
            Tokenized inputs dict with input_ids and attention_mask
        """
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            add_special_tokens=True,
        )

    def is_even_layer(self, layer_idx: int) -> bool:
        """
        Check if layer has global attention.

        Gemma 2 architecture:
        - Even layers (0, 2, 4, ...): Global attention (8192 tokens)
        - Odd layers (1, 3, 5, ...): Local sliding window (4096 tokens)

        For long-range reasoning, focus on even layers.
        """
        return layer_idx % 2 == 0

    def get_global_attention_layers(self) -> List[int]:
        """Get indices of layers with global attention (even layers)."""
        return [i for i in range(self.N_LAYERS) if self.is_even_layer(i)]

    def get_kv_group(self, query_head: int) -> int:
        """
        Get KV group for a query head (GQA).

        With 16 query heads and 8 KV heads, each KV group has 2 query heads:
        - Query heads 0, 1 -> KV group 0
        - Query heads 2, 3 -> KV group 1
        - etc.
        """
        return query_head // 2

    def get_query_heads_for_kv_group(self, kv_group: int) -> List[int]:
        """Get query head indices for a KV group."""
        return [kv_group * 2, kv_group * 2 + 1]


def load_gemma2_for_probing(
    use_4bit: bool = False,
    device: str = "cuda",
) -> Gemma2Wrapper:
    """
    Convenience function to load Gemma 2 9B IT for probing experiments.

    Args:
        use_4bit: Use 4-bit quantization (for memory-constrained setups)
        device: Device to load on

    Returns:
        Gemma2Wrapper instance
    """
    return Gemma2Wrapper(
        model_id="google/gemma-2-9b-it",
        device=device,
        dtype=torch.bfloat16,
        use_4bit=use_4bit,
    )
