"""
Extract activations from Gemma 2 9B IT using nnsight.

Handles memory-efficient extraction for RTX 4090 (24GB) with ~18GB model.
"""

import torch
import gc
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

from .data_loader import ProbingExample
from .tokenizer_utils import TokenPositionFinder


@dataclass
class ActivationCache:
    """Cached activations for a single example."""

    example_idx: int
    h1_or_h2: str
    layer_activations: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)
    # layer_idx -> {'child_positions': Tensor, 'parent_positions': Tensor, 'final_position': Tensor}
    positions: Dict[str, List[int]] = field(default_factory=dict)


def clear_memory():
    """Clear CUDA cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ActivationExtractor:
    """Extract activations from specific positions and layers."""

    def __init__(
        self,
        model_wrapper,
        target_layers: List[int] = None
    ):
        """
        Initialize extractor.

        Args:
            model_wrapper: Gemma2Wrapper instance
            target_layers: Which layers to extract from (default: [8, 15, 20, 25, 30, 35, 40])
        """
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.tokenizer
        self.device = model_wrapper.device
        self.target_layers = target_layers or model_wrapper.target_layers
        self.hidden_size = model_wrapper.hidden_size

        self.position_finder = TokenPositionFinder(self.tokenizer)

    @torch.no_grad()
    def extract_activations(
        self,
        prompt: str,
        positions: Dict[str, List[int]],
        layers: Optional[List[int]] = None
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Extract activations at specified positions from specified layers.

        Uses nnsight's tracing context for efficient extraction.

        Args:
            prompt: Input prompt string
            positions: Dict mapping position type to list of token indices
                       e.g., {'child_positions': [5, 12], 'parent_positions': [8], 'final_position': 45}
            layers: Which layers to extract from (default: self.target_layers)

        Returns:
            Dict[layer_idx, Dict[position_type, Tensor of shape (n_positions, hidden_size)]]
        """
        layers = layers or self.target_layers

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)

        activations = {}

        # nnsight extraction is deprecated - use extract_activations_hf instead
        # Fall back to HF method
        return self.extract_activations_hf(prompt, positions, layers)

        # Old nnsight code (kept for reference):
        # saved_outputs = {}
        # with self.model.trace(input_ids) as tracer:
        #     for layer_idx in layers:
        #         saved_outputs[layer_idx] = self.model.model.layers[layer_idx].output[0].save()
        # for layer_idx in layers:
        #     layer_output = saved_outputs[layer_idx].value

        # This code is now unreachable but kept for structure:
        activations = {}
        for layer_idx in layers:
            try:
                layer_output = None  # placeholder
                # layer_output shape: (batch=1, seq_len, hidden_size)

                layer_acts = {}

                for pos_type, pos_indices in positions.items():
                    if pos_indices is None:
                        continue

                    # Handle single int (final_position) vs list
                    if isinstance(pos_indices, int):
                        pos_indices = [pos_indices]

                    if len(pos_indices) == 0:
                        continue

                    # Gather activations at specified positions
                    # Clamp indices to valid range
                    seq_len = layer_output.shape[1]
                    valid_indices = [min(idx, seq_len - 1) for idx in pos_indices if idx < seq_len]

                    if valid_indices:
                        acts = layer_output[0, valid_indices, :]  # (n_pos, hidden_size)
                        layer_acts[pos_type] = acts.cpu()

                activations[layer_idx] = layer_acts

            except Exception as e:
                print(f"Warning: Failed to extract layer {layer_idx}: {e}")
                activations[layer_idx] = {}

        return activations

    @torch.no_grad()
    def extract_activations_hf(
        self,
        prompt: str,
        positions: Dict[str, List[int]],
        layers: Optional[List[int]] = None
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Alternative extraction using HuggingFace's output_hidden_states.

        Use this if nnsight tracing has issues.

        Args:
            prompt: Input prompt string
            positions: Dict mapping position type to list of token indices
            layers: Which layers to extract from

        Returns:
            Dict[layer_idx, Dict[position_type, Tensor]]
        """
        layers = layers or self.target_layers

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)

        # Forward pass with hidden states
        outputs = self.model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_size)

        activations = {}

        for layer_idx in layers:
            # hidden_states[0] is embedding, hidden_states[1] is layer 0, etc.
            if layer_idx + 1 < len(hidden_states):
                hidden = hidden_states[layer_idx + 1]

                layer_acts = {}
                for pos_type, pos_indices in positions.items():
                    if pos_indices is None:
                        continue

                    if isinstance(pos_indices, int):
                        pos_indices = [pos_indices]

                    if len(pos_indices) == 0:
                        continue

                    seq_len = hidden.shape[1]
                    valid_indices = [min(idx, seq_len - 1) for idx in pos_indices if idx < seq_len]

                    if valid_indices:
                        acts = hidden[0, valid_indices, :]
                        layer_acts[pos_type] = acts.cpu()

                activations[layer_idx] = layer_acts

        return activations

    def extract_for_example(
        self,
        example: ProbingExample,
        use_nnsight: bool = True
    ) -> ActivationCache:
        """
        Extract activations for a single ProbingExample.

        Args:
            example: ProbingExample to process
            use_nnsight: Whether to use nnsight (True) or HF hidden_states (False)

        Returns:
            ActivationCache with activations at all target layers
        """
        # Find positions
        positions = self.position_finder.get_positions_for_probing(
            example.prompt,
            example.child_concept,
            example.root_concept
        )

        # Extract activations
        if use_nnsight:
            activations = self.extract_activations(example.prompt, positions)
        else:
            activations = self.extract_activations_hf(example.prompt, positions)

        return ActivationCache(
            example_idx=example.idx,
            h1_or_h2=example.h1_or_h2,
            layer_activations=activations,
            positions=positions
        )

    def extract_batch(
        self,
        examples: List[ProbingExample],
        chunk_size: int = 10,
        use_nnsight: bool = True,
        show_progress: bool = True
    ) -> List[ActivationCache]:
        """
        Extract activations for a batch of examples.

        Processes one example at a time for memory efficiency,
        with periodic memory cleanup.

        Args:
            examples: List of ProbingExample to process
            chunk_size: Clear memory every N examples
            use_nnsight: Whether to use nnsight or HF hidden_states
            show_progress: Whether to show progress bar

        Returns:
            List of ActivationCache objects
        """
        caches = []

        iterator = tqdm(examples, desc="Extracting activations") if show_progress else examples

        for i, example in enumerate(iterator):
            try:
                cache = self.extract_for_example(example, use_nnsight=use_nnsight)
                caches.append(cache)
            except Exception as e:
                print(f"Warning: Failed to extract example {example.idx}: {e}")
                # Create empty cache
                caches.append(ActivationCache(
                    example_idx=example.idx,
                    h1_or_h2=example.h1_or_h2,
                    layer_activations={},
                    positions={}
                ))

            # Clear memory periodically
            if (i + 1) % chunk_size == 0:
                clear_memory()

        # Final cleanup
        clear_memory()

        return caches


class MemoryEfficientExtractor:
    """
    Context manager for memory-efficient extraction.

    Processes examples in chunks and can optionally save to disk.
    """

    def __init__(
        self,
        model_wrapper,
        chunk_size: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Initialize memory-efficient extractor.

        Args:
            model_wrapper: Gemma2Wrapper instance
            chunk_size: Process this many examples before clearing memory
            save_path: Optional path to save intermediate results
        """
        self.extractor = ActivationExtractor(model_wrapper)
        self.chunk_size = chunk_size
        self.save_path = save_path

    def extract_all(
        self,
        examples: List[ProbingExample],
        use_nnsight: bool = True
    ) -> List[ActivationCache]:
        """
        Extract activations for all examples with memory management.

        Args:
            examples: List of examples to process
            use_nnsight: Whether to use nnsight

        Returns:
            List of ActivationCache objects
        """
        all_caches = []

        for i in range(0, len(examples), self.chunk_size):
            chunk = examples[i:i + self.chunk_size]

            print(f"Processing examples {i} to {i + len(chunk) - 1}...")

            # Extract chunk
            caches = self.extractor.extract_batch(
                chunk,
                chunk_size=self.chunk_size,
                use_nnsight=use_nnsight,
                show_progress=True
            )

            # Move all tensors to CPU
            for cache in caches:
                for layer_idx in cache.layer_activations:
                    for pos_type in cache.layer_activations[layer_idx]:
                        cache.layer_activations[layer_idx][pos_type] = \
                            cache.layer_activations[layer_idx][pos_type].cpu()

            all_caches.extend(caches)

            # Clear GPU memory
            clear_memory()

            # Optionally save intermediate results
            if self.save_path:
                import pickle
                with open(self.save_path, 'wb') as f:
                    pickle.dump(all_caches, f)
                print(f"Saved {len(all_caches)} caches to {self.save_path}")

        return all_caches
