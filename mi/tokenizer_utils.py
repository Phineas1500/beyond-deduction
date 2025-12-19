"""
Tokenization utilities and position finding for concept tokens.

Handles SentencePiece tokenization where concepts may span multiple tokens.
"""

import re
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedTokenizer


class TokenPositionFinder:
    """Find positions of specific concepts in tokenized sequences."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize with a tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer (e.g., from Gemma 2)
        """
        self.tokenizer = tokenizer

    def tokenize_prompt(self, prompt: str) -> Dict:
        """
        Tokenize a prompt and return token info.

        Args:
            prompt: Input text string

        Returns:
            Dict with:
                - input_ids: Token IDs tensor
                - attention_mask: Attention mask tensor
                - tokens: List of token strings
                - text: Original text
        """
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )

        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'tokens': tokens,
            'text': prompt,
        }

    def find_concept_positions(
        self,
        prompt: str,
        concept: str,
        return_all: bool = True
    ) -> List[int]:
        """
        Find token positions where a concept appears.

        Handles SentencePiece tokenization where:
        - Concepts may be split across tokens (e.g., "dal" + "pist")
        - Token boundaries may not align with word boundaries

        Strategy: Find character positions, then map to tokens.

        Args:
            prompt: Full prompt text
            concept: Concept name to find (e.g., "dalpist")
            return_all: If True, return all occurrences; if False, return first only

        Returns:
            List of token position indices (first token of each occurrence)
        """
        encoded = self.tokenize_prompt(prompt)
        tokens = encoded['tokens']
        input_ids = encoded['input_ids'][0]

        # Find all character positions of the concept
        prompt_lower = prompt.lower()
        concept_lower = concept.lower()

        char_positions = []
        start = 0
        while True:
            pos = prompt_lower.find(concept_lower, start)
            if pos == -1:
                break
            char_positions.append(pos)
            start = pos + 1
            if not return_all:
                break

        if not char_positions:
            return []

        # Build character-to-token mapping
        # Decode each token and track character offsets
        char_to_token = {}
        current_char = 0

        for tok_idx, tok in enumerate(tokens):
            # Skip special tokens
            if tok in ['<bos>', '<eos>', '<pad>', '<unk>']:
                continue

            # Get the decoded text for this token
            tok_text = self.tokenizer.decode([input_ids[tok_idx].item()])

            # Map each character in this token's decoded text
            for i in range(len(tok_text)):
                char_to_token[current_char + i] = tok_idx

            current_char += len(tok_text)

        # Alternative: Use offset_mapping if available (more reliable)
        # But Gemma's tokenizer may not support it well, so we use the above approach

        # Map character positions to token positions
        token_positions = []
        for char_pos in char_positions:
            # Find the token that contains this character position
            # Search in a small window since tokenization may shift things
            for offset in range(-5, len(concept) + 5):
                test_pos = char_pos + offset
                if test_pos in char_to_token:
                    tok_idx = char_to_token[test_pos]
                    if tok_idx not in token_positions:
                        token_positions.append(tok_idx)
                    break

        return token_positions

    def find_concept_positions_simple(
        self,
        prompt: str,
        concept: str,
    ) -> List[int]:
        """
        Simple approach: find tokens whose decoded text contains the concept.

        This is more robust for SentencePiece but may include partial matches.

        Args:
            prompt: Full prompt text
            concept: Concept name to find

        Returns:
            List of token positions
        """
        encoded = self.tokenize_prompt(prompt)
        tokens = encoded['tokens']
        concept_lower = concept.lower()

        positions = []
        for idx, tok in enumerate(tokens):
            # Clean up SentencePiece markers
            tok_clean = tok.replace('▁', ' ').strip().lower()
            if concept_lower in tok_clean or tok_clean in concept_lower:
                if len(tok_clean) >= 2:  # Avoid single-char matches
                    positions.append(idx)

        return positions

    def get_final_position(self, input_ids) -> int:
        """
        Get the final (last) token position for output prediction probe.

        Args:
            input_ids: Token IDs tensor

        Returns:
            Index of last token
        """
        if hasattr(input_ids, 'shape'):
            seq_len = input_ids.shape[-1]
        else:
            seq_len = len(input_ids)
        return seq_len - 1

    def find_subsumption_sentence(
        self,
        prompt: str,
        child_concept: str,
        root_concept: str
    ) -> List[int]:
        """
        Find token positions for the subsumption sentence "Each X is a Y".

        This is important for attention analysis - do models attend to
        the subsumption when deciding output level?

        Args:
            prompt: Full prompt text
            child_concept: Child concept name
            root_concept: Root/parent concept name

        Returns:
            List of token positions spanning the subsumption sentence
        """
        # Common patterns for subsumption sentences
        patterns = [
            f"each {child_concept} is a {root_concept}",
            f"every {child_concept} is a {root_concept}",
            f"all {child_concept}s are {root_concept}s",
            f"{child_concept}s are {root_concept}s",
        ]

        prompt_lower = prompt.lower()

        for pattern in patterns:
            start_pos = prompt_lower.find(pattern.lower())
            if start_pos != -1:
                end_pos = start_pos + len(pattern)

                # Find all tokens in this range
                encoded = self.tokenize_prompt(prompt)
                tokens = encoded['tokens']
                input_ids = encoded['input_ids'][0]

                # Build cumulative char count per token
                positions = []
                current_char = 0

                for idx, tok in enumerate(tokens):
                    if tok in ['<bos>', '<eos>', '<pad>', '<unk>']:
                        continue

                    tok_text = self.tokenizer.decode([input_ids[idx].item()])
                    tok_end = current_char + len(tok_text)

                    # Check if this token overlaps with the subsumption sentence
                    if current_char < end_pos and tok_end > start_pos:
                        positions.append(idx)

                    current_char = tok_end

                return positions

        return []

    def get_positions_for_probing(
        self,
        prompt: str,
        child_concept: str,
        root_concept: str
    ) -> Dict[str, List[int]]:
        """
        Get all relevant positions for probing.

        Returns:
            Dict with keys:
                - 'child_positions': Where child concept appears
                - 'parent_positions': Where parent/root concept appears
                - 'final_position': Last token position
                - 'subsumption_positions': Tokens for "Each X is a Y" sentence
        """
        encoded = self.tokenize_prompt(prompt)

        # Try simple approach first (more robust for SentencePiece)
        child_positions = self.find_concept_positions_simple(prompt, child_concept)
        parent_positions = self.find_concept_positions_simple(prompt, root_concept)

        # Fall back to char-mapping if simple approach fails
        if not child_positions:
            child_positions = self.find_concept_positions(prompt, child_concept)
        if not parent_positions:
            parent_positions = self.find_concept_positions(prompt, root_concept)

        return {
            'child_positions': child_positions,
            'parent_positions': parent_positions,
            'final_position': self.get_final_position(encoded['input_ids']),
            'subsumption_positions': self.find_subsumption_sentence(
                prompt, child_concept, root_concept
            ),
        }

    def debug_tokenization(self, prompt: str, concepts: List[str] = None) -> None:
        """
        Debug helper to visualize tokenization.

        Args:
            prompt: Input text
            concepts: Optional list of concepts to highlight
        """
        encoded = self.tokenize_prompt(prompt)
        tokens = encoded['tokens']
        input_ids = encoded['input_ids'][0]

        print("=" * 60)
        print("TOKENIZATION DEBUG")
        print("=" * 60)
        print(f"Input: {prompt[:100]}...")
        print(f"Total tokens: {len(tokens)}")
        print("-" * 60)

        for idx, (tok, tok_id) in enumerate(zip(tokens, input_ids)):
            decoded = self.tokenizer.decode([tok_id.item()])
            highlight = ""

            if concepts:
                for concept in concepts:
                    if concept.lower() in decoded.lower():
                        highlight = f" <-- {concept}"
                        break

            print(f"[{idx:3d}] {tok:20s} -> '{decoded}'{highlight}")

        print("=" * 60)
