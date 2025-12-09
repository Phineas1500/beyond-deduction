"""
Expanded tokenizer supporting full INABHYD vocabulary.

Token layout (~311 tokens):
- 0-3: Special tokens (PAD, BOS, EOS, UNK)
- 4-7: Structure markers ([WORLD_MODEL], [OBSERVATIONS], [TASK], [ANSWER])
- 8-17: Logical symbols (forall, ->, (, ), (x), newline, ~, ., :, x)
- 18-106: Concepts (89 tokens: c0-c88)
- 107-210: Properties (104 tokens: p1-p52 and ~p1-~p52)
- 211-310: Entities (100 tokens: e0-e99)

Total: ~311 tokens

This tokenizer is designed for training transformers on symbolic FOL
representations of INABHYD-style reasoning tasks.
"""
from typing import List, Dict, Optional, Tuple
import re
import json


class IsomorphicTokenizer:
    """
    Tokenizer for isomorphic symbolic FOL representations.

    Supports the full INABHYD vocabulary with:
    - 89 concepts
    - 52 properties (with negation variants)
    - 100 entities
    """

    # Special tokens
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    # Structure markers
    WORLD_MODEL_TOKEN = "[WORLD_MODEL]"
    OBSERVATIONS_TOKEN = "[OBSERVATIONS]"
    TASK_TOKEN = "[TASK]"
    ANSWER_TOKEN = "[ANSWER]"

    # Logical symbols
    FORALL_TOKEN = "forall"
    IMPLIES_TOKEN = "->"
    OPEN_PAREN = "("
    CLOSE_PAREN = ")"
    PRED_X = "(x)"
    NEWLINE = "\n"
    NEGATION = "~"
    PERIOD = "."
    COLON = ":"
    VAR_X = "x"

    def __init__(
        self,
        num_concepts: int = 89,
        num_properties: int = 52,  # Base properties (will double for negation)
        num_entities: int = 100,
        max_seq_len: int = 1024
    ):
        """
        Initialize the tokenizer.

        Args:
            num_concepts: Number of concept tokens (c0 to c{n-1})
            num_properties: Number of base property tokens (p1 to p{n})
            num_entities: Number of entity tokens (e0 to e{n-1})
            max_seq_len: Maximum sequence length for padding
        """
        self.num_concepts = num_concepts
        self.num_properties = num_properties
        self.num_entities = num_entities
        self.max_seq_len = max_seq_len

        self._build_vocab()

    def _build_vocab(self):
        """Build token-to-ID and ID-to-token mappings."""
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        current_id = 0

        # Special tokens (0-3)
        for token in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1

        # Structure markers (4-7)
        for token in [self.WORLD_MODEL_TOKEN, self.OBSERVATIONS_TOKEN,
                      self.TASK_TOKEN, self.ANSWER_TOKEN]:
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1

        # Logical symbols (8-17)
        for token in [self.FORALL_TOKEN, self.IMPLIES_TOKEN, self.OPEN_PAREN,
                      self.CLOSE_PAREN, self.PRED_X, self.NEWLINE,
                      self.NEGATION, self.PERIOD, self.COLON, self.VAR_X]:
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1

        # Record start indices
        self.concept_start_id = current_id

        # Concept tokens: c0, c1, ..., c{num_concepts-1}
        for i in range(self.num_concepts):
            token = f"c{i}"
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1

        self.property_start_id = current_id

        # Property tokens: p1, ..., p{num_properties}
        # Plus negated: ~p1, ..., ~p{num_properties}
        for i in range(1, self.num_properties + 1):
            # Positive property
            token = f"p{i}"
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1

            # Negated property
            neg_token = f"~p{i}"
            self.token_to_id[neg_token] = current_id
            self.id_to_token[current_id] = neg_token
            current_id += 1

        self.entity_start_id = current_id

        # Entity tokens: e0, e1, ..., e{num_entities-1}
        for i in range(self.num_entities):
            token = f"e{i}"
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1

        self.vocab_size = current_id
        self.pad_token_id = self.token_to_id[self.PAD_TOKEN]
        self.bos_token_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_token_id = self.token_to_id[self.EOS_TOKEN]
        self.unk_token_id = self.token_to_id[self.UNK_TOKEN]

    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[str]:
        """
        Tokenize symbolic FOL text to token strings.

        Args:
            text: Symbolic FOL text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token strings
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.BOS_TOKEN)

        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Structure markers
            if line in [self.WORLD_MODEL_TOKEN, self.OBSERVATIONS_TOKEN,
                        self.TASK_TOKEN, self.ANSWER_TOKEN]:
                tokens.append(line)
                tokens.append(self.NEWLINE)
                continue

            # Skip task description text
            if line.startswith("Infer"):
                continue

            # Parse logical statements
            stmt_tokens = self._tokenize_statement(line)
            if stmt_tokens:
                tokens.extend(stmt_tokens)
                tokens.append(self.NEWLINE)

        if add_special_tokens:
            tokens.append(self.EOS_TOKEN)

        return tokens

    def _tokenize_statement(self, stmt: str) -> List[str]:
        """Tokenize a single FOL statement."""
        tokens = []
        stmt = stmt.strip()

        # Check for quantifier
        if stmt.startswith("forall"):
            tokens.append(self.FORALL_TOKEN)
            # Skip "forall x:" or "forall x :"
            match = re.match(r'forall\s+x\s*:\s*', stmt)
            if match:
                tokens.append(self.VAR_X)
                tokens.append(self.COLON)
                stmt = stmt[match.end():]

        # Split by ->
        if "->" in stmt:
            left, right = stmt.split("->", 1)
            tokens.extend(self._tokenize_predicate(left.strip()))
            tokens.append(self.IMPLIES_TOKEN)
            tokens.extend(self._tokenize_predicate(right.strip()))
        else:
            tokens.extend(self._tokenize_predicate(stmt))

        return tokens

    def _tokenize_predicate(self, pred: str) -> List[str]:
        """Tokenize a predicate like c0(x), p1(e0), ~p2(x)."""
        tokens = []
        pred = pred.strip()

        # Check for negation
        if pred.startswith("~"):
            # Handle ~pN(x) or ~pN(eM) - negation is part of property token
            match = re.match(r'(~p\d+)\(([xe]\d*)\)', pred)
            if match:
                tokens.append(match.group(1))  # ~p1
                arg = match.group(2)
                if arg == "x":
                    tokens.append(self.PRED_X)
                else:
                    tokens.append(self.OPEN_PAREN)
                    tokens.append(arg)  # e0
                    tokens.append(self.CLOSE_PAREN)
                return tokens

        # Match predicate pattern: cN(x), pN(x), cN(eM), pN(eM)
        match = re.match(r'([cp]\d+)\(([xe]\d*)\)', pred)
        if match:
            tokens.append(match.group(1))  # c0 or p1
            arg = match.group(2)
            if arg == "x":
                tokens.append(self.PRED_X)
            else:
                tokens.append(self.OPEN_PAREN)
                tokens.append(arg)  # e0
                tokens.append(self.CLOSE_PAREN)

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Symbolic FOL text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text, add_special_tokens)
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special: Whether to skip PAD/BOS/EOS tokens

        Returns:
            Decoded text string
        """
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        tokens = []
        for tid in token_ids:
            if skip_special and tid in special:
                continue
            tokens.append(self.id_to_token.get(tid, self.UNK_TOKEN))
        return self._reconstruct_text(tokens)

    def _reconstruct_text(self, tokens: List[str]) -> str:
        """Reconstruct readable text from tokens."""
        result = []
        for i, token in enumerate(tokens):
            if token == self.NEWLINE:
                result.append("\n")
            elif token == self.IMPLIES_TOKEN:
                result.append(" -> ")
            elif token == self.FORALL_TOKEN:
                result.append("forall ")
            elif token == self.COLON:
                result.append(": ")
            elif token == self.PRED_X:
                result.append("(x)")
            elif token == self.VAR_X:
                # Skip standalone x (handled by COLON)
                pass
            else:
                result.append(token)
        return "".join(result)

    def pad_sequence(self, ids: List[int], max_len: Optional[int] = None) -> List[int]:
        """
        Pad or truncate sequence to max_len.

        Args:
            ids: Token IDs to pad
            max_len: Maximum length (defaults to self.max_seq_len)

        Returns:
            Padded/truncated token IDs
        """
        max_len = max_len or self.max_seq_len
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [self.pad_token_id] * (max_len - len(ids))

    def create_attention_mask(self, ids: List[int]) -> List[int]:
        """Create attention mask (1 for real tokens, 0 for padding)."""
        return [1 if tid != self.pad_token_id else 0 for tid in ids]

    def get_vocab_info(self) -> Dict[str, any]:
        """Get vocabulary information."""
        return {
            "vocab_size": self.vocab_size,
            "num_concepts": self.num_concepts,
            "num_properties": self.num_properties,
            "num_entities": self.num_entities,
            "concept_range": (self.concept_start_id, self.concept_start_id + self.num_concepts),
            "property_range": (self.property_start_id, self.property_start_id + self.num_properties * 2),
            "entity_range": (self.entity_start_id, self.entity_start_id + self.num_entities),
            "special_tokens": {
                "pad": self.pad_token_id,
                "bos": self.bos_token_id,
                "eos": self.eos_token_id,
                "unk": self.unk_token_id,
            }
        }

    def save_vocab(self, path: str):
        """Save vocabulary to JSON file."""
        vocab_data = {
            "config": {
                "num_concepts": self.num_concepts,
                "num_properties": self.num_properties,
                "num_entities": self.num_entities,
                "max_seq_len": self.max_seq_len,
            },
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()}
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)

    @classmethod
    def load_vocab(cls, path: str) -> 'IsomorphicTokenizer':
        """Load vocabulary from JSON file."""
        with open(path, 'r') as f:
            vocab_data = json.load(f)

        config = vocab_data["config"]
        tokenizer = cls(
            num_concepts=config["num_concepts"],
            num_properties=config["num_properties"],
            num_entities=config["num_entities"],
            max_seq_len=config["max_seq_len"]
        )
        return tokenizer
