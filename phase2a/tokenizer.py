"""
Tokenizer for symbolic ontology notation.

This tokenizer converts symbolic logic expressions like:
    ∀x: c1(x) -> p2(x)
    c3(e0)
    
into integer token sequences for transformer training.

Design choices:
- Each logical symbol is one token (cleaner for MI analysis)
- Small vocabulary (~100 tokens)
- Special tokens for structure markers
"""

from typing import List, Dict, Tuple, Optional
import re
import json


class SymbolicOntologyTokenizer:
    """Tokenizer for symbolic ontology reasoning tasks."""
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    
    # Structure markers (from our dataset format)
    WORLD_MODEL_TOKEN = "[WORLD_MODEL]"
    OBSERVATIONS_TOKEN = "[OBSERVATIONS]"
    TASK_TOKEN = "[TASK]"
    ANSWER_TOKEN = "[ANSWER]"
    
    # Logical symbols
    FORALL_TOKEN = "∀x:"
    IMPLIES_TOKEN = "->"
    OPEN_PAREN = "("
    CLOSE_PAREN = ")"
    PRED_X = "(x)"  # predicate applied to variable x
    
    def __init__(
        self,
        max_concepts: int = 30,
        max_properties: int = 15,
        max_entities: int = 30,
        max_seq_len: int = 512
    ):
        """
        Initialize tokenizer with vocabulary.
        
        Args:
            max_concepts: Maximum number of concept tokens (c0, c1, ..., c{max_concepts-1})
            max_properties: Maximum number of property tokens (p1, p2, ..., p{max_properties})
            max_entities: Maximum number of entity tokens (e0, e1, ..., e{max_entities-1})
            max_seq_len: Maximum sequence length
        """
        self.max_concepts = max_concepts
        self.max_properties = max_properties
        self.max_entities = max_entities
        self.max_seq_len = max_seq_len
        
        self._build_vocab()
        
    def _build_vocab(self):
        """Build the token to ID mapping."""
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        current_id = 0
        
        # Special tokens (IDs 0-3)
        for token in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1
        
        # Structure markers (IDs 4-7)
        for token in [self.WORLD_MODEL_TOKEN, self.OBSERVATIONS_TOKEN, 
                      self.TASK_TOKEN, self.ANSWER_TOKEN]:
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1
        
        # Logical symbols (IDs 8-12)
        for token in [self.FORALL_TOKEN, self.IMPLIES_TOKEN, 
                      self.OPEN_PAREN, self.CLOSE_PAREN, self.PRED_X]:
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1
        
        # Newline token for separating statements
        self.token_to_id["\n"] = current_id
        self.id_to_token[current_id] = "\n"
        current_id += 1
        
        # Concept tokens: c0, c1, ..., c{max_concepts-1}
        for i in range(self.max_concepts):
            token = f"c{i}"
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1
        
        # Property tokens: p1, p2, ..., p{max_properties}
        for i in range(1, self.max_properties + 1):
            token = f"p{i}"
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1
        
        # Entity tokens: e0, e1, ..., e{max_entities-1}
        for i in range(self.max_entities):
            token = f"e{i}"
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1
        
        self.vocab_size = current_id
        self.pad_token_id = self.token_to_id[self.PAD_TOKEN]
        self.bos_token_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_token_id = self.token_to_id[self.EOS_TOKEN]
        self.unk_token_id = self.token_to_id[self.UNK_TOKEN]
        
    def _tokenize_statement(self, statement: str) -> List[str]:
        """
        Tokenize a single logical statement.
        
        Examples:
            "∀x: c1(x) -> p2(x)" -> ["∀x:", "c1", "(x)", "->", "p2", "(x)"]
            "c3(e0)" -> ["c3", "(", "e0", ")"]
            "p2(e1)" -> ["p2", "(", "e1", ")"]
        """
        statement = statement.strip()
        tokens = []
        
        # Check for universal quantifier
        if statement.startswith("∀x:"):
            tokens.append(self.FORALL_TOKEN)
            statement = statement[3:].strip()
        
        # Split by ->
        if "->" in statement:
            parts = statement.split("->")
            # Left side
            left = parts[0].strip()
            left_match = re.match(r'([cp]\d+)\(x\)', left)
            if left_match:
                tokens.append(left_match.group(1))
                tokens.append(self.PRED_X)
            
            tokens.append(self.IMPLIES_TOKEN)
            
            # Right side
            right = parts[1].strip()
            right_match = re.match(r'([cp]\d+)\(x\)', right)
            if right_match:
                tokens.append(right_match.group(1))
                tokens.append(self.PRED_X)
        else:
            # Ground fact: c3(e0) or p2(e1)
            match = re.match(r'([cp]\d+)\(([e]\d+)\)', statement)
            if match:
                tokens.append(match.group(1))
                tokens.append(self.OPEN_PAREN)
                tokens.append(match.group(2))
                tokens.append(self.CLOSE_PAREN)
        
        return tokens
    
    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[str]:
        """
        Tokenize a full input text.
        
        Args:
            text: Full input text with [WORLD_MODEL], [OBSERVATIONS], etc.
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of string tokens
        """
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.BOS_TOKEN)
        
        # Split by section markers
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section markers
            if line == "[WORLD_MODEL]":
                tokens.append(self.WORLD_MODEL_TOKEN)
                tokens.append("\n")
            elif line == "[OBSERVATIONS]":
                tokens.append(self.OBSERVATIONS_TOKEN)
                tokens.append("\n")
            elif line == "[TASK]":
                tokens.append(self.TASK_TOKEN)
                tokens.append("\n")
            elif line == "[ANSWER]":
                tokens.append(self.ANSWER_TOKEN)
                tokens.append("\n")
            elif line.startswith("Infer"):
                # Task description - skip for now (or could add as special token)
                continue
            else:
                # Logical statement
                stmt_tokens = self._tokenize_statement(line)
                tokens.extend(stmt_tokens)
                tokens.append("\n")
        
        if add_special_tokens:
            tokens.append(self.EOS_TOKEN)
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text, add_special_tokens)
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip PAD, BOS, EOS
            
        Returns:
            Decoded text string
        """
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            tokens.append(self.id_to_token.get(tid, self.UNK_TOKEN))
        
        # Reconstruct text
        result = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token in [self.WORLD_MODEL_TOKEN, self.OBSERVATIONS_TOKEN, 
                        self.TASK_TOKEN, self.ANSWER_TOKEN]:
                result.append(token)
            elif token == self.FORALL_TOKEN:
                # Start of a rule
                result.append(token + " ")
            elif token == self.IMPLIES_TOKEN:
                result.append(" " + token + " ")
            elif token == self.PRED_X:
                result.append(token)
            elif token == self.OPEN_PAREN:
                result.append(token)
            elif token == self.CLOSE_PAREN:
                result.append(token)
            elif token == "\n":
                result.append(token)
            elif token.startswith("c") or token.startswith("p") or token.startswith("e"):
                result.append(token)
            else:
                result.append(token)
            
            i += 1
        
        return "".join(result)
    
    def pad_sequence(
        self, 
        token_ids: List[int], 
        max_len: Optional[int] = None,
        padding_side: str = "right"
    ) -> List[int]:
        """
        Pad sequence to max_len.
        
        Args:
            token_ids: Input token IDs
            max_len: Maximum length (uses self.max_seq_len if None)
            padding_side: "left" or "right"
            
        Returns:
            Padded token IDs
        """
        if max_len is None:
            max_len = self.max_seq_len
            
        if len(token_ids) >= max_len:
            return token_ids[:max_len]
        
        padding = [self.pad_token_id] * (max_len - len(token_ids))
        
        if padding_side == "right":
            return token_ids + padding
        else:
            return padding + token_ids
    
    def save(self, path: str):
        """Save tokenizer config to JSON."""
        config = {
            "max_concepts": self.max_concepts,
            "max_properties": self.max_properties,
            "max_entities": self.max_entities,
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "SymbolicOntologyTokenizer":
        """Load tokenizer from JSON config."""
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(
            max_concepts=config["max_concepts"],
            max_properties=config["max_properties"],
            max_entities=config["max_entities"],
            max_seq_len=config["max_seq_len"]
        )


def test_tokenizer():
    """Test the tokenizer with sample inputs."""
    tokenizer = SymbolicOntologyTokenizer()
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print(f"BOS token ID: {tokenizer.bos_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print()
    
    # Test input from our dataset
    test_input = """[WORLD_MODEL]
∀x: c3(x) -> c1(x)
∀x: c4(x) -> c1(x)
∀x: c5(x) -> c2(x)
c3(e0)
c4(e2)
c5(e3)
∀x: c1(x) -> p2(x)
[OBSERVATIONS]
p2(e0)
p2(e2)
p2(e3)
[TASK]
Infer the most general rule."""

    print("Input text:")
    print(test_input)
    print()
    
    tokens = tokenizer.tokenize(test_input)
    print("Tokens:")
    print(tokens)
    print()
    
    token_ids = tokenizer.encode(test_input)
    print("Token IDs:")
    print(token_ids)
    print()
    
    decoded = tokenizer.decode(token_ids)
    print("Decoded:")
    print(decoded)
    print()
    
    # Test target encoding
    target = "∀x: c1(x) -> p2(x)"
    target_tokens = tokenizer.tokenize(target, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    print(f"Target: {target}")
    print(f"Target tokens: {target_tokens}")
    print(f"Target IDs: {target_ids}")


if __name__ == "__main__":
    test_tokenizer()
