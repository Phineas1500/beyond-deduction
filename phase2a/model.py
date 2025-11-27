"""
Simple Transformer Model for Mechanistic Interpretability.

Design choices inspired by Saparov et al. "Transformers Struggle to Learn to Search":
- Single attention head per layer (easier to analyze)
- Concatenated position embeddings (not added - cleaner separation)
- Pre-layer normalization option
- Small model size for full circuit reconstruction

This model is designed for MI analysis, not SOTA performance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for the interpretable transformer."""
    vocab_size: int = 100
    max_seq_len: int = 512
    n_layers: int = 4
    n_heads: int = 1  # Single head for MI analysis
    d_model: int = 64  # Small for tractability
    d_ff: int = 256  # Feedforward dimension
    dropout: float = 0.0  # No dropout for deterministic MI
    pad_token_id: int = 0
    
    # Architecture choices for interpretability
    concat_pos_emb: bool = True  # Concatenate vs add position embeddings
    pre_ln: bool = True  # Pre-layer norm (more stable)
    causal: bool = True  # Causal attention mask
    use_ff: bool = True  # Whether to use feedforward layers
    
    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads
    
    @property
    def embedding_dim(self) -> int:
        """Total embedding dimension including position if concatenated."""
        if self.concat_pos_emb:
            return self.d_model + self.max_seq_len
        return self.d_model


class SingleHeadAttention(nn.Module):
    """
    Single-head attention for maximum interpretability.
    
    With one head, we can directly visualize what the attention is doing
    without needing to decompose across heads.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Input dimension depends on whether we concat position embeddings
        input_dim = config.embedding_dim
        
        # Q, K, V projections
        self.proj_q = nn.Linear(input_dim, config.d_model, bias=True)
        self.proj_k = nn.Linear(input_dim, config.d_model, bias=True)
        self.proj_v = nn.Linear(input_dim, config.d_model, bias=True)
        
        # Output projection
        self.proj_out = nn.Linear(config.d_model, input_dim, bias=True)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(config.d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, embedding_dim)
            attention_mask: (batch, seq_len) with 1s for real tokens, 0s for padding
            return_attention: Whether to return attention weights
            
        Returns:
            output: (batch, seq_len, embedding_dim)
            attention_weights: (batch, seq_len, seq_len) if return_attention=True
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.proj_q(x)  # (batch, seq, d_model)
        K = self.proj_k(x)
        V = self.proj_v(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, seq, seq)
        
        # Apply causal mask if needed
        if self.config.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply padding mask
        if attention_mask is not None:
            # attention_mask: (batch, seq_len)
            # We need to mask positions where key is padding
            padding_mask = (attention_mask == 0).unsqueeze(1)  # (batch, 1, seq_len)
            scores = scores.masked_fill(padding_mask, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # (batch, seq, d_model)
        
        # Output projection
        output = self.proj_out(output)  # (batch, seq, embedding_dim)
        
        if return_attention:
            return output, attention_weights
        return output, None


class FeedForward(nn.Module):
    """Simple feedforward network with ReLU activation."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        input_dim = config.embedding_dim
        
        self.fc1 = nn.Linear(input_dim, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, input_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)  # ReLU for easier MI analysis
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-LN or post-LN."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.ln_attn = nn.LayerNorm(config.embedding_dim)
        self.attn = SingleHeadAttention(config)
        
        if config.use_ff:
            self.ln_ff = nn.LayerNorm(config.embedding_dim)
            self.ff = FeedForward(config)
        else:
            self.ln_ff = None
            self.ff = None
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, embedding_dim)
            attention_mask: (batch, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            output: (batch, seq_len, embedding_dim)
            attention_weights: Optional attention weights
        """
        if self.config.pre_ln:
            # Pre-layer normalization
            attn_out, attn_weights = self.attn(
                self.ln_attn(x), attention_mask, return_attention
            )
            x = x + self.dropout(attn_out)
            
            if self.ff is not None:
                x = x + self.dropout(self.ff(self.ln_ff(x)))
        else:
            # Post-layer normalization
            attn_out, attn_weights = self.attn(x, attention_mask, return_attention)
            x = self.ln_attn(x + self.dropout(attn_out))
            
            if self.ff is not None:
                x = self.ln_ff(x + self.dropout(self.ff(x)))
        
        return x, attn_weights


class InterpretableTransformer(nn.Module):
    """
    Interpretable transformer for MI research.
    
    Designed for full circuit reconstruction with:
    - Single attention head per layer
    - Concatenated position embeddings
    - Small dimensions
    - Easy activation caching
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(
            config.vocab_size, 
            config.d_model,
            padding_idx=config.pad_token_id
        )
        
        # Position embeddings
        if config.concat_pos_emb:
            # One-hot position encoding for concatenation
            # This creates a (max_seq_len, max_seq_len) identity matrix
            self.register_buffer(
                'position_embedding',
                torch.eye(config.max_seq_len)
            )
        else:
            # Learned position embeddings (added)
            self.position_embedding = nn.Embedding(
                config.max_seq_len,
                config.d_model
            )
        
        self.dropout_embedding = nn.Dropout(config.dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output layer norm and head
        self.ln_final = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get token + position embeddings.
        
        Args:
            input_ids: (batch, seq_len)
            
        Returns:
            embeddings: (batch, seq_len, embedding_dim)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # (batch, seq, d_model)
        
        # Position embeddings
        if self.config.concat_pos_emb:
            # Concatenate one-hot position encoding
            pos_emb = self.position_embedding[:seq_len, :seq_len]  # (seq, seq)
            pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, seq, seq)
            embeddings = torch.cat([token_emb, pos_emb], dim=-1)  # (batch, seq, d_model+seq)
        else:
            # Add learned position embeddings
            positions = torch.arange(seq_len, device=input_ids.device)
            pos_emb = self.position_embedding(positions)  # (seq, d_model)
            embeddings = token_emb + pos_emb
        
        return self.dropout_embedding(embeddings)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass with optional activation caching for MI.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            return_hidden_states: Return all layer outputs
            return_attention: Return attention weights
            
        Returns:
            dict with:
                logits: (batch, seq_len, vocab_size)
                hidden_states: List of (batch, seq_len, embedding_dim) if requested
                attentions: List of (batch, seq_len, seq_len) if requested
        """
        # Get embeddings
        x = self.get_embeddings(input_ids)
        
        hidden_states = [x] if return_hidden_states else None
        attentions = [] if return_attention else None
        
        # Apply transformer layers
        for layer in self.layers:
            x, attn_weights = layer(x, attention_mask, return_attention)
            
            if return_hidden_states:
                hidden_states.append(x)
            if return_attention:
                attentions.append(attn_weights)
        
        # Final layer norm and LM head
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        output = {"logits": logits}
        if return_hidden_states:
            output["hidden_states"] = hidden_states
        if return_attention:
            output["attentions"] = attentions
        
        return output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> torch.Tensor:
        """
        Simple greedy/sampling generation.
        
        Args:
            input_ids: (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            generated: (batch, seq_len + new_tokens)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions
                outputs = self(input_ids)
                logits = outputs["logits"][:, -1, :]  # (batch, vocab_size)
                
                if do_sample:
                    # Sample from distribution
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = logits.argmax(dim=-1, keepdim=True)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Stop if all sequences have EOS (assuming EOS = 2)
                if (next_token == 2).all():
                    break
        
        return input_ids


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the transformer model."""
    print("Testing InterpretableTransformer...")
    print("=" * 50)
    
    # Create config
    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=128,
        n_layers=4,
        n_heads=1,
        d_model=64,
        d_ff=256,
        concat_pos_emb=True,
        pre_ln=True,
        use_ff=True
    )
    
    print(f"Config:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  embedding_dim: {config.embedding_dim}")
    
    # Create model
    model = InterpretableTransformer(config)
    n_params = count_parameters(model)
    print(f"\nTotal parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward with all outputs
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        return_hidden_states=True,
        return_attention=True
    )
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Number of hidden states: {len(outputs['hidden_states'])}")
    print(f"Hidden state shape: {outputs['hidden_states'][0].shape}")
    print(f"Number of attention matrices: {len(outputs['attentions'])}")
    print(f"Attention shape: {outputs['attentions'][0].shape}")
    
    # Test generation
    print("\nTesting generation...")
    generated = model.generate(input_ids[:1], max_new_tokens=10)
    print(f"Generated shape: {generated.shape}")
    
    print("\nModel test complete!")


if __name__ == "__main__":
    test_model()
