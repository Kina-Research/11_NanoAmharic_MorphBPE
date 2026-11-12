"""
attention.py — Causal Self-Attention and Rotary Position Embeddings (RoPE).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) as introduced by Su et al. (2024).
    Computes rotary frequencies and applies 2D rotation to pairs of hidden dimensions.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin tables
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return precomputed (cos, sin) up to seq_len."""
        if seq_len > self.max_seq_len:
            # Recompute on the fly if sequence exceeds cached length
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos(), emb.sin()
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    q, k: (batch_size, num_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim)
    """
    # Reshape cos and sin to (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention with Rotary Position Embeddings and Multi-Head / Multi-Query support.
    Strict lower-triangular causal masking ensures no future token information leakage.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else (embedding_dim // num_heads)
        self.dropout_p = dropout
        self.max_seq_len = max_seq_len

        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_queries_per_kv = num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(embedding_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, embedding_dim, bias=False)
        self.out_proj._is_residual_proj = True

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

        # Precompute causal mask (lower triangular)
        # register_buffer ensures proper device placement and serialization
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Causal Self Attention.
        x: (batch_size, seq_len, embedding_dim)
        attention_mask: optional (batch_size, 1, 1, seq_len) padding mask
        """
        B, T, C = x.shape

        # Linear projections
        q = self.q_proj(x)  # (B, T, num_heads * head_dim)
        k = self.k_proj(x)  # (B, T, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, T, num_kv_heads * head_dim)

        # Reshape to (B, heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(T, device=x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # If Multi-Query Attention / Grouped Query Attention: repeat K, V heads
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Scaled dot-product attention scores
        # (B, num_heads, T, head_dim) x (B, num_heads, head_dim, T) -> (B, num_heads, T, T)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask (strictly upper triangular positions masked out)
        if T <= self.max_seq_len:
            causal_mask = self.causal_mask[:T, :T]
        else:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))

        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply optional additional padding mask
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax and attention dropout
        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)

        # Context output: (B, num_heads, T, head_dim)
        context = torch.matmul(probs, v)

        # Transpose and reshape back: (B, T, num_heads * head_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)

        # Output projection and residual dropout
        out = self.out_proj(context)
        out = self.resid_dropout(out)
        return out

