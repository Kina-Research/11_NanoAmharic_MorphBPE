"""
model_config.py — Configuration dataclass for NanoAmharic decoder-only transformer.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NanoAmharicConfig:
    """Configuration for NanoAmharic decoder-only transformer."""

    name: str = "NanoAmharic"
    vocab_size: int = 4000
    embedding_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: Optional[int] = 8  # Set to 1 for MQA, num_heads for MHA, or intermediate for GQA
    head_dim: int = 32               # embedding_dim // num_heads
    ffn_dim: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    tie_embeddings: bool = True
    rmsnorm_eps: float = 1e-6
    rope_base: float = 10000.0
    activation: str = "gelu"
    pad_token_id: int = 0
    eos_token_id: int = 2

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.embedding_dim // self.num_heads
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

    @classmethod
    def prototype(cls, vocab_size: int = 4000) -> "NanoAmharicConfig":
        """
        Prototype configuration (~5-8M parameters).
        Designed for CPU-only pretraining and fast validation.
        """
        return cls(
            name="NanoAmharic-Prototype",
            vocab_size=vocab_size,
            embedding_dim=256,
            num_layers=6,
            num_heads=8,
            num_kv_heads=8,
            head_dim=32,
            ffn_dim=1024,
            max_seq_len=256,
            dropout=0.1,
            tie_embeddings=True,
            rmsnorm_eps=1e-6,
            rope_base=10000.0,
            activation="gelu",
        )

    @classmethod
    def standard(cls, vocab_size: int = 4000) -> "NanoAmharicConfig":
        """
        Standard configuration (~15-20M parameters).
        """
        return cls(
            name="NanoAmharic-Standard",
            vocab_size=vocab_size,
            embedding_dim=384,
            num_layers=8,
            num_heads=8,
            num_kv_heads=8,
            head_dim=48,
            ffn_dim=1536,
            max_seq_len=512,
            dropout=0.1,
            tie_embeddings=True,
            rmsnorm_eps=1e-6,
            rope_base=10000.0,
            activation="gelu",
        )

    @classmethod
    def large(cls, vocab_size: int = 8000) -> "NanoAmharicConfig":
        """
        Large configuration (~40-50M parameters <= 50M hard cap).
        """
        return cls(
            name="NanoAmharic-Large",
            vocab_size=vocab_size,
            embedding_dim=512,
            num_layers=12,
            num_heads=8,
            num_kv_heads=8,
            head_dim=64,
            ffn_dim=2048,
            max_seq_len=512,
            dropout=0.1,
            tie_embeddings=True,
            rmsnorm_eps=1e-6,
            rope_base=10000.0,
            activation="gelu",
        )

