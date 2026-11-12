"""
embeddings.py — Token embedding representation for NanoAmharic.
"""

import torch
import torch.nn as nn


class NanoEmbedding(nn.Module):
    """Token embedding layer with optional padding index."""

    def __init__(self, vocab_size: int, embedding_dim: int, pad_token_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.weight = nn.Parameter(torch.empty(vocab_size, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if pad_token_id is not None and 0 <= pad_token_id < vocab_size:
            with torch.no_grad():
                self.weight[pad_token_id].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch_size, seq_len)
        Returns: (batch_size, seq_len, embedding_dim)
        """
        return nn.functional.embedding(input_ids, self.weight, padding_idx=self.pad_token_id)

