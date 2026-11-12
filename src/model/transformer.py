"""
transformer.py — NanoAmharic decoder-only transformer architecture.
Features:
- Pre-RMSNorm
- Rotary Position Embeddings (RoPE)
- Causal Self-Attention (MHA/MQA)
- Weight tying between embedding and LM head
- Autoregressive generation with temperature, top-k, and nucleus (top-p) sampling
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model_config import NanoAmharicConfig
from src.model.embeddings import NanoEmbedding
from src.model.attention import CausalSelfAttention
from src.model.initialization import init_weights


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    More computationally efficient on CPU than standard LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS variance across last dimension
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with GELU activation.
    """

    def __init__(self, embedding_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.up_proj = nn.Linear(embedding_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embedding_dim, bias=False)
        self.down_proj._is_residual_proj = True
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.up_proj(x))
        out = self.down_proj(h)
        return self.dropout(out)


class TransformerBlock(nn.Module):
    """
    Single Decoder Block with Pre-RMSNorm residual connections.
    """

    def __init__(self, config: NanoAmharicConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.embedding_dim, eps=config.rmsnorm_eps)
        self.attention = CausalSelfAttention(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            rope_base=config.rope_base,
        )
        self.ffn_norm = RMSNorm(config.embedding_dim, eps=config.rmsnorm_eps)
        self.ffn = FeedForward(
            embedding_dim=config.embedding_dim,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm Self-Attention with residual connection
        x = x + self.attention(self.attn_norm(x), attention_mask=attention_mask)
        # Pre-norm Feed-Forward with residual connection
        x = x + self.ffn(self.ffn_norm(x))
        return x


class NanoAmharicLM(nn.Module):
    """
    NanoAmharic: Morphology-aware, CPU-optimized Decoder-Only Language Model.
    Strictly constrained to <= 50M parameters.
    """

    def __init__(self, config: NanoAmharicConfig):
        super().__init__()
        self.config = config

        # Token Embeddings
        self.token_embeddings = NanoEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            pad_token_id=config.pad_token_id,
        )

        # Transformer Decoder Layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final Layer Normalization
        self.final_norm = RMSNorm(config.embedding_dim, eps=config.rmsnorm_eps)

        # Language Modeling Head
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # Weight tying (Press & Wolf, 2017)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        # Initialize all model weights
        self.apply(lambda m: init_weights(m, n_layers=config.num_layers))

    def count_parameters(self) -> Dict[str, int]:
        """
        Count total, trainable, and non-embedding parameters.
        Returns a dictionary of counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Non-embedding parameters count
        emb_params = self.token_embeddings.weight.numel()
        # If tied, lm_head does not double-count in parameters(), but we reflect unique params
        non_emb_params = total_params - emb_params

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_embedding_params": non_emb_params,
            "tied_embeddings": self.config.tie_embeddings,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        input_ids: (batch_size, seq_len)
        targets: optional (batch_size, seq_len) for autoregressive cross-entropy loss computation
        attention_mask: optional attention mask
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar tensor if targets is provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        # Embedding lookup
        x = self.token_embeddings(input_ids)  # (B, T, embedding_dim)

        # Pass through decoder blocks
        for block in self.layers:
            x = block(x, attention_mask=attention_mask)

        # Final RMSNorm
        x = self.final_norm(x)

        # Project to vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Autoregressive next-token prediction loss
            # Shift so that tokens < t predict token at t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()

            # Flatten batch and sequence dimensions for cross entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        """
        Autoregressive generation on CPU.
        Supports temperature scaling, top-k filtering, and top-p (nucleus) sampling.
        """
        self.eval()
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        curr_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Crop to context window if needed
            cond_ids = curr_ids if curr_ids.size(1) <= self.config.max_seq_len else curr_ids[:, -self.config.max_seq_len :]

            logits, _ = self(cond_ids)
            next_token_logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply repetition penalty if > 1.0
            if repetition_penalty != 1.0:
                for b in range(curr_ids.size(0)):
                    for prev_token in set(curr_ids[b].tolist()):
                        if next_token_logits[b, prev_token] < 0:
                            next_token_logits[b, prev_token] *= repetition_penalty
                        else:
                            next_token_logits[b, prev_token] /= repetition_penalty

            # Apply temperature
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature

                # Top-K filtering
                if top_k > 0:
                    top_k_val = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_val)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                # Top-P (nucleus) filtering
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            # Check if all batches reached EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return curr_ids

