"""
test_model.py — Unit and property tests for NanoAmharic decoder-only transformer.
Verifies CPU compatibility, causal masking, RoPE embeddings, parameter budgets, and gradient flows.
"""

import pytest
import torch
from src.model.model_config import NanoAmharicConfig
from src.model.transformer import NanoAmharicLM, RMSNorm
from src.model.attention import RotaryEmbedding, CausalSelfAttention, rotate_half


def test_parameter_counts_under_budget():
    """Verify that all configurations respect the <= 50M parameter ceiling."""
    proto_cfg = NanoAmharicConfig.prototype(vocab_size=4000)
    proto_model = NanoAmharicLM(proto_cfg)
    proto_stats = proto_model.count_parameters()
    # Prototype should be between 3M and 10M
    assert 3_000_000 <= proto_stats["trainable_params"] <= 10_000_000, f"Prototype params {proto_stats['trainable_params']} outside expected range"
    assert proto_stats["trainable_params"] <= 50_000_000

    std_cfg = NanoAmharicConfig.standard(vocab_size=4000)
    std_model = NanoAmharicLM(std_cfg)
    std_stats = std_model.count_parameters()
    assert 10_000_000 <= std_stats["trainable_params"] <= 30_000_000
    assert std_stats["trainable_params"] <= 50_000_000

    large_cfg = NanoAmharicConfig.large(vocab_size=8000)
    large_model = NanoAmharicLM(large_cfg)
    large_stats = large_model.count_parameters()
    assert large_stats["trainable_params"] <= 50_000_000, f"Large model exceeded 50M: {large_stats['trainable_params']}"


def test_weight_tying():
    """Verify that LM head and token embeddings share the exact same weight tensor."""
    cfg = NanoAmharicConfig.prototype(vocab_size=4000)
    model = NanoAmharicLM(cfg)
    assert model.lm_head.weight is model.token_embeddings.weight


def test_rope_rotation_properties():
    """Verify RoPE mathematical rotation properties."""
    dim = 32
    seq_len = 16
    rope = RotaryEmbedding(dim=dim, max_seq_len=64)
    cos, sin = rope(seq_len, device=torch.device("cpu"))

    assert cos.shape == (seq_len, dim)
    assert sin.shape == (seq_len, dim)

    # Position 0 cos should be 1, sin should be 0
    assert torch.allclose(cos[0], torch.ones(dim), atol=1e-5)
    assert torch.allclose(sin[0], torch.zeros(dim), atol=1e-5)

    # Test rotate_half preserves L2 norm
    x = torch.randn(2, 4, seq_len, dim)
    x_rot = rotate_half(x)
    assert torch.allclose(x.norm(dim=-1), x_rot.norm(dim=-1), atol=1e-5)


def test_causal_mask_strictly_prevents_lookahead():
    """
    Verify that earlier token representations are completely invariant
    to alterations of subsequent/future tokens in the sequence.
    """
    cfg = NanoAmharicConfig.prototype(vocab_size=4000)
    cfg.dropout = 0.0  # disable dropout for deterministic comparison
    model = NanoAmharicLM(cfg)
    model.eval()

    # Sequence A: [10, 20, 30, 40]
    seq_a = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    # Sequence B: [10, 20, 99, 99] (identical prefix of length 2)
    seq_b = torch.tensor([[10, 20, 99, 99]], dtype=torch.long)

    with torch.no_grad():
        logits_a, _ = model(seq_a)
        logits_b, _ = model(seq_b)

    # Predictions for token at position 0 and position 1 MUST be strictly identical
    assert torch.allclose(logits_a[:, :2, :], logits_b[:, :2, :], atol=1e-5), (
        "Causal mask failed: earlier tokens were affected by changes in later tokens!"
    )


def test_forward_pass_and_loss_computation():
    """Verify forward pass outputs and next-token autoregressive cross-entropy calculation."""
    cfg = NanoAmharicConfig.prototype(vocab_size=4000)
    model = NanoAmharicLM(cfg)

    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 4000, (batch_size, seq_len), dtype=torch.long)

    # Inference forward
    logits, loss = model(input_ids)
    assert logits.shape == (batch_size, seq_len, 4000)
    assert loss is None

    # Training forward with targets
    logits, loss = model(input_ids, targets=input_ids)
    assert logits.shape == (batch_size, seq_len, 4000)
    assert loss is not None
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0.0


def test_cpu_backward_gradient_flow():
    """Verify that a full backward step on CPU produces finite gradients across all layers."""
    cfg = NanoAmharicConfig.prototype(vocab_size=4000)
    model = NanoAmharicLM(cfg)
    model.train()

    input_ids = torch.randint(0, 4000, (2, 8), dtype=torch.long)
    _, loss = model(input_ids, targets=input_ids)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradient"


def test_generation_runs_on_cpu():
    """Verify autoregressive generation runs cleanly on CPU."""
    cfg = NanoAmharicConfig.prototype(vocab_size=4000)
    model = NanoAmharicLM(cfg)

    prompt = torch.tensor([[10, 20, 30]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=10)

    assert generated.shape == (1, 8)
    assert (generated[:, :3] == prompt).all(), "Prompt was not preserved in generated output"

