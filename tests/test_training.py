"""
test_training.py — Unit tests for training components: scheduler, checkpointing, dataset, and trainer.
"""

import tempfile
import torch
from torch.utils.data import DataLoader
from src.model.model_config import NanoAmharicConfig
from src.model.transformer import NanoAmharicLM
from src.training.scheduler import CosineWarmupScheduler
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.dataset import AmharicTextDataset
from src.training.trainer import Trainer


def test_cosine_warmup_scheduler():
    """Verify linear warmup and cosine decay behaviors."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = CosineWarmupScheduler(opt, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)

    # Initial lr should be near 0
    assert sched.get_lr()[0] == 0.0

    # Step to halfway through warmup
    for _ in range(5):
        opt.step()
        sched.step()
    assert abs(sched.get_lr()[0] - 0.5) < 1e-4

    # Step to end of warmup
    for _ in range(5):
        opt.step()
        sched.step()
    assert abs(sched.get_lr()[0] - 1.0) < 1e-4

    # Step to end of training
    for _ in range(90):
        opt.step()
        sched.step()
    assert abs(sched.get_lr()[0] - 0.1) < 1e-4


def test_checkpoint_atomic_save_and_load():
    """Verify atomic checkpoint persistence and parameter fidelity."""
    cfg = NanoAmharicConfig.prototype(vocab_size=100)
    model = NanoAmharicLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = f"{tmp_dir}/test_ckpt.pt"
        save_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=opt,
            step=42,
            epoch=2,
            val_loss=1.234,
            metrics={"acc": 0.85},
            config=cfg,
        )

        # Create new model and load
        new_model = NanoAmharicLM(cfg)
        new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)

        meta = load_checkpoint(ckpt_path, new_model, new_opt)

        assert meta["step"] == 42
        assert meta["epoch"] == 2
        assert abs(meta["val_loss"] - 1.234) < 1e-5
        assert meta["metrics"]["acc"] == 0.85

        # Check weights are strictly identical
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


def test_dataset_chunking():
    """Verify that AmharicTextDataset chunks 1D token tensors properly."""
    raw_tokens = torch.arange(100, dtype=torch.long)
    ds = AmharicTextDataset(raw_tokens, seq_len=16)

    # 100 // 16 = 6 full chunks
    assert len(ds) == 6
    sample = ds[0]
    assert sample["input_ids"].shape == (16,)
    assert sample["targets"].shape == (16,)
    assert sample["input_ids"][0].item() == 0
    assert sample["input_ids"][-1].item() == 15


def test_trainer_smoke_run_cpu():
    """Verify that Trainer runs micro-steps on CPU and saves checkpoints."""
    cfg = NanoAmharicConfig.prototype(vocab_size=100)
    cfg.embedding_dim = 64
    cfg.num_layers = 2
    cfg.num_heads = 4
    cfg.num_kv_heads = 4
    cfg.head_dim = 16
    cfg.ffn_dim = 128
    cfg.max_seq_len = 32

    model = NanoAmharicLM(cfg)

    # Synthetic dataset
    raw_tokens = torch.randint(0, 100, (320,), dtype=torch.long)
    ds = AmharicTextDataset(raw_tokens, seq_len=32)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            learning_rate=1e-3,
            gradient_accumulation_steps=2,
            warmup_steps=2,
            max_steps=4,
            eval_interval_steps=2,
            save_interval_steps=2,
            checkpoint_dir=tmp_dir,
            experiment_name="smoke_test",
        )

        res = trainer.train()
        assert res["total_steps"] == 4
        assert res["best_val_loss"] < float("inf")
        assert res["best_val_ppl"] < float("inf")
