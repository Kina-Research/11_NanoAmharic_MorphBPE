"""
checkpoint.py — Atomic checkpoint saving and loading for NanoAmharic.
"""

import os
import tempfile
import torch
from pathlib import Path
from typing import Dict, Any, Optional


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    epoch: int = 0,
    val_loss: float = float("inf"),
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
) -> None:
    """
    Atomically save model and optimizer state.
    Writes to temporary file first, then atomic rename to prevent corrupt files.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss,
        "metrics": metrics or {},
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state["scheduler_state_dict"] = scheduler.state_dict()
    if config is not None:
        state["config"] = config.__dict__ if hasattr(config, "__dict__") else config

    temp_dir = path.parent
    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".tmp") as tmp_file:
        tmp_name = tmp_file.name
        torch.save(state, tmp_name)

    # Atomic rename (on Windows os.replace is atomic if on same drive)
    os.replace(tmp_name, str(path))


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Load model state and return checkpoint metadata.
    """
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return {
        "step": state.get("step", 0),
        "epoch": state.get("epoch", 0),
        "val_loss": state.get("val_loss", float("inf")),
        "metrics": state.get("metrics", {}),
        "config": state.get("config", None),
    }

