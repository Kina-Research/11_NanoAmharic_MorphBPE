"""
trainer.py — CPU-optimized training loop with gradient accumulation, clipping, and evaluation.
"""

import math
import time
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.transformer import NanoAmharicLM
from src.training.scheduler import CosineWarmupScheduler
from src.training.checkpoint import save_checkpoint
from src.training.monitoring import TrainingMonitor


class Trainer:
    """
    Orchestrates CPU pretraining of NanoAmharic language models.
    """

    def __init__(
        self,
        model: NanoAmharicLM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.95,
        gradient_clip_norm: float = 1.0,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        max_steps: int = 1000,
        eval_interval_steps: int = 100,
        save_interval_steps: int = 250,
        checkpoint_dir: str = "eval/results/checkpoints/",
        experiment_name: str = "nanoamharic_experiment",
        log_file: Optional[str] = None,
        seed: int = 42,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.gradient_clip_norm = gradient_clip_norm
        self.max_steps = max_steps
        self.eval_interval_steps = eval_interval_steps
        self.save_interval_steps = save_interval_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name

        # Ensure reproducibility
        torch.manual_seed(seed)

        # Optimizer: apply weight decay only to 2D weight tensors (linear layers), not biases or 1D norms
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=1e-8,
        )

        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=max_steps,
            min_lr_ratio=0.1,
        )

        self.monitor = TrainingMonitor(log_file=log_file)
        self.best_val_loss = float("inf")
        self.best_val_ppl = float("inf")
        self.global_step = 0
        self.epoch = 0

    def evaluate(self, max_batches: int = 50) -> Dict[str, float]:
        """
        Evaluate validation loss and perplexity on CPU.
        """
        if self.val_loader is None:
            return {"val_loss": 0.0, "val_ppl": 0.0}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= max_batches:
                    break
                input_ids = batch["input_ids"]
                targets = batch["targets"]

                _, loss = self.model(input_ids, targets=targets)
                if loss is not None:
                    tokens_in_batch = input_ids.numel()
                    total_loss += loss.item() * tokens_in_batch
                    total_tokens += tokens_in_batch

        avg_loss = (total_loss / total_tokens) if total_tokens > 0 else float("inf")
        ppl = math.exp(min(avg_loss, 20.0))  # clamp to avoid overflow
        self.model.train()

        return {
            "val_loss": round(avg_loss, 4),
            "val_ppl": round(ppl, 4),
        }

    def train(self) -> Dict[str, Any]:
        """
        Execute training loop on CPU until max_steps.
        """
        print(f"[{self.experiment_name}] Starting training: max_steps={self.max_steps}, grad_accum={self.gradient_accumulation_steps}")
        self.model.train()

        accum_loss = 0.0
        accum_tokens = 0
        step_start_time = time.time()

        train_iter = iter(self.train_loader)

        while self.global_step < self.max_steps:
            self.optimizer.zero_grad()
            batch_loss_val = 0.0

            for micro_step in range(self.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    self.epoch += 1
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                input_ids = batch["input_ids"]
                targets = batch["targets"]

                # Forward pass
                _, loss = self.model(input_ids, targets=targets)
                # Scale loss by accumulation factor
                scaled_loss = loss / self.gradient_accumulation_steps
                scaled_loss.backward()

                batch_loss_val += loss.item() / self.gradient_accumulation_steps
                accum_tokens += input_ids.numel()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip_norm,
            ).item()

            # Optimizer and scheduler steps
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            current_lr = self.scheduler.get_lr()[0]

            # Periodic validation
            val_metrics = None
            if self.global_step % self.eval_interval_steps == 0 or self.global_step == self.max_steps:
                val_metrics = self.evaluate()
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.best_val_ppl = val_metrics["val_ppl"]
                    save_checkpoint(
                        checkpoint_path=str(self.checkpoint_dir / f"{self.experiment_name}_best.pt"),
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=self.global_step,
                        epoch=self.epoch,
                        val_loss=self.best_val_loss,
                        metrics=val_metrics,
                        config=self.model.config,
                    )

            # Log metrics
            record = self.monitor.log_step(
                step=self.global_step,
                epoch=self.epoch,
                loss=batch_loss_val,
                lr=current_lr,
                tokens_in_batch=accum_tokens,
                grad_norm=grad_norm,
                val_loss=val_metrics["val_loss"] if val_metrics else None,
                val_ppl=val_metrics["val_ppl"] if val_metrics else None,
            )
            accum_tokens = 0

            # Console output
            if self.global_step % max(1, self.eval_interval_steps // 2) == 0 or self.global_step == self.max_steps:
                val_info = f" | val_loss={val_metrics['val_loss']:.4f} val_ppl={val_metrics['val_ppl']:.2f}" if val_metrics else ""
                print(
                    f"[{self.experiment_name}] Step {self.global_step:4d}/{self.max_steps} | "
                    f"Loss: {batch_loss_val:.4f} | LR: {current_lr:.2e} | "
                    f"GradNorm: {grad_norm:.2f} | {record['tokens_per_sec']:.0f} tok/s | "
                    f"RSS: {record['ram_rss_mb']:.0f}MB{val_info}"
                )

            # Periodic checkpoint saving
            if self.global_step % self.save_interval_steps == 0 or self.global_step == self.max_steps:
                save_checkpoint(
                    checkpoint_path=str(self.checkpoint_dir / f"{self.experiment_name}_last.pt"),
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=self.global_step,
                    epoch=self.epoch,
                    val_loss=val_metrics["val_loss"] if val_metrics else batch_loss_val,
                    metrics={"loss": batch_loss_val, "best_val_loss": self.best_val_loss},
                    config=self.model.config,
                )

        final_val = self.evaluate()
        print(f"[{self.experiment_name}] Training finished. Best Val Loss: {self.best_val_loss:.4f}, Best Val PPL: {self.best_val_ppl:.2f}")

        return {
            "experiment_name": self.experiment_name,
            "total_steps": self.global_step,
            "final_epoch": self.epoch,
            "final_train_loss": batch_loss_val,
            "best_val_loss": self.best_val_loss,
            "best_val_ppl": self.best_val_ppl,
            "final_val_loss": final_val["val_loss"],
            "final_val_ppl": final_val["val_ppl"],
            "total_tokens_processed": self.monitor.total_tokens_processed,
            "elapsed_seconds": time.time() - self.monitor.start_time,
        }

