"""
monitoring.py — Hardware resource tracking, throughput measurement, and training metrics logging.
"""

import time
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class TrainingMonitor:
    """Tracks training progress, hardware resource utilization, and tokens per second."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            # Initialize empty or clear existing log
            with open(self.log_file, "w", encoding="utf-8") as f:
                pass

        self.start_time = time.time()
        self.last_step_time = time.time()
        self.total_tokens_processed = 0

    def get_memory_usage_mb(self) -> float:
        """Get current process RSS memory in MB."""
        if _PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        return 0.0

    def log_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        lr: float,
        tokens_in_batch: int,
        grad_norm: float = 0.0,
        val_loss: Optional[float] = None,
        val_ppl: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Record metrics for an optimization step."""
        now = time.time()
        step_elapsed = now - self.last_step_time
        total_elapsed = now - self.start_time

        self.total_tokens_processed += tokens_in_batch
        tokens_per_sec = (tokens_in_batch / step_elapsed) if step_elapsed > 0 else 0.0
        overall_throughput = (self.total_tokens_processed / total_elapsed) if total_elapsed > 0 else 0.0

        record = {
            "step": step,
            "epoch": epoch,
            "loss": round(loss, 4),
            "lr": round(lr, 8),
            "grad_norm": round(grad_norm, 4),
            "tokens_in_batch": tokens_in_batch,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "overall_tokens_per_sec": round(overall_throughput, 1),
            "total_tokens_processed": self.total_tokens_processed,
            "ram_rss_mb": round(self.get_memory_usage_mb(), 1),
            "elapsed_seconds": round(total_elapsed, 2),
        }

        if val_loss is not None:
            record["val_loss"] = round(val_loss, 4)
        if val_ppl is not None:
            record["val_ppl"] = round(val_ppl, 4)

        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.last_step_time = now
        return record

