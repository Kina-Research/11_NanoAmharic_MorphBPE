"""
scheduler.py — Learning rate scheduler with linear warmup and cosine annealing.
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """
    Linear warmup for warmup_steps, followed by cosine decay down to min_lr.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(0, self.last_epoch)

        if step < self.warmup_steps:
            # Linear warmup: scale from 0 to 1
            scale = float(step) / float(self.warmup_steps)
        elif step >= self.total_steps:
            # Floor at min_lr_ratio
            scale = self.min_lr_ratio
        else:
            # Cosine decay between 1.0 and min_lr_ratio
            decay_ratio = float(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            scale = self.min_lr_ratio + coeff * (1.0 - self.min_lr_ratio)

        return [base_lr * scale for base_lr in self.base_lrs]

