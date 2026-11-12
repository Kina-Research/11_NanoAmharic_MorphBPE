"""
initialization.py — Weight initialization for NanoAmharic decoder-only transformer.
"""

import math
import torch
import torch.nn as nn


def init_weights(module: nn.Module, n_layers: int = 6, init_std: float = 0.02) -> None:
    """
    Initialize transformer weights:
    - Linear layers: N(0, init_std)
    - Output projection layers: scaled by 1 / sqrt(2 * n_layers) for residual stability
    - Embedding layers: N(0, init_std)
    - RMSNorm weights: initialized to 1.0
    - Biases: initialized to 0.0
    """
    if isinstance(module, nn.Linear):
        # Base initialization
        torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
        if hasattr(module, "_is_residual_proj") and module._is_residual_proj:
            # Scaled initialization for residual projections (GPT-2 style)
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std / math.sqrt(2 * n_layers))
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

