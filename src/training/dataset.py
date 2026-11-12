"""
dataset.py — Tokenized Amharic dataset loader for causal language model pretraining.
"""

import json
from pathlib import Path
from typing import List, Optional, Iterator, Any
import torch
from torch.utils.data import Dataset

from src.tokenizer.morphbpe import MorphBPETokenizer


class AmharicTextDataset(Dataset):
    """
    Fixed-window token dataset for autoregressive causal language modeling.
    Takes token sequences, chunks them into non-overlapping windows of `seq_len`,
    and produces (input_ids, target_ids) pairs shifted by 1 position.
    """

    def __init__(
        self,
        token_ids: torch.Tensor,
        seq_len: int = 256,
    ):
        self.seq_len = seq_len
        # Ensure 1D tensor
        if token_ids.dim() > 1:
            token_ids = token_ids.view(-1)

        # Truncate to multiple of seq_len
        num_tokens = (len(token_ids) // seq_len) * seq_len
        self.data = token_ids[:num_tokens].view(-1, seq_len)

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> dict:
        tokens = self.data[idx]
        return {
            "input_ids": tokens.clone(),
            "targets": tokens.clone(),
        }


def build_token_cache(
    corpus_jsonl_path: str,
    tokenizer: Any,
    cache_path: str,
    max_documents: Optional[int] = None,
    eos_token_id: int = 2,
) -> torch.Tensor:
    """
    Tokenize corpus JSONL and cache as flat 1D torch.Tensor.
    If cache already exists, load and return it.
    """
    cache_file = Path(cache_path)
    if cache_file.exists():
        print(f"Loading cached tokens from {cache_file}...")
        return torch.load(cache_file, map_location="cpu", weights_only=True)

    print(f"Tokenizing corpus {corpus_jsonl_path} (max_docs={max_documents})...")
    all_tokens: List[int] = []
    count = 0

    with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text", "")
            if not text:
                continue

            doc_tokens = tokenizer.encode(text)
            all_tokens.extend(doc_tokens)
            all_tokens.append(eos_token_id)

            count += 1
            if max_documents and count >= max_documents:
                break
            if count % 5000 == 0:
                print(f"Tokenized {count} documents ({len(all_tokens):,} tokens)...")

    tensor = torch.tensor(all_tokens, dtype=torch.long)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, cache_file)
    print(f"Saved {len(tensor):,} tokens to {cache_file}.")
    return tensor
