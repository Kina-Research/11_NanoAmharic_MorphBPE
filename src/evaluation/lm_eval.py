"""
lm_eval.py — Language model evaluation: Perplexity, Bits-per-Character (BPC), and Word Perplexity.
"""

import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.model_config import NanoAmharicConfig
from src.model.transformer import NanoAmharicLM
from src.tokenizer.morphbpe import MorphBPETokenizer
from src.training.checkpoint import load_checkpoint
from src.training.dataset import AmharicTextDataset


def evaluate_model_on_corpus(
    model: NanoAmharicLM,
    tokenizer: MorphBPETokenizer,
    corpus_jsonl_path: str,
    seq_len: int = 128,
    batch_size: int = 8,
    max_documents: int = 1000,
) -> Dict[str, Any]:
    """
    Evaluates language model cross-entropy loss, token PPL, word PPL, and Bits-Per-Character (BPC).
    Returns aggregate metrics and per-document metric arrays for significance testing.
    """
    import json
    model.eval()

    doc_losses: List[float] = []
    doc_bpcs: List[float] = []

    total_nll = 0.0
    total_tokens = 0
    total_words = 0
    total_chars = 0

    docs_processed = 0

    with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text", "")
            if not text or len(text.strip()) < 10:
                continue

            words = text.split()
            chars_len = len(text)
            tokens = tokenizer.encode(text)
            tokens.append(tokenizer.eos_token_id)

            if len(tokens) < 2:
                continue

            # Process document through model
            input_tensor = torch.tensor([tokens], dtype=torch.long)
            # Truncate to seq_len if document is longer
            if input_tensor.size(1) > seq_len:
                input_tensor = input_tensor[:, :seq_len]

            with torch.no_grad():
                logits, loss = model(input_tensor, targets=input_tensor)

            if loss is not None:
                doc_loss = loss.item()
                num_tok = input_tensor.size(1) - 1
                doc_nll = doc_loss * num_tok

                total_nll += doc_nll
                total_tokens += num_tok
                total_words += len(words)
                total_chars += chars_len

                # BPC for this document
                doc_bpc = (doc_nll / math.log(2)) / max(1, chars_len)
                doc_losses.append(doc_loss)
                doc_bpcs.append(doc_bpc)

            docs_processed += 1
            if docs_processed >= max_documents:
                break

    avg_loss = (total_nll / total_tokens) if total_tokens > 0 else float("inf")
    token_ppl = math.exp(min(avg_loss, 20.0))
    word_loss = (total_nll / total_words) if total_words > 0 else float("inf")
    word_ppl = math.exp(min(word_loss, 20.0))
    bpc = (total_nll / math.log(2)) / total_chars if total_chars > 0 else float("inf")
    char_ppl = math.pow(2, min(bpc, 20.0))

    return {
        "documents_evaluated": docs_processed,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_chars": total_chars,
        "token_cross_entropy": round(avg_loss, 4),
        "token_perplexity": round(token_ppl, 2),
        "word_perplexity": round(word_ppl, 2),
        "bits_per_character": round(bpc, 4),
        "character_perplexity": round(char_ppl, 2),
        "per_doc_bpcs": doc_bpcs,
        "per_doc_losses": doc_losses,
    }

