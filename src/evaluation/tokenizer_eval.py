"""
src/evaluation/tokenizer_eval.py
Rigorous intrinsic evaluation suite for subword tokenizers.

Evaluates:
1. Fertility (tokens per whitespace-delimited word): F = N_tokens / N_words (lower is better)
2. Compression Ratio (characters per token): C = N_chars / N_tokens (higher is better)
3. Morphological Boundary Consistency (MBC): % of true morpheme boundaries preserved by subword boundaries
4. Unknown Token Rate: % of tokens mapped to <unk> (must be 0.0% with byte fallback)
5. Vocabulary Utilization: % of learned vocabulary active on held-out test data
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer.morphbpe import MorphBPETokenizer
from src.tokenizer.morph_rules import identify_morphological_boundaries


def evaluate_tokenizer_on_samples(
    tokenizer: MorphBPETokenizer,
    sample_texts: List[str],
) -> Dict[str, float]:
    """
    Compute intrinsic evaluation metrics on a list of Amharic texts.
    """
    total_words = 0
    total_tokens = 0
    total_chars = 0
    total_unk_tokens = 0

    active_vocab: Set[int] = set()

    # Morphological boundary consistency tracking
    total_gold_boundaries = 0
    matched_gold_boundaries = 0

    for text in sample_texts:
        words = text.strip().split()
        if not words:
            continue

        for word in words:
            total_words += 1
            total_chars += len(word)

            # Tokenize word
            subwords = tokenizer._encode_word(word)
            total_tokens += len(subwords)

            # Check UNK
            for sub in subwords:
                tid = tokenizer.vocab.get(sub)
                if tid is not None:
                    active_vocab.add(tid)
                else:
                    total_unk_tokens += 1

            # Check Morphological Boundary Consistency (MBC)
            gold_bounds = identify_morphological_boundaries(word)
            if gold_bounds:
                total_gold_boundaries += len(gold_bounds)
                # Subword boundary character offsets
                sub_bounds = set()
                curr_offset = 0
                for s in subwords[:-1]:
                    curr_offset += len(s)
                    sub_bounds.add(curr_offset)

                matched_gold_boundaries += len(gold_bounds.intersection(sub_bounds))

    # Metric computations
    fertility = round(total_tokens / max(1, total_words), 3)
    compression = round(total_chars / max(1, total_tokens), 3)
    unk_rate = round((total_unk_tokens / max(1, total_tokens)) * 100, 4)
    mbc_rate = round((matched_gold_boundaries / max(1, total_gold_boundaries)) * 100, 2)
    vocab_utilization = round((len(active_vocab) / max(1, tokenizer.vocab_size)) * 100, 2)

    return {
        "fertility_tokens_per_word": fertility,
        "compression_chars_per_token": compression,
        "morphological_boundary_consistency_pct": mbc_rate,
        "unknown_token_rate_pct": unk_rate,
        "vocab_utilization_pct": vocab_utilization,
        "total_words_evaluated": total_words,
        "total_tokens_evaluated": total_tokens,
        "total_chars_evaluated": total_chars,
        "total_gold_boundaries_tested": total_gold_boundaries,
    }


def run_evaluation_suite(
    tokenizer_dir: Path,
    test_jsonl_path: Path,
    max_eval_docs: int = 1000,
) -> Dict[str, any]:
    tokenizer = MorphBPETokenizer.load(tokenizer_dir)

    # Load test texts
    test_texts = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_eval_docs and idx >= max_eval_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                t = rec.get("text", "").strip()
                if t:
                    test_texts.append(t)
            except json.JSONDecodeError:
                continue

    results = evaluate_tokenizer_on_samples(tokenizer, test_texts)
    results["model_dir"] = str(tokenizer_dir)
    results["vocab_size"] = tokenizer.vocab_size
    results["boundary_lambda"] = tokenizer.boundary_lambda
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MorphBPE Tokenizer on Test Corpus")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--test-file", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "test.jsonl"))
    parser.add_argument("--max-docs", type=int, default=1000)
    args = parser.parse_args()

    res = run_evaluation_suite(Path(args.model_dir), Path(args.test_file), max_eval_docs=args.max_docs)
    print(json.dumps(res, indent=2))

