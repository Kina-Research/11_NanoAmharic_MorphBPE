"""
src/tokenizer/train_tokenizer.py
Train MorphBPE and Standard BPE tokenizer configurations.

Usage:
  python src/tokenizer/train_tokenizer.py --vocab-size 4000 --lambda-penalty 0.50 --output-dir eval/results/tokenizer/morphbpe_v4k_l050
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer.morphbpe import MorphBPETokenizer


def load_corpus_texts(jsonl_path: Path, max_docs: int = 30000) -> list[str]:
    """Extract clean text strings from JSONL corpus."""
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_docs and idx >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                t = rec.get("text", "").strip()
                if t:
                    texts.append(t)
            except json.JSONDecodeError:
                continue
    return texts


def train_and_save(
    corpus_path: Path,
    vocab_size: int = 4000,
    boundary_lambda: float = 0.50,
    output_dir: Path = None,
    max_docs: int = 30000,
    verbose: bool = True,
) -> dict:
    start_time = time.time()
    if verbose:
        print(f"Loading training data from: {corpus_path}")

    texts = load_corpus_texts(corpus_path, max_docs=max_docs)
    if verbose:
        print(f"Loaded {len(texts):,} documents ({sum(len(t.split()) for t in texts):,} words).")

    tokenizer = MorphBPETokenizer(boundary_lambda=boundary_lambda)
    tokenizer.train(texts, target_vocab_size=vocab_size, verbose=verbose)

    if output_dir:
        output_dir = Path(output_dir)
        tokenizer.save(output_dir)
        if verbose:
            print(f"Saved tokenizer artifacts to: {output_dir}")

    elapsed = time.time() - start_time
    stats = {
        "vocab_size": tokenizer.vocab_size,
        "boundary_lambda": boundary_lambda,
        "num_merges": len(tokenizer.merges),
        "training_time_seconds": round(elapsed, 2),
        "output_dir": str(output_dir) if output_dir else None,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Train MorphBPE Tokenizer on AmharicGenCorpus")
    parser.add_argument("--corpus", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "train.jsonl"))
    parser.add_argument("--vocab-size", type=int, default=4000)
    parser.add_argument("--lambda-penalty", type=float, default=0.50)
    parser.add_argument("--max-docs", type=int, default=30000)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "eval" / "results" / "tokenizer" / f"morphbpe_v{args.vocab_size}_l{int(args.lambda_penalty*100):03d}"
    )

    train_and_save(
        corpus_path=Path(args.corpus),
        vocab_size=args.vocab_size,
        boundary_lambda=args.lambda_penalty,
        output_dir=out_dir,
        max_docs=args.max_docs,
        verbose=True,
    )


if __name__ == "__main__":
    main()

