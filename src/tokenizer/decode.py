"""
src/tokenizer/decode.py
CLI and library interface to decode token IDs back to Amharic text using MorphBPE.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer.morphbpe import MorphBPETokenizer


def decode_ids(model_dir: str, token_ids: list[int]) -> str:
    tokenizer = MorphBPETokenizer.load(model_dir)
    return tokenizer.decode(token_ids)


def main():
    parser = argparse.ArgumentParser(description="Decode token IDs using MorphBPE")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to trained tokenizer directory")
    parser.add_argument("--ids", type=int, nargs="+", required=True, help="List of integer token IDs")
    args = parser.parse_args()

    text = decode_ids(args.model_dir, args.ids)
    print("Decoded Text:", text)


if __name__ == "__main__":
    main()

