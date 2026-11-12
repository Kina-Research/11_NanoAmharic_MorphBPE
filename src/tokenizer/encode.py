"""
src/tokenizer/encode.py
CLI and library interface to encode Amharic text using a trained MorphBPE model.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer.morphbpe import MorphBPETokenizer


def encode_text(model_dir: str, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    tokenizer = MorphBPETokenizer.load(model_dir)
    return tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)


def main():
    parser = argparse.ArgumentParser(description="Encode Amharic text using MorphBPE")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to trained tokenizer directory")
    parser.add_argument("--text", type=str, required=True, help="Input Amharic text to encode")
    parser.add_argument("--add-bos", action="store_true", help="Prepend <bos> token ID")
    parser.add_argument("--add-eos", action="store_true", help="Append <eos> token ID")
    args = parser.parse_args()

    ids = encode_text(args.model_dir, args.text, add_bos=args.add_bos, add_eos=args.add_eos)
    print("Token IDs:", ids)
    print("Token Count:", len(ids))


if __name__ == "__main__":
    main()

