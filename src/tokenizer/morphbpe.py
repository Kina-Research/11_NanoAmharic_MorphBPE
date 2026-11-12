"""
src/tokenizer/morphbpe.py
Morphology-Constrained Byte Pair Encoding (MorphBPE) Tokenizer.

Implements:
1. Boundary penalty scoring: S(p, q) = f(p, q) * (1 - lambda * b(p, q))
2. Deterministic alphabetical tie-breaking: ties broken by sorted (p, q).
3. 256-byte fallback for 100% lossless UTF-8 round-trip coverage (0% UNK).
4. Configurable special tokens (<pad>, <unk>, <bos>, <eos>, <story_start>, <story_end>).
5. O(len) rank-based subword encoding and exact decoding.
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.tokenizer.morph_rules import (
    identify_morphological_boundaries,
    crosses_morphological_boundary,
)

# Standard special tokens
SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<story_start>",
    "<story_end>",
]


def byte_to_token(b: int) -> str:
    """Format an integer byte (0-255) as a byte token string."""
    return f"<0x{b:02X}>"


def token_to_byte(token: str) -> Optional[int]:
    """Parse a byte token string '<0xHH>' back to its integer byte value."""
    if token.startswith("<0x") and token.endswith(">") and len(token) == 6:
        try:
            return int(token[3:5], 16)
        except ValueError:
            return None
    return None


class MorphBPETokenizer:
    """
    Morphology-Constrained Byte Pair Encoding Tokenizer for Amharic.
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        boundary_lambda: float = 0.5,
    ):
        self.boundary_lambda = float(boundary_lambda)
        self.special_tokens = list(SPECIAL_TOKENS)

        # Vocabulary maps
        self.vocab = vocab or {}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = merges or []
        self.merge_ranks = {pair: idx for idx, pair in enumerate(self.merges)}

        # Memoization cache for encoding
        self._encode_cache: Dict[str, List[str]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def unk_token_id(self) -> int:
        return self.vocab["<unk>"]

    @property
    def bos_token_id(self) -> int:
        return self.vocab["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.vocab["<eos>"]

    # ─────────────────────────────────────────────────────────────────────────
    # Training Pipeline (Fast Inverted-Index Delta Updates)
    # ─────────────────────────────────────────────────────────────────────────
    def train(
        self,
        corpus_texts: List[str],
        target_vocab_size: int = 4000,
        min_frequency: int = 2,
        verbose: bool = True,
    ):
        """
        Train MorphBPE on a list of Amharic text strings with inverted-index delta updates.
        """
        if verbose:
            print(f"Training MorphBPE: target_vocab={target_vocab_size:,}, lambda={self.boundary_lambda}", flush=True)

        # 1. Initialize base vocabulary: special tokens + 256 byte tokens
        vocab: Dict[str, int] = {}
        for idx, tok in enumerate(self.special_tokens):
            vocab[tok] = idx

        # Add all 256 single-byte fallback tokens
        for b in range(256):
            btok = byte_to_token(b)
            if btok not in vocab:
                vocab[btok] = len(vocab)

        # 2. Build word frequency table from corpus
        raw_word_freqs = Counter()
        for text in corpus_texts:
            for word in text.strip().split():
                if word:
                    raw_word_freqs[word] += 1

        # Prune hapax legomena for efficient merge indexing
        if min_frequency > 1:
            filtered_freqs = {w: c for w, c in raw_word_freqs.items() if c >= min_frequency}
            if len(filtered_freqs) < 200:
                filtered_freqs = dict(raw_word_freqs)
        else:
            filtered_freqs = dict(raw_word_freqs)

        if verbose:
            print(f"  Unique words in training set: {len(raw_word_freqs):,} (indexed: {len(filtered_freqs):,})", flush=True)

        # 3. Represent words as list of tokens + precomputed boundary positions
        words_list = []
        words_tokens = []
        words_counts = []
        words_boundaries = []

        for word, freq in filtered_freqs.items():
            chars = list(word)
            if not chars:
                continue
            for ch in chars:
                if ch not in vocab:
                    vocab[ch] = len(vocab)
            bounds = identify_morphological_boundaries(word)
            words_list.append(word)
            words_tokens.append(chars)
            words_counts.append(freq)
            words_boundaries.append(bounds)

        # 4. Initialize pair statistics and inverted index
        # pair_stats: pair -> [total_freq, boundary_crossings]
        pair_stats: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
        pair_to_words: Dict[Tuple[str, str], Set[int]] = defaultdict(set)

        for w_idx in range(len(words_tokens)):
            toks = words_tokens[w_idx]
            bounds = words_boundaries[w_idx]
            freq = words_counts[w_idx]

            curr_offset = 0
            for i in range(len(toks) - 1):
                pair = (toks[i], toks[i + 1])
                pair_stats[pair][0] += freq
                if crosses_morphological_boundary(toks[i], toks[i + 1], bounds, curr_offset):
                    pair_stats[pair][1] += freq
                pair_to_words[pair].add(w_idx)
                curr_offset += len(toks[i])

        merges: List[Tuple[str, str]] = []
        num_merges_target = max(0, target_vocab_size - len(vocab))
        if verbose:
            print(f"  Base vocabulary size: {len(vocab):,}", flush=True)
            print(f"  Target merges to learn: {num_merges_target:,}", flush=True)

        # 5. Fast Merge Loop
        step = 0
        while len(vocab) < target_vocab_size and pair_stats:
            # Find the best pair according to score: freq * (1 - lambda * b_ratio)
            # Deterministic tie-breaking: (score, freq, -pair)
            best_pair = None
            best_score = -1.0
            best_freq = 0

            # Scan pair_stats
            for pair, (freq, crossings) in pair_stats.items():
                if freq <= 0:
                    continue
                b_ratio = (crossings / freq) if freq > 0 else 0.0
                score = freq * (1.0 - (self.boundary_lambda * b_ratio))

                if score > best_score or (score == best_score and (freq > best_freq or (freq == best_freq and pair < best_pair))):
                    best_score = score
                    best_freq = freq
                    best_pair = pair

            if not best_pair or best_score <= 0.0 and self.boundary_lambda >= 1.0:
                # If all remaining pairs are penalized, take highest frequency
                for pair, (freq, _) in pair_stats.items():
                    if freq > best_freq:
                        best_freq = freq
                        best_pair = pair
                if not best_pair or best_freq <= 0:
                    break

            p1, p2 = best_pair
            new_token = p1 + p2
            vocab[new_token] = len(vocab)
            merges.append(best_pair)
            step += 1

            if verbose and (step % 500 == 0 or step == 1 or len(vocab) == target_vocab_size):
                print(f"  [Merge {step:05d}/{num_merges_target}] '{p1}' + '{p2}' -> '{new_token}' (freq={best_freq})", flush=True)

            # Update ONLY affected words
            affected_w_indices = list(pair_to_words.get(best_pair, set()))
            # Remove best_pair
            if best_pair in pair_stats:
                del pair_stats[best_pair]
            if best_pair in pair_to_words:
                del pair_to_words[best_pair]

            for w_idx in affected_w_indices:
                toks = words_tokens[w_idx]
                bounds = words_boundaries[w_idx]
                freq = words_counts[w_idx]

                # Decrement counts of old pairs in this word
                curr_offset = 0
                for i in range(len(toks) - 1):
                    old_pair = (toks[i], toks[i + 1])
                    if old_pair in pair_stats:
                        pair_stats[old_pair][0] -= freq
                        if crosses_morphological_boundary(toks[i], toks[i + 1], bounds, curr_offset):
                            pair_stats[old_pair][1] -= freq
                        if pair_stats[old_pair][0] <= 0:
                            del pair_stats[old_pair]
                    if old_pair in pair_to_words:
                        pair_to_words[old_pair].discard(w_idx)
                    curr_offset += len(toks[i])

                # Perform merge
                new_toks = []
                i = 0
                while i < len(toks):
                    if i < len(toks) - 1 and toks[i] == p1 and toks[i + 1] == p2:
                        new_toks.append(new_token)
                        i += 2
                    else:
                        new_toks.append(toks[i])
                        i += 1
                words_tokens[w_idx] = new_toks

                # Increment counts of new pairs in this word
                curr_offset = 0
                for i in range(len(new_toks) - 1):
                    new_pair = (new_toks[i], new_toks[i + 1])
                    pair_stats[new_pair][0] += freq
                    if crosses_morphological_boundary(new_toks[i], new_toks[i + 1], bounds, curr_offset):
                        pair_stats[new_pair][1] += freq
                    pair_to_words[new_pair].add(w_idx)
                    curr_offset += len(new_toks[i])

        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merge_ranks = {pair: idx for idx, pair in enumerate(merges)}
        self._encode_cache.clear()

        if verbose:
            print(f"Training completed. Final vocabulary size: {len(self.vocab):,}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Encoding & Decoding
    # ─────────────────────────────────────────────────────────────────────────
    def _encode_word(self, word: str) -> List[str]:
        """Encode a single whitespace-delimited word using learned merges."""
        if word in self._encode_cache:
            return list(self._encode_cache[word])

        # Initial characters
        subwords = list(word)
        if len(subwords) <= 1:
            self._encode_cache[word] = subwords
            return subwords

        while len(subwords) > 1:
            # Find the pair with the lowest merge rank
            best_pair = None
            best_rank = float("inf")
            best_idx = -1

            for i in range(len(subwords) - 1):
                pair = (subwords[i], subwords[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    best_idx = i

            if best_pair is None:
                break  # No more eligible merges

            # Merge the winning pair
            p1, p2 = best_pair
            new_subwords = []
            i = 0
            while i < len(subwords):
                if i < len(subwords) - 1 and subwords[i] == p1 and subwords[i + 1] == p2:
                    new_subwords.append(p1 + p2)
                    i += 2
                else:
                    new_subwords.append(subwords[i])
                    i += 1
            subwords = new_subwords

        self._encode_cache[word] = subwords
        return subwords

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize an input text into a list of token strings.
        Decomposes out-of-vocabulary characters into UTF-8 byte tokens.
        """
        tokens: List[str] = []
        words = text.split(" ")

        for w_idx, word in enumerate(words):
            if w_idx > 0:
                # Represent space as standard space or special whitespace token
                tokens.append(" ")

            if not word:
                continue

            subwords = self._encode_word(word)
            for sub in subwords:
                if sub in self.vocab:
                    tokens.append(sub)
                else:
                    # Byte fallback for any unknown subword character
                    for b in sub.encode("utf-8"):
                        tokens.append(byte_to_token(b))

        return tokens

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Convert a text string into a list of token IDs."""
        tokens = self.tokenize(text)
        ids = []
        if add_bos:
            ids.append(self.bos_token_id)

        for tok in tokens:
            if tok in self.vocab:
                ids.append(self.vocab[tok])
            elif tok == " ":
                # Space character
                if " " in self.vocab:
                    ids.append(self.vocab[" "])
                else:
                    for b in " ".encode("utf-8"):
                        ids.append(self.vocab[byte_to_token(b)])
            else:
                for b in tok.encode("utf-8"):
                    ids.append(self.vocab.get(byte_to_token(b), self.unk_token_id))

        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token IDs back into an exact string.
        Gracefully reconstructs UTF-8 byte streams.
        """
        byte_buffer = bytearray()
        output_chunks = []

        def flush_bytes():
            if byte_buffer:
                try:
                    output_chunks.append(byte_buffer.decode("utf-8", errors="replace"))
                except Exception:
                    pass
                byte_buffer.clear()

        for tid in token_ids:
            if tid not in self.id_to_token:
                continue
            token_str = self.id_to_token[tid]

            # Ignore non-printable special tokens
            if token_str in ("<pad>", "<unk>", "<bos>", "<eos>"):
                continue

            byte_val = token_to_byte(token_str)
            if byte_val is not None:
                # Accumulate raw bytes
                byte_buffer.append(byte_val)
            else:
                # String token
                flush_bytes()
                output_chunks.append(token_str)

        flush_bytes()
        return "".join(output_chunks)

    # ─────────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────────
    def save(self, output_dir: str | Path):
        """Save vocabulary, merges, and configuration to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save vocab
        vocab_path = out / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save merges
        merges_path = out / "merges.txt"
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: 1.0\n")
            for p1, p2 in self.merges:
                f.write(f"{p1} {p2}\n")

        # Save config
        config_path = out / "tokenizer_config.json"
        config = {
            "algorithm": "MorphBPE",
            "vocab_size": len(self.vocab),
            "boundary_lambda": self.boundary_lambda,
            "special_tokens": self.special_tokens,
            "num_merges": len(self.merges),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, model_dir: str | Path) -> "MorphBPETokenizer":
        """Load a saved tokenizer from a directory."""
        path = Path(model_dir)
        vocab_path = path / "vocab.json"
        merges_path = path / "merges.txt"
        config_path = path / "tokenizer_config.json"

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        merges = []
        if merges_path.exists():
            with open(merges_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))

        boundary_lambda = 0.5
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                boundary_lambda = config.get("boundary_lambda", 0.5)

        return cls(vocab=vocab, merges=merges, boundary_lambda=boundary_lambda)
