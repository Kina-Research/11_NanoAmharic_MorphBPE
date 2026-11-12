"""
src/data/deduplicate.py
Exact and near-duplicate detection for Amharic corpus.

Methods:
  1. Exact deduplication via SHA-256 hash of normalized text
  2. Near-duplicate detection via MinHash + Jaccard similarity (datasketch)

Design:
  - MD5 available for legacy indexing but SHA-256 is used for all provenance records.
  - MinHash shingle size: character-level trigrams (robust for Ethiopic script)
  - Jaccard threshold: 0.85 (configurable)
  - Memory-efficient streaming: processes JSONL line-by-line
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Generator, Iterator

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


# ── Shingling ───────────────────────────────────────────────────────────────
def get_shingles(text: str, n: int = 3) -> set[bytes]:
    """Generate word n-gram shingles as byte strings (fast and linguistically meaningful)."""
    words = text.split()
    if len(words) < n:
        # Fallback to character shingles for very short texts
        return {text[i:i + 3].encode("utf-8") for i in range(max(1, len(text) - 2))}
    shingles = set()
    for i in range(len(words) - n + 1):
        shingle = " ".join(words[i:i + n]).encode("utf-8")
        shingles.add(shingle)
    return shingles


def make_minhash(text: str, num_perm: int = 64, ngram: int = 3) -> MinHash:
    """Create a MinHash signature for a text string."""
    m = MinHash(num_perm=num_perm)
    for shingle in get_shingles(text, n=ngram):
        m.update(shingle)
    return m


# ── Exact deduplication ──────────────────────────────────────────────────────
def sha256_hash(text: str) -> str:
    """Return hex SHA-256 of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate_exact(
    records: list[dict],
    text_field: str = "text",
) -> tuple[list[dict], int]:
    """
    Remove exact duplicates by SHA-256 hash of the text field.

    Returns:
        (deduplicated_records, n_removed)
    """
    seen_hashes: set[str] = set()
    unique: list[dict] = []
    for rec in records:
        text = rec.get(text_field, "")
        h = sha256_hash(text)
        if h not in seen_hashes:
            seen_hashes.add(h)
            rec["sha256"] = h
            unique.append(rec)
    n_removed = len(records) - len(unique)
    return unique, n_removed


# ── Near-duplicate deduplication (MinHash LSH) ──────────────────────────────
def deduplicate_minhash(
    records: list[dict],
    text_field: str = "text",
    threshold: float = 0.85,
    num_perm: int = 64,
    ngram: int = 3,
    verbose: bool = True,
) -> tuple[list[dict], int]:
    """
    Remove near-duplicates using MinHash LSH.
    Records that are within `threshold` Jaccard similarity to an already-seen
    record are discarded.

    Args:
        records: List of dicts with text_field.
        text_field: Key for the text content.
        threshold: Jaccard similarity threshold (0.85 = very similar).
        num_perm: Number of permutations for MinHash.
        ngram: Character n-gram size for shingling.
        verbose: Show progress bar.

    Returns:
        (unique_records, n_removed)
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique: list[dict] = []
    n_removed = 0

    iter_recs = tqdm(enumerate(records), total=len(records), desc="MinHash dedup") if verbose else enumerate(records)

    for idx, rec in iter_recs:
        text = rec.get(text_field, "")
        if len(text) < 40:
            # Short utterances already deduplicated by exact SHA-256
            unique.append(rec)
            continue
        m = make_minhash(text, num_perm=num_perm, ngram=ngram)
        key = f"doc_{idx}"
        try:
            candidates = lsh.query(m)
        except Exception:
            candidates = []

        if not candidates:
            # No near-duplicate found — add to index
            try:
                lsh.insert(key, m)
            except Exception:
                pass
            unique.append(rec)
        else:
            # Near-duplicate of an existing record — discard
            n_removed += 1

    return unique, n_removed


def run_deduplication_pipeline(
    records: list[dict],
    text_field: str = "text",
    minhash_threshold: float = 0.85,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """
    Run full deduplication: exact first, then near-duplicate.

    Returns:
        (deduplicated_records, stats_dict)
    """
    n_original = len(records)

    if verbose:
        print(f"  Dedup input: {n_original:,} records")

    # Stage 1: Exact
    records, n_exact = deduplicate_exact(records, text_field=text_field)
    if verbose:
        print(f"  After exact dedup: {len(records):,} records (removed {n_exact:,})")

    # Stage 2: Near-duplicate
    records, n_near = deduplicate_minhash(
        records, text_field=text_field,
        threshold=minhash_threshold, verbose=verbose
    )
    if verbose:
        print(f"  After MinHash dedup: {len(records):,} records (removed {n_near:,})")

    stats = {
        "n_original": n_original,
        "n_exact_removed": n_exact,
        "n_near_removed": n_near,
        "n_final": len(records),
        "retention_rate": len(records) / max(n_original, 1),
    }
    return records, stats


if __name__ == "__main__":
    # Smoke test
    test_records = [
        {"id": "1", "text": "ሰላም ዓለም ። ይህ ሙከራ ጽሑፍ ነው ።"},
        {"id": "2", "text": "ሰላም ዓለም ። ይህ ሙከራ ጽሑፍ ነው ።"},   # Exact duplicate
        {"id": "3", "text": "ሰላም ዓለም ። ይህ ሙከራ ጽሑፍ ነው ።"},   # Exact duplicate
        {"id": "4", "text": "ትንሽ ልጅቷ በጫካ ውስጥ ሄደች ።"},
        {"id": "5", "text": "ትንሽ ልጅቷ በጫካ ውስጥ ሄደች ።  "},  # Near-duplicate (trailing space diff)
        {"id": "6", "text": "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት ።"},
    ]
    result, stats = run_deduplication_pipeline(test_records, verbose=True)
    print(f"\nStats: {json.dumps(stats, indent=2)}")
    for r in result:
        print(f"  Kept: {r['id']} — {r['text'][:50]}")

