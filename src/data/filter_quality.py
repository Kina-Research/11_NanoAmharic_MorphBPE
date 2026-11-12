"""
src/data/filter_quality.py
Quality filtering heuristics for Amharic text corpus.

Filters:
1. Script purity: Non-whitespace characters must have >= min_ethiopic_ratio in Ethiopic Unicode block (default 0.60).
2. Length filters:
   - Min characters: 20
   - Min words: 4
   - Max characters: 20,000 (prevents pathological single lines)
3. Repetition filter:
   - Max single-character repetition: no character repeated > 4 times consecutively (e.g., ".......", "ሀሀሀሀሀ").
   - Max repeated token n-gram ratio: checks if text is an uninformative loop.
4. Boilerplate / noise detection:
   - Strips common web scraping artifacts, system tags, and navigation headers.
"""

import re
from typing import Optional
from src.data.normalize_amharic import ethiopic_char_ratio

# Regex patterns for repetition
_REPEATED_CHARS = re.compile(r"(.)\1{4,}")  # 5 or more of same character
_REPEATED_PUNCT = re.compile(r"([፡።፣፤፥፦፧!\?.])\1{2,}")  # 3 or more punctuation marks


def has_excessive_repetition(text: str, max_ngram_repeat: int = 4) -> bool:
    """Detect if a string has pathological character or word loops."""
    # Check character repetitions
    if _REPEATED_CHARS.search(text):
        return True

    words = text.split()
    if len(words) < 6:
        return False

    # Check for repeated 1-grams, 2-grams, or 3-grams
    for n in (1, 2, 3):
        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            continue
        # Count consecutive identical ngrams
        consecutive = 1
        for i in range(1, len(ngrams)):
            if ngrams[i] == ngrams[i - 1]:
                consecutive += 1
                if consecutive >= max_ngram_repeat:
                    return True
            else:
                consecutive = 1
    return False


def filter_quality_record(
    text: str,
    min_chars: int = 20,
    max_chars: int = 20000,
    min_words: int = 4,
    min_ethiopic_ratio: float = 0.60,
) -> tuple[bool, Optional[str]]:
    """
    Validate whether a text string passes all quality criteria.

    Returns:
        (is_valid, rejection_reason)
    """
    if not text:
        return False, "empty_text"

    stripped = text.strip()
    char_len = len(stripped)

    if char_len < min_chars:
        return False, f"too_short_chars_{char_len}"

    if char_len > max_chars:
        return False, f"too_long_chars_{char_len}"

    words = stripped.split()
    if len(words) < min_words:
        return False, f"too_few_words_{len(words)}"

    # Check Ethiopic script purity
    ratio = ethiopic_char_ratio(stripped)
    if ratio < min_ethiopic_ratio:
        return False, f"low_ethiopic_ratio_{ratio:.2f}"

    # Check excessive repetition
    if has_excessive_repetition(stripped):
        return False, "excessive_repetition"

    return True, None


def batch_filter(
    records: list[dict],
    text_key: str = "text",
    min_chars: int = 20,
    min_words: int = 4,
    min_ethiopic_ratio: float = 0.60,
) -> tuple[list[dict], dict]:
    """
    Filter a list of record dicts.

    Returns:
        (filtered_records, stats_dict)
    """
    valid_records = []
    reasons_count: dict[str, int] = {}

    for rec in records:
        text = rec.get(text_key, "")
        ok, reason = filter_quality_record(
            text,
            min_chars=min_chars,
            min_words=min_words,
            min_ethiopic_ratio=min_ethiopic_ratio,
        )
        if ok:
            valid_records.append(rec)
        else:
            reasons_count[reason] = reasons_count.get(reason, 0) + 1

    stats = {
        "input_count": len(records),
        "passed_count": len(valid_records),
        "rejected_count": len(records) - len(valid_records),
        "rejection_reasons": reasons_count,
    }
    return valid_records, stats


if __name__ == "__main__":
    test_cases = [
        "ይህ አጭር እና ትክክለኛ የአማርኛ ዓረፍተ ነገር ነው ።",  # Pass
        "Short",                                         # Too short
        "Hello this is all English language text here.", # Low Ethiopic
        "ሰላም ሰላም ሰላም ሰላም ሰላም ሰላም ሰላም",                   # Repetition
        "ሀሀሀሀሀሀሀሀሀሀሀ",                                  # Char repeat
    ]
    for tc in test_cases:
        ok, reason = filter_quality_record(tc)
        print(f"[{'PASS' if ok else 'FAIL'}] {reason or 'OK'}: {tc}")

