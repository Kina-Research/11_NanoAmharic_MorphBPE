"""
src/data/normalize_amharic.py
Unicode normalization for Ethiopic (Ge'ez) script.

Design decisions:
- NFC normalization is applied (standard Unicode canonical composition)
- Canonical homophone folding (ሀ/ሐ/ኀ, ሰ/ሠ, አ/ዐ, ጸ/ፀ) is DISABLED by default.
  These distinctions are preserved. Folding is an optional experimental transformation.
- Ethiopic punctuation is preserved and standardized.
- ASCII-only records are flagged for review.
"""

import re
import unicodedata
from typing import Optional


# ── Ethiopic Unicode block ranges ──────────────────────────────────────────
ETHIOPIC_RANGES = [
    (0x1200, 0x137F),   # Ethiopic
    (0x1380, 0x139F),   # Ethiopic Supplement
    (0x2D80, 0x2DDF),   # Ethiopic Extended
    (0xAB01, 0xAB2F),   # Ethiopic Extended-A
    (0x1E7E0, 0x1E7FF), # Ethiopic Extended-B (Unicode 14+)
]

# Ethiopic punctuation characters to preserve
ETHIOPIC_PUNCT = set("፡።፣፤፥፦፧፨")

# Whitespace normalization — collapse to single space / newline
_MULTI_SPACE = re.compile(r"[ \t\u00a0\u200b\u200c\u200d\ufeff]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Noise patterns to remove
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Optional homophone folding map (NOT applied by default)
HOMOPHONE_FOLD_MAP = {
    # ሀ / ሐ / ኀ → ሀ (all represent /hä/)
    "ሐ": "ሀ", "ኀ": "ሀ",
    "ሑ": "ሁ", "ኁ": "ሁ",
    "ሒ": "ሂ", "ኂ": "ሂ",
    "ሓ": "ሃ", "ኃ": "ሃ",
    "ሔ": "ሄ", "ኄ": "ሄ",
    "ሕ": "ህ", "ኅ": "ህ",
    "ሖ": "ሆ", "ኆ": "ሆ",
    # ሠ → ሰ (/sä/ merger in modern Amharic)
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    # ዐ → አ (/ʕä/ → /ʔä/ in modern spoken Amharic; preservation preferred)
    # NOT included: ዐ→አ is controversial and may cause semantic loss
    # ፀ → ጸ (/ts'ä/ merger in some dialects; preservation preferred)
    # NOT included: ፀ→ጸ is controversial
}


def is_ethiopic_char(ch: str) -> bool:
    """Return True if character falls within Ethiopic Unicode ranges."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ETHIOPIC_RANGES)


def ethiopic_char_ratio(text: str) -> float:
    """Return fraction of non-whitespace characters that are Ethiopic."""
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    ethiopic = sum(1 for ch in chars if is_ethiopic_char(ch))
    return ethiopic / len(chars)


def normalize_text(
    text: str,
    apply_homophone_fold: bool = False,
    remove_urls: bool = True,
    remove_html: bool = True,
    min_ethiopic_ratio: float = 0.0,  # 0 = no filter here; filtering done separately
) -> Optional[str]:
    """
    Normalize a single Amharic text string.

    Args:
        text: Raw input string.
        apply_homophone_fold: If True, apply canonical homophone folding
                              (ሐ→ሀ, ሠ→ሰ, etc.). Default False — preserve distinctions.
        remove_urls: Strip URL patterns.
        remove_html: Strip HTML tags.
        min_ethiopic_ratio: Minimum Ethiopic character ratio; return None if below threshold.

    Returns:
        Normalized string, or None if the text fails quality checks.
    """
    if not text or not text.strip():
        return None

    # 1. Remove control characters
    text = _CONTROL_CHARS.sub("", text)

    # 2. Remove HTML tags
    if remove_html:
        text = _HTML_TAG.sub(" ", text)

    # 3. Remove URLs and emails
    if remove_urls:
        text = _URL.sub("", text)
        text = _EMAIL.sub("", text)

    # 4. Unicode NFC normalization (canonical decomposition + composition)
    text = unicodedata.normalize("NFC", text)

    # 5. Optional homophone folding (disabled by default)
    if apply_homophone_fold:
        text = "".join(HOMOPHONE_FOLD_MAP.get(ch, ch) for ch in text)

    # 6. Normalize whitespace (preserve single newlines as paragraph separators)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = text.strip()

    if not text:
        return None

    # 7. Ethiopic ratio check (optional early exit)
    if min_ethiopic_ratio > 0.0:
        ratio = ethiopic_char_ratio(text)
        if ratio < min_ethiopic_ratio:
            return None

    return text


def normalize_batch(
    texts: list[str],
    apply_homophone_fold: bool = False,
    min_ethiopic_ratio: float = 0.3,
) -> list[Optional[str]]:
    """Normalize a list of texts. Returns None for rejected records."""
    return [
        normalize_text(t, apply_homophone_fold=apply_homophone_fold,
                       min_ethiopic_ratio=min_ethiopic_ratio)
        for t in texts
    ]


if __name__ == "__main__":
    # Quick smoke test
    samples = [
        "ሰላም ፣ ዓለም ።",
        "   ትንሽ ልጅቷ   በጫካ ውስጥ   ሄደች ።  ",
        "This is English only text.",
        "",
        "ሐሙስ ቀን <b>ብርሃን</b> https://example.com ወደ ገበያ ሄደ",
        "ሠማይ ሰማያዊ ነው ።",   # ሠ vs ሰ preserved by default
    ]
    for s in samples:
        result = normalize_text(s, min_ethiopic_ratio=0.3)
        print(f"IN:  {repr(s[:60])}")
        print(f"OUT: {repr(result)}")
        print()

