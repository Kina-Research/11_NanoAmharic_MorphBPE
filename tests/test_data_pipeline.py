"""
tests/test_data_pipeline.py
Unit tests for data normalization, quality filtering, and deduplication.
Run: pytest tests/test_data_pipeline.py -v
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.normalize_amharic import (
    normalize_text,
    is_ethiopic_char,
    ethiopic_char_ratio,
    ETHIOPIC_PUNCT,
)
from src.data.filter_quality import (
    filter_quality_record,
    has_excessive_repetition,
    batch_filter,
)
from src.data.deduplicate import (
    sha256_hash,
    deduplicate_exact,
    deduplicate_minhash,
    make_minhash,
)


# =========================================================================
# 1. Normalization Tests
# =========================================================================
def test_ethiopic_char_detection():
    # Ethiopic characters
    assert is_ethiopic_char("ሀ")
    assert is_ethiopic_char("ለ")
    assert is_ethiopic_char("ሐ")
    assert is_ethiopic_char("ፐ")
    assert is_ethiopic_char("።")  # Ethiopic full stop
    assert is_ethiopic_char("፣")  # Ethiopic comma

    # Non-Ethiopic characters
    assert not is_ethiopic_char("A")
    assert not is_ethiopic_char("9")
    assert not is_ethiopic_char("$")
    assert not is_ethiopic_char("中")


def test_ethiopic_char_ratio():
    text_pure = "በጥንት ዘመን አንድ ንጉስ ነበር።"
    ratio = ethiopic_char_ratio(text_pure)
    assert ratio > 0.95

    text_mixed = "Hello በጥንት ዘመን"
    ratio_mixed = ethiopic_char_ratio(text_mixed)
    assert 0.4 < ratio_mixed < 0.7

    text_latin = "This is entirely English text without Ethiopic."
    assert ethiopic_char_ratio(text_latin) == 0.0


def test_normalize_text_clean():
    raw = "   ትንሽ   ልጅቷ \t\t በጫካ   ውስጥ \n\n ሄደች ።   "
    clean = normalize_text(raw)
    assert clean == "ትንሽ ልጅቷ በጫካ ውስጥ \n\n ሄደች ።"


def test_normalize_text_strips_html_and_urls():
    raw = "የኢትዮጵያ <b>ታሪክ</b> በ https://example.com/history ላይ ይገኛል ።"
    clean = normalize_text(raw)
    assert "<b>" not in clean
    assert "https://" not in clean
    assert "የኢትዮጵያ" in clean
    assert "ታሪክ" in clean


def test_normalize_text_preserves_homophones_by_default():
    # ሠ vs ሰ must be preserved by default
    sample = "ሠራዊት እና ሰላም"
    clean = normalize_text(sample, apply_homophone_fold=False)
    assert "ሠ" in clean
    assert "ሰ" in clean


def test_normalize_text_optional_homophone_folding():
    # When explicitly requested, ሠ converts to ሰ
    sample = "ሠራዊት"
    folded = normalize_text(sample, apply_homophone_fold=True)
    assert "ሰ" in folded
    assert "ሠ" not in folded


# =========================================================================
# 2. Quality Filtering Tests
# =========================================================================
def test_filter_valid_amharic_record():
    valid = "በጥንት ዘመን አንድ ታላቅ ንጉሥ በኢትዮጵያ ምድር ይገዛ ነበር ።"
    ok, reason = filter_quality_record(valid)
    assert ok
    assert reason is None


def test_filter_rejects_too_short():
    short = "አጭር ።"
    ok, reason = filter_quality_record(short, min_chars=20)
    assert not ok
    assert "too_short" in reason


def test_filter_rejects_low_ethiopic():
    mostly_english = "The capital city of Ethiopia is አዲስ አበባ in the horn."
    ok, reason = filter_quality_record(mostly_english, min_ethiopic_ratio=0.60)
    assert not ok
    assert "low_ethiopic_ratio" in reason


def test_filter_rejects_char_repetition():
    repeated = "ሀሀሀሀሀሀሀሀሀሀ በጫካ ውስጥ ነበር ።"
    ok, reason = filter_quality_record(repeated)
    assert not ok
    assert reason == "excessive_repetition"


def test_filter_rejects_word_loop():
    loop = "ሰላም ሰላም ሰላም ሰላም ሰላም ሰላም ሰላም ።"
    assert has_excessive_repetition(loop)
    ok, reason = filter_quality_record(loop)
    assert not ok
    assert reason == "excessive_repetition"


# =========================================================================
# 3. Deduplication Tests
# =========================================================================
def test_exact_deduplication():
    records = [
        {"id": "1", "text": "በጥንት ዘመን አንድ ንጉስ ነበር ።"},
        {"id": "2", "text": "በጥንት ዘመን አንድ ንጉስ ነበር ።"},  # Duplicate
        {"id": "3", "text": "ትንሽ ልጅቷ በጫካ ውስጥ ሄደች ።"},
    ]
    unique, removed = deduplicate_exact(records)
    assert len(unique) == 2
    assert removed == 1
    assert unique[0]["id"] == "1"
    assert unique[1]["id"] == "3"
    assert "sha256" in unique[0]


def test_minhash_near_deduplication():
    records = [
        {"id": "1", "text": "በጥንት ዘመን አንድ ታላቅ ንጉሥ በኢትዮጵያ ምድር በፍትሕ እና በሰላም ይገዛ ነበር። ንጉሡም ለሕዝቡ መልካም መሪ ነበር ።"},
        # Near duplicate (nearly identical paragraph with minor trailing variation)
        {"id": "2", "text": "በጥንት ዘመን አንድ ታላቅ ንጉሥ በኢትዮጵያ ምድር በፍትሕ እና በሰላም ይገዛ ነበር። ንጉሡም ለሕዝቡ መልካም መሪ ነበረ ።"},
        # Completely different record
        {"id": "3", "text": "አንበሳው እና ጥንቸሏ በጫካ ውስጥ ተገናኝተው የጓደኝነት ውል ተፈራረሙ ።"},
    ]
    unique, removed = deduplicate_minhash(records, threshold=0.70, num_perm=64, verbose=False)
    # The near duplicate (id 2) should be filtered out
    assert len(unique) == 2
    assert removed == 1
    assert unique[0]["id"] == "1"
    assert unique[1]["id"] == "3"

