"""
morph_eval.py — Morphological error rate and linguistic integrity evaluation for Amharic generation.
Evaluates affix consistency, root stem preservation, and syllable formation validity.
"""

import re
from typing import List, Dict, Any, Tuple
from src.tokenizer.morph_rules import AMHARIC_PREFIXES as PREFIXES, AMHARIC_SUFFIXES as SUFFIXES
from src.data.normalize_amharic import is_ethiopic_char


def is_valid_ethiopic_word(word: str) -> bool:
    """Check if token consists strictly of valid Ethiopic syllabary characters."""
    if not word:
        return False
    return all(is_ethiopic_char(c) for c in word)


def audit_word_morphology(word: str) -> Dict[str, Any]:
    """
    Audits an Amharic word for morphological anomalies:
    1. Syllable stuttering (e.g. character repeated >= 3 times).
    2. Detached or malformed prefixes.
    3. Degenerate stem (root stem < 2 characters when bearing affixes).
    4. Mixed invalid characters.
    """
    clean_word = word.strip(" ፣፤።?!:-\"'\n\r\t")
    if not clean_word:
        return {"is_valid": True, "has_affix": False, "has_error": False, "error_type": None}

    # 1. Check repetition stutter (e.g. 3 consecutive identical characters)
    if re.search(r"(.)\1{2,}", clean_word):
        return {"is_valid": False, "has_affix": False, "has_error": True, "error_type": "syllable_stutter"}

    # 2. Check for mixed non-Ethiopic junk inside word
    non_eth = [c for c in clean_word if not is_ethiopic_char(c)]
    if non_eth:
        return {"is_valid": False, "has_affix": False, "has_error": True, "error_type": "non_ethiopic_character"}

    # 3. Check affix attachments (only applicable for words of length >= 3)
    if len(clean_word) < 3:
        return {"is_valid": True, "has_affix": False, "has_error": False, "error_type": None}

    has_prefix = False
    prefix_matched = None
    remaining = clean_word

    for p in sorted(PREFIXES, key=len, reverse=True):
        if remaining.startswith(p):
            has_prefix = True
            prefix_matched = p
            remaining = remaining[len(p):]
            break

    has_suffix = False
    suffix_matched = None
    for s in sorted(SUFFIXES, key=len, reverse=True):
        if remaining.endswith(s):
            has_suffix = True
            suffix_matched = s
            remaining = remaining[:-len(s)]
            break

    # If both prefix and suffix are matched on a short word (< 4 chars), it cannot be a multi-affix derivative
    if has_prefix and has_suffix and len(clean_word) < 4:
        stem_if_prefix = clean_word[len(prefix_matched):]
        stem_if_suffix = clean_word[:-len(suffix_matched)]
        if len(stem_if_prefix) >= 2:
            has_suffix = False
            suffix_matched = None
            remaining = stem_if_prefix
        elif len(stem_if_suffix) >= 2:
            has_prefix = False
            prefix_matched = None
            remaining = stem_if_suffix

    # If it has affix, remaining must be a valid root stem of length >= 2
    if has_prefix or has_suffix:
        if len(remaining) < 2:
            return {
                "is_valid": False,
                "has_affix": True,
                "has_error": True,
                "error_type": "degenerate_stem",
                "stem": remaining,
                "prefix": prefix_matched,
                "suffix": suffix_matched,
            }
        return {
            "is_valid": True,
            "has_affix": True,
            "has_error": False,
            "error_type": None,
            "stem": remaining,
            "prefix": prefix_matched,
            "suffix": suffix_matched,
        }

    return {"is_valid": True, "has_affix": False, "has_error": False, "error_type": None}


def evaluate_text_morphology(texts: List[str]) -> Dict[str, Any]:
    """
    Evaluates morphological error rate and stem preservation across a collection of texts.
    """
    total_words = 0
    affix_words = 0
    valid_affix_words = 0
    morph_errors = 0
    error_breakdown: Dict[str, int] = {}

    for text in texts:
        words = text.split()
        for w in words:
            res = audit_word_morphology(w)
            total_words += 1
            if res["has_affix"]:
                affix_words += 1
                if not res["has_error"]:
                    valid_affix_words += 1

            if res["has_error"]:
                morph_errors += 1
                err_type = res["error_type"] or "unknown"
                error_breakdown[err_type] = error_breakdown.get(err_type, 0) + 1

    mer = (morph_errors / total_words * 100.0) if total_words > 0 else 0.0
    stem_preservation = (valid_affix_words / affix_words * 100.0) if affix_words > 0 else 100.0

    return {
        "total_words_audited": total_words,
        "affix_words_identified": affix_words,
        "valid_affix_words": valid_affix_words,
        "morphological_errors_total": morph_errors,
        "morphological_error_rate_pct": round(mer, 2),
        "stem_preservation_score_pct": round(stem_preservation, 2),
        "error_type_breakdown": error_breakdown,
    }
