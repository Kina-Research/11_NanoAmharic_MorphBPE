"""
test_evaluation.py — Unit tests for evaluation modules: morphological auditing, generation diversity, and significance testing.
"""

import pytest
from src.evaluation.morph_eval import audit_word_morphology, evaluate_text_morphology
from src.evaluation.generation_eval import compute_diversity_metrics
from src.evaluation.significance import paired_bootstrap_test


def test_audit_word_morphology_valid_and_anomalies():
    """Verify detection of valid words, affix-bearing stems, and morphological anomalies."""
    # 1. Normal valid word without affixes
    res_normal = audit_word_morphology("ሰላም")
    assert res_normal["is_valid"] is True
    assert res_normal["has_error"] is False

    # 2. Valid prefix attachment with valid stem
    res_prefix = audit_word_morphology("የከተማ")
    assert res_prefix["has_affix"] is True
    assert res_prefix["has_error"] is False
    assert res_prefix["stem"] == "ከተማ"

    # 3. Syllable stuttering anomaly
    res_stutter = audit_word_morphology("በበበሰው")
    assert res_stutter["has_error"] is True
    assert res_stutter["error_type"] == "syllable_stutter"

    # 4. Degenerate stem (< 2 characters) with multi-character prefix
    res_degenerate = audit_word_morphology("እንደአ")
    assert res_degenerate["has_error"] is True
    assert res_degenerate["error_type"] == "degenerate_stem"


def test_evaluate_text_morphology():
    """Verify aggregated morphological metrics."""
    corpus = [
        "የአዲስ አበባ ከተማ በጣም ቆንጆ ነው።",
        "በጥንት ዘመን አንድ ንጉሥ ነበረ።",
    ]
    res = evaluate_text_morphology(corpus)
    assert res["total_words_audited"] > 0
    assert res["affix_words_identified"] > 0
    assert res["morphological_error_rate_pct"] == 0.0
    assert res["stem_preservation_score_pct"] == 100.0


def test_compute_diversity_metrics():
    """Verify computation of Distinct-1, Distinct-2, and Repetition Rate."""
    # Highly diverse sentence
    diverse_texts = ["አንድ ሁለት ሦስት አራት አምስት ስድስት ሰባት ስምንት ዘጠኝ አስር"]
    metrics = compute_diversity_metrics(diverse_texts)
    assert metrics["distinct_1"] == 1.0
    assert metrics["distinct_2"] == 1.0
    assert metrics["repetition_3gram_rate"] == 0.0

    # Repetitive sentence
    repetitive_texts = ["ውሻ ውሻ ውሻ ውሻ ውሻ ውሻ"]
    metrics_rep = compute_diversity_metrics(repetitive_texts)
    assert metrics_rep["distinct_1"] < 0.5
    assert metrics_rep["repetition_3gram_rate"] > 0.0


def test_paired_bootstrap_test():
    """Verify paired bootstrap resampling behavior."""
    # Case 1: identical arrays
    arr1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    res_same = paired_bootstrap_test(arr1, arr1, num_bootstraps=200)
    assert res_same["observed_mean_difference"] == 0.0
    assert res_same["is_statistically_significant"] is False

    # Case 2: significantly different arrays
    arr_low = [1.0, 1.2, 1.1, 1.3, 1.0] * 10
    arr_high = [2.0, 2.2, 2.1, 2.3, 2.0] * 10
    res_diff = paired_bootstrap_test(arr_low, arr_high, num_bootstraps=500)
    assert res_diff["observed_mean_difference"] > 0.9
    assert res_diff["ci_lower"] > 0.8
    assert res_diff["is_statistically_significant"] is True
