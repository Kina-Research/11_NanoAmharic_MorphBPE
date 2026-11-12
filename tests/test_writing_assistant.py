"""
test_writing_assistant.py — Unit tests for AI writing assistant: grounding, citation verification, integrity audit, and section generation.
"""

import pytest
from src.writing_assistant.grounding import EmpiricalGroundingContext
from src.writing_assistant.citation_verifier import verify_citations, extract_citations
from src.writing_assistant.integrity_auditor import ManuscriptIntegrityAuditor
from src.writing_assistant.section_generator import AcademicSectionGenerator


def test_empirical_grounding_loads_real_metrics():
    """Verify that grounding loads empirical figures directly from material_passport.json."""
    ctx = EmpiricalGroundingContext()
    facts = ctx.get_all_grounded_facts()

    # Corpus facts
    assert facts["corpus"]["total_clean_documents"] == 120930
    assert facts["corpus"]["total_tokens"] > 5000000

    # Tokenizer facts
    assert facts["tokenizer"]["morphbpe_default_mbc_pct"] == 25.10
    assert facts["tokenizer"]["baseline_bpe_mbc_pct"] == 17.89

    # Pretraining facts
    assert facts["pretraining"]["parameter_count_millions"] == 5.75
    assert facts["pretraining"]["morphbpe_val_ppl"] == 33.51

    # Evaluation facts
    assert facts["evaluation"]["morphbpe_bpc"] == 1.2664
    assert facts["evaluation"]["relative_mer_reduction_pct"] == 51.82

    # Quantization facts
    assert facts["quantization"]["compression_ratio"] == 6.97
    assert facts["quantization"]["int8_size_mb"] == 9.44


def test_citation_verifier_valid_and_invalid():
    """Verify detection of canonical citations vs hallucinated citation keys."""
    # LaTeX citations
    text = "We use BPE \\cite{sennrich2016subword} and RoPE \\cite{su2024roformer, fake_author_2099}."
    keys = extract_citations(text)
    assert "sennrich2016subword" in keys
    assert "su2024roformer" in keys
    assert "fake_author_2099" in keys

    audit = verify_citations(keys)
    assert audit["valid_citations_count"] == 2
    assert audit["invalid_citations_count"] == 1
    assert "fake_author_2099" in audit["invalid_keys"]
    assert audit["all_citations_verified"] is False


def test_integrity_auditor_flags_banned_hardware_claims():
    """Verify that the integrity auditor detects ungrounded claims and prohibited hardware assertions."""
    auditor = ManuscriptIntegrityAuditor()

    # Text containing prohibited GPU assertion
    bad_text = "The model was trained on 8x NVIDIA A100 CUDA GPUs."
    res_bad = auditor.audit_text(bad_text)
    assert len(res_bad["flags_detected"]) > 0
    assert res_bad["audit_passed"] is False

    # Text containing verified grounded empirical claims
    good_text = (
        "We curated 120,930 clean documents. The 5.75M model achieved 1.2664 BPC, "
        "a 51.82% reduction in morphological errors, and INT8 compressed by 6.97x to 9.44 MB. "
        "Pretraining completed in 29.1 minutes on CPU. MorphBPE achieved 25.10% MBC."
    )
    res_good = auditor.audit_text(good_text)
    assert len(res_good["flags_detected"]) == 0
    assert res_good["verified_claims_count"] >= 6


def test_section_generator_completeness():
    """Verify that the section generator produces all mandatory academic sections."""
    generator = AcademicSectionGenerator()
    full_text = generator.generate_full_manuscript()

    assert "## Abstract" in full_text
    assert "## 1. Introduction" in full_text
    assert "## 2. Methodology" in full_text
    assert "## 3. Experimental Evaluation" in full_text
    assert "## 4. Model Quantization" in full_text
    assert "## 5. Conclusion" in full_text

    # Verify key grounded empirical metrics appear in text
    assert "120,930" in full_text
    assert "5.75M" in full_text
    assert "1.2664" in full_text
    assert "51.82%" in full_text
    assert "6.97x" in full_text

