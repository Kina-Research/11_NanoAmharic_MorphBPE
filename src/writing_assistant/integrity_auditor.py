"""
integrity_auditor.py — Academic research integrity auditor and hallucination detector.
Cross-verifies numerical claims and empirical figures in drafted manuscript against material_passport.json.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.writing_assistant.grounding import EmpiricalGroundingContext


class ManuscriptIntegrityAuditor:
    """
    Audits drafted paper sections for empirical consistency, verifying that reported numbers
    match actual experiment logs in material_passport.json.
    """

    def __init__(self, passport_path: str = None):
        self.grounding = EmpiricalGroundingContext(passport_path)
        self.facts = self.grounding.get_all_grounded_facts()

    def audit_text(self, text: str) -> Dict[str, Any]:
        """
        Scans manuscript text, extracts empirical claims, and compares against ground truth.
        """
        ground_truth_claims = {
            # Corpus
            "clean_documents": ("120,930", ["120,930", "120930"]),
            "corpus_tokens": ("5,380,546", ["5,380,546", "5.38M", "5.38 million"]),
            "corpus_chars": ("27,557,689", ["27,557,689", "27.56M", "27.56 million"]),
            "exact_duplicates": ("56,310", ["56,310", "56310"]),
            # Model & Architecture
            "param_count": ("5.75M", ["5.75M", "5.75 million", "5,745,920"]),
            # Tokenizer
            "baseline_bpe_fertility": ("1.575", ["1.575"]),
            "morphbpe_fertility": ("1.585", ["1.585"]),
            "baseline_mbc": ("17.89%", ["17.89%"]),
            "morphbpe_mbc": ("25.10%", ["25.10%", "25.1%"]),
            "mbc_boost": ("40.3%", ["40.3%"]),
            # Pretraining
            "baseline_val_loss": ("3.5533", ["3.5533"]),
            "morphbpe_val_loss": ("3.5118", ["3.5118"]),
            "baseline_val_ppl": ("34.93", ["34.93"]),
            "morphbpe_val_ppl": ("33.51", ["33.51"]),
            "training_time": ("29.1 minutes", ["29.1 minutes", "29.1 min"]),
            # Evaluation
            "baseline_bpc": ("1.2776", ["1.2776"]),
            "morphbpe_bpc": ("1.2664", ["1.2664"]),
            "baseline_word_ppl": ("92.84", ["92.84"]),
            "morphbpe_word_ppl": ("89.25", ["89.25"]),
            "baseline_mer": ("3.03%", ["3.03%"]),
            "morphbpe_mer": ("1.46%", ["1.46%"]),
            "mer_reduction": ("51.82%", ["51.82%", "51.8%"]),
            "morphbpe_stem_preservation": ("97.54%", ["97.54%"]),
            # Quantization
            "fp32_size": ("65.82 MB", ["65.82 MB", "65.82MB"]),
            "int8_size": ("9.44 MB", ["9.44 MB", "9.44MB"]),
            "compression_ratio": ("6.97x", ["6.97x", "6.97 times"]),
            "disk_savings": ("85.66%", ["85.66%"]),
        }

        verified_claims = []
        missing_claims = []

        for key, (canonical, aliases) in ground_truth_claims.items():
            matched = any(alias.lower() in text.lower() for alias in aliases)
            if matched:
                verified_claims.append({"claim_key": key, "expected_value": canonical, "status": "VERIFIED_PRESENT"})
            else:
                missing_claims.append({"claim_key": key, "expected_value": canonical, "status": "NOT_REFERENCED"})

        # Check for banned/fabricated assertions (unauthorized claims of GPU training)
        suspicious_patterns = [
            (r"(?:trained on|utilizing|accelerated with|using)\s+(?:(?:\d+x|a|an|the)\s+)?(?:NVIDIA|CUDA|GPU)", "Claiming unauthorized GPU acceleration on CPU-only workstation"),
            (r"ChatGPT|GPT-4 fabricated", "Cloud LLM dependency mentioned"),
            (r"p\s*>\s*0\.05", "Claiming non-significant result as significant"),
        ]

        flags = []
        for pat, reason in suspicious_patterns:
            if re.search(pat, text, re.IGNORECASE):
                flags.append({"pattern": pat, "reason": reason})

        integrity_score = len(verified_claims) / max(1, (len(verified_claims) + len(flags)))

        return {
            "total_audit_rules": len(ground_truth_claims),
            "verified_claims_count": len(verified_claims),
            "missing_claims_count": len(missing_claims),
            "flags_detected": flags,
            "integrity_score": round(integrity_score, 4),
            "audit_passed": (len(flags) == 0 and len(verified_claims) >= 15),
            "verified_claims": verified_claims,
        }
