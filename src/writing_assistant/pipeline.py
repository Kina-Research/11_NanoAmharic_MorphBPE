"""
pipeline.py — Master orchestrator for the AI Academic Writing Assistant.
Generates full manuscript draft, verifies bibliographic citations, executes integrity audits, and compiles reports.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.writing_assistant.grounding import EmpiricalGroundingContext
from src.writing_assistant.section_generator import AcademicSectionGenerator
from src.writing_assistant.citation_verifier import (
    extract_citations,
    verify_citations,
    generate_bibtex_file,
)
from src.writing_assistant.integrity_auditor import ManuscriptIntegrityAuditor


def run_writing_assistant_pipeline(
    passport_path: str = "paper/material_passport.json",
    output_manuscript_md: str = "paper/manuscript_draft.md",
    output_bibtex: str = "paper/references.bib",
    output_audit_json: str = "eval/results/academic_integrity_report.json",
    output_audit_md: str = "eval/reports/academic_integrity_report.md",
) -> Dict[str, Any]:
    """
    Executes complete academic drafting and research integrity verification lifecycle.
    """
    print("\n========================================================")
    print(" PHASE 7: AI ACADEMIC WRITING ASSISTANT PIPELINE")
    print("========================================================\n")

    passport_p = PROJECT_ROOT / passport_path
    manuscript_p = PROJECT_ROOT / output_manuscript_md
    bibtex_p = PROJECT_ROOT / output_bibtex
    audit_json_p = PROJECT_ROOT / output_audit_json
    audit_md_p = PROJECT_ROOT / output_audit_md

    manuscript_p.parent.mkdir(parents=True, exist_ok=True)
    audit_json_p.parent.mkdir(parents=True, exist_ok=True)
    audit_md_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Generate Canonical BibTeX File
    print(f"Generating verified bibliography at {bibtex_p}...")
    generate_bibtex_file(str(bibtex_p))

    # 2. Generate Full Manuscript Draft Grounded in Empirical Evidence
    print(f"Drafting academic manuscript from empirical facts in {passport_p}...")
    generator = AcademicSectionGenerator(str(passport_p))
    manuscript_text = generator.generate_full_manuscript()

    with open(manuscript_p, "w", encoding="utf-8") as f:
        f.write(manuscript_text)
    print(f"Saved complete manuscript draft ({len(manuscript_text):,} characters) to {manuscript_p}.")

    # 3. Citation Verification
    print("Verifying in-text citations against canonical verified bibliography...")
    citations_found = extract_citations(manuscript_text)
    citation_audit = verify_citations(citations_found)
    print(f"Citations Verified: {citation_audit['valid_citations_count']}/{citation_audit['total_citations']} (Integrity: {citation_audit['integrity_ratio']*100:.1f}%)")

    # 4. Academic Integrity & Hallucination Audit
    print("Auditing numerical claims and experimental assertions...")
    auditor = ManuscriptIntegrityAuditor(str(passport_p))
    integrity_audit = auditor.audit_text(manuscript_text)
    print(f"Integrity Audit: {integrity_audit['verified_claims_count']}/{integrity_audit['total_audit_rules']} verified claims, {len(integrity_audit['flags_detected'])} flags.")

    # 5. Compile Comprehensive Report
    combined_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manuscript_path": str(manuscript_p),
        "bibtex_path": str(bibtex_p),
        "citation_verification": citation_audit,
        "integrity_audit": integrity_audit,
        "overall_status": "APPROVED" if (citation_audit["all_citations_verified"] and integrity_audit["audit_passed"]) else "REQUIRES_ATTENTION",
    }

    with open(audit_json_p, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2, ensure_ascii=False)

    # Write Markdown Audit Report
    write_integrity_markdown_report(combined_report, audit_md_p)
    print(f"Saved integrity report to {audit_md_p}")

    return combined_report


def write_integrity_markdown_report(report: Dict[str, Any], path: Path):
    """Write human-readable integrity and citation verification report."""
    c_audit = report["citation_verification"]
    i_audit = report["integrity_audit"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Academic Integrity & Citation Verification Report\n")
        f.write(f"*Generated: {report['timestamp']} | Verification Status: **{report['overall_status']}***\n\n---\n\n")

        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Overall Verification Status:** `{report['overall_status']}`\n")
        f.write(f"- **Verified Numerical Claims:** {i_audit['verified_claims_count']} / {i_audit['total_audit_rules']} verified against `material_passport.json`\n")
        f.write(f"- **Hallucination / Banned Assertions:** {len(i_audit['flags_detected'])} detected\n")
        f.write(f"- **Citation Integrity:** {c_audit['valid_citations_count']} / {c_audit['total_citations']} citations verified ({c_audit['integrity_ratio']*100:.1f}%)\n")
        f.write(f"- **Integrity Score:** {i_audit['integrity_score']*100:.2f}%\n\n---\n\n")

        f.write("## 2. Citation Verification Details\n\n")
        f.write("| Key | Title | Verification Status |\n")
        f.write("|:---|:---|:---:|\n")
        for key in c_audit["valid_keys"]:
            f.write(f"| `\\cite{{{key}}}` | Verified Canonical Entry | **VERIFIED** |\n")

        if c_audit["invalid_keys"]:
            f.write("\n### Unverified / Flagged Keys:\n")
            for key in c_audit["invalid_keys"]:
                f.write(f"- ⚠️ `{key}` (Not found in canonical bibliography)\n")

        f.write("\n---\n\n## 3. Verified Numerical Claims in Manuscript Draft\n\n")
        f.write("| Metric / Claim Key | Canonical Ground Truth Value | Verification Result |\n")
        f.write("|:---|:---:|:---:|\n")
        for claim in i_audit["verified_claims"]:
            f.write(f"| `{claim['claim_key']}` | `{claim['expected_value']}` | **VERIFIED MATCH** |\n")


if __name__ == "__main__":
    run_writing_assistant_pipeline()

