"""
Writing assistant package for NanoAmharic academic drafting and integrity verification.
"""

from src.writing_assistant.grounding import EmpiricalGroundingContext
from src.writing_assistant.section_generator import AcademicSectionGenerator
from src.writing_assistant.citation_verifier import verify_citations, extract_citations, generate_bibtex_file
from src.writing_assistant.integrity_auditor import ManuscriptIntegrityAuditor
from src.writing_assistant.pipeline import run_writing_assistant_pipeline

__all__ = [
    "EmpiricalGroundingContext",
    "AcademicSectionGenerator",
    "verify_citations",
    "extract_citations",
    "generate_bibtex_file",
    "ManuscriptIntegrityAuditor",
    "run_writing_assistant_pipeline",
]
