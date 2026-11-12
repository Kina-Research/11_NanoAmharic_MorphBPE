import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.writing_assistant.citation_verifier import generate_bibtex_file, VERIFIED_BIBLIOGRAPHY

out_path = generate_bibtex_file("paper/references.bib")
print(f"Generated {out_path} with {len(VERIFIED_BIBLIOGRAPHY)} verified entries.")

