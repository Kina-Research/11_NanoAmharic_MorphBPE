"""
save_manuscript.py — Clean export and backup utility for manuscript artifacts.
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
paper_dir = PROJECT_ROOT / "paper"
dist_dir = PROJECT_ROOT / "dist" / "manuscript"
dist_dir.mkdir(parents=True, exist_ok=True)

artifacts = [
    "manuscript.pdf",
    "manuscript.tex",
    "manuscript_revised.pdf",
    "manuscript_revised.tex",
    "manuscript_revised.docx",
    "manuscript_revised.md",
    "references.bib",
]

print(f"Exporting publication manuscript artifacts to {dist_dir}...")
saved_count = 0
for f in artifacts:
    src_f = paper_dir / f
    if src_f.exists():
        dst_f = dist_dir / f
        shutil.copy2(src_f, dst_f)
        print(f"  -> Exported {f} ({dst_f.stat().st_size:,} bytes)")
        saved_count += 1
    else:
        print(f"  -> Notice: {f} not found, skipping.")

print(f"Manuscript export completed: {saved_count} files exported to {dist_dir}")
