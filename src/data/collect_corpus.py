"""
src/data/collect_corpus.py
Raw Amharic text collection and provenance tracking.

Ingests text from verified sources and formats records into a uniform representation:
{
    "id": str,
    "text": str,
    "source": str,
    "license": str,
    "domain": str,
    "raw_length": int
}
"""

import json
import os
from pathlib import Path
from typing import Generator, Iterator

# Verified source manifests
SOURCE_MANIFESTS = {
    "amharic_clean_corpus": {
        "path": "d:/AMUPHD2015/Papers/16StorytellerLLM/data/processed/amharic_clean_corpus.txt",
        "domain": "storytelling_and_narrative",
        "license": "CC-BY-4.0 / Open Research",
        "type": "plaintext_paragraphs",
    },
    "amharicdc_v3": {
        "path": "d:/AMUPHD2015/Papers/16StorytellerLLM/data/corpus/AmharicdcV3.jsonl",
        "domain": "dialogue_and_instruction",
        "license": "Apache-2.0",
        "type": "jsonl_dialogue",
    },
    "amharichat": {
        "path": "d:/AMUPHD2015/Papers/16StorytellerLLM/data/raw/Amharichat.jsonl",
        "domain": "conversational",
        "license": "Apache-2.0",
        "type": "jsonl_dialogue",
    },
    "xlsum_corpus": {
        "path": "d:/AMUPHD2015/Papers/16StorytellerLLM/data/raw/xlsum_corpus.txt",
        "domain": "news_and_summarization",
        "license": "CC-BY-NC-SA-4.0 (Non-Commercial Academic)",
        "type": "plaintext_lines",
    },
}


def extract_dialogue_text(content: list[dict]) -> str:
    """Extract and format dialogue turns into narrative/conversational text."""
    turns = []
    for item in content:
        role = item.get("role", "")
        text = item.get("text", "").strip()
        if text:
            if role == "user":
                turns.append(f"ተጠያቂ፡ {text}")
            elif role == "assistant":
                turns.append(f"መልስ፡ {text}")
            else:
                turns.append(text)
    return " \n".join(turns)


def stream_source_records(
    source_key: str,
    max_records: int = None,
) -> Iterator[dict]:
    """
    Stream records from a verified source with metadata tracking.
    """
    manifest = SOURCE_MANIFESTS.get(source_key)
    if not manifest:
        raise ValueError(f"Unknown source: {source_key}")

    path = Path(manifest["path"])
    if not path.exists():
        print(f"Warning: Path does not exist: {path}")
        return

    domain = manifest["domain"]
    lic = manifest["license"]
    stype = manifest["type"]
    count = 0

    if stype == "jsonl_dialogue":
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if max_records and count >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("content", [])
                    if isinstance(content, list):
                        text = extract_dialogue_text(content)
                    else:
                        text = str(content)

                    if text.strip():
                        count += 1
                        yield {
                            "id": f"{source_key}_{count:07d}",
                            "text": text,
                            "source": source_key,
                            "license": lic,
                            "domain": domain,
                            "raw_length": len(text),
                        }
                except json.JSONDecodeError:
                    continue

    elif stype in ("plaintext_paragraphs", "plaintext_lines"):
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if max_records and count >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                count += 1
                yield {
                    "id": f"{source_key}_{count:07d}",
                    "text": line,
                    "source": source_key,
                    "license": lic,
                    "domain": domain,
                    "raw_length": len(line),
                }


def collect_all_sources(
    source_limits: dict[str, int] = None,
) -> Iterator[dict]:
    """
    Collect records across all registered sources.
    """
    source_limits = source_limits or {}
    for source_key in SOURCE_MANIFESTS:
        limit = source_limits.get(source_key)
        yield from stream_source_records(source_key, max_records=limit)


if __name__ == "__main__":
    # Test streaming first 3 records of each source
    for key in SOURCE_MANIFESTS:
        print(f"=== Testing {key} ===")
        sample_count = 0
        for rec in stream_source_records(key, max_records=3):
            sample_count += 1
            print(f"[{rec['id']}] ({rec['domain']}): {rec['text'][:80]}...")
        print(f"Total retrieved: {sample_count}")

