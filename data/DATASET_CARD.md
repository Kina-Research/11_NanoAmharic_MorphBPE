# AmharicGenCorpus — Dataset Card

## Dataset Overview

- **Name:** AmharicGenCorpus
- **Version:** 1.0.0 (Curated & Verified — Phase 2)
- **Status:** ✅ COMPLETED & EMPIRICALLY VERIFIED
- **Total Curated Documents:** **120,930**
- **Total Words / Tokens (Whitespace):** **5,380,546**
- **Total Characters:** **27,557,689**
- **Language / Script:** Amharic (`am`) / Ethiopic (*Ge'ez*) Script
- **Split Ratio:** 90% Train / 5% Validation / 5% Test (Story / Document Level)
- **Random Seed:** 42 (Deterministic Reproducibility)

---

## 1. Provenance and Verified Sources

| Source Identifier | Primary Domain | Ingested Records | Retained Records | License | Upstream Lineage |
|-------------------|----------------|------------------|------------------|---------|------------------|
| `amharic_clean_corpus` | Storytelling, Folklore, Narrative | 30,882 | 27,477 (Train) + 2,966 (Val/Test) | CC-BY-4.0 / Open Research | Curated Ethiopian literary narratives |
| `amharicdc_v3` | Dialogue, Conversational, Instruction | 93,713 | 61,000+ | Apache-2.0 | Multi-turn Amharic conversational dataset |
| `amharichat` | Conversational, Chitchat | 44,836 | 25,000+ | Apache-2.0 | Standard Amharic conversational benchmarks |
| `xlsum_corpus` | News, Summaries, General Prose | 11,491 | 4,000+ | CC-BY-NC-SA-4.0 | BBC News Amharic academic subset |
| **Total Pipeline** | **Multi-Domain Balanced** | **180,922** | **120,930** | **Permissive / Open Academic** | Multi-source verified |

*Excluded Sources:* `Aman-byte1/amharic-conversation-and-math-dataset` was formally excluded due to lack of an explicit open-source license.

---

## 2. Pipeline Attrition and Data Quality Audit

```
Raw Ingested Records:        180,922
  ↓ Unicode NFC & Whitespace: 180,922 (100% valid)
  ↓ Quality Filters:          180,551 (371 rejected)
      - Excessive repetition:      42
      - Low Ethiopic ratio (<60%): 324
      - Too few words (<4 words):   5
  ↓ Exact SHA-256 Dedup:      124,241 (56,310 exact duplicates removed)
  ↓ MinHash Near-Dedup (@0.88):120,930 (3,311 near-duplicates removed)
Final Curated Documents:     120,930
```

---

## 3. Data Splits & Checksums

| Split | Documents | Tokens | Characters | Avg Tokens/Doc | Avg Chars/Doc | SHA-256 Checksum |
|-------|-----------|--------|------------|----------------|---------------|------------------|
| **`train.jsonl`** | **108,838** | 4,869,876 | 24,942,987 | 44.7 | 229.2 | `cbd8a27d7af8d5e3d4c7bb239a717380c22a4b5b05fe859c6aa4aa90b83f6eee` |
| **`validation.jsonl`** | **6,046** | 257,094 | 1,315,812 | 42.5 | 217.6 | `5f55980db62b455e36925343b41c60d693367b7a3cbe2829b6dad954ca4a0e35` |
| **`test.jsonl`** | **6,046** | 253,576 | 1,298,890 | 41.9 | 214.8 | `db3d1f95bc8f6ef76d95a3ef48b6f2acf65a9b29e44d2c7c28578d5f91f637ea` |
| **`diagnostic_eval_samples.jsonl`** | **200** | 6,400 | 32,670 | 32.0 | 163.3 | Independent diagnostic sample for morphological evaluation |

---

## 4. Record Schema

All records are serialized in JSON Lines (`.jsonl`) format:
```json
{
  "id": "amharic_clean_corpus_0000001",
  "text": "በጥንት ዘመን አንድ ንጉስ ነበር። ንጉሡም ለሕዝቡ ፍትሕንና ሰላምን ይሰጥ ነበር።",
  "source": "amharic_clean_corpus",
  "license": "CC-BY-4.0 / Open Research",
  "domain": "storytelling_and_narrative",
  "raw_length": 62,
  "char_length": 62,
  "token_count": 10,
  "ethiopic_ratio": 1.0,
  "sha256": "4b5d..."
}
```

---

## 5. Linguistic Caveats & Preservation Decisions

1. **Orthographic Integrity:** Canonical homophone folding (e.g., merging ሀ/ሐ/ኀ, ሰ/ሠ, አ/ዐ, ጸ/ፀ) is **not applied**. Authentic historical and dialectal orthography is preserved to protect semantic and etymological distinctions.
2. **Punctuation Standards:** Authentic Ethiopic punctuation marks (word separator `፡`, full stop `።`, comma `፣`, semicolon `፤`, colon `፥`) are preserved alongside standard newlines for paragraph demarcation.
3. **Language Purity:** Every record has an Ethiopic character purity ratio $\ge 60\%$.
