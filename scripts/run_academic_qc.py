"""
run_academic_qc.py — Comprehensive Academic Quality Control Suite
Implements grammar, readability, plagiarism/overlap, style/anti-AI, and consistency checks.
"""

import sys
import re
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.writing_assistant.citation_verifier import VERIFIED_BIBLIOGRAPHY, extract_citations, verify_citations

# ---------------------------------------------------------------------------
# 1. Grammar & Style Linter
# ---------------------------------------------------------------------------
def check_grammar_and_style(text: str) -> Dict[str, Any]:
    issues = []
    
    # Strip LaTeX commands and environment markup for grammatical analysis of prose
    prose_text = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^\}]*\})*", " ", text)
    prose_text = re.sub(r"[{}\\%_#&~^]", " ", prose_text)
    
    # Check for duplicate consecutive words (e.g., "the the")
    dup_matches = re.finditer(r"\b([a-zA-Z]{2,})\s+\1\b", prose_text, re.IGNORECASE)
    for m in dup_matches:
        issues.append({
            "type": "Duplicate Word",
            "match": m.group(0),
            "context": prose_text[max(0, m.start()-20):min(len(prose_text), m.end()+20)].strip()
        })
        
    # Check for informal contractions in academic text
    contractions = ["don't", "doesn't", "can't", "won't", "it's", "they're", "we've", "you're", "isn't", "aren't"]
    for c in contractions:
        pattern = rf"\b{c}\b"
        for m in re.finditer(pattern, prose_text, re.IGNORECASE):
            issues.append({
                "type": "Informal Contraction",
                "match": m.group(0),
                "context": prose_text[max(0, m.start()-20):min(len(prose_text), m.end()+20)].strip()
            })
            
    # Check for weak academic phrases / filler
    weak_phrases = ["a lot of", "kind of", "sort of", "needless to say", "in a nutshell", "at the end of the day"]
    for wp in weak_phrases:
        pattern = rf"\b{wp}\b"
        for m in re.finditer(pattern, prose_text, re.IGNORECASE):
            issues.append({
                "type": "Colloquialism / Filler",
                "match": m.group(0),
                "context": prose_text[max(0, m.start()-20):min(len(prose_text), m.end()+20)].strip()
            })
            
    # Check for unspaced punctuation
    unspaced = re.finditer(r"[a-zA-Z],[a-zA-Z]", prose_text)
    for m in unspaced:
        issues.append({
            "type": "Missing Space after Comma",
            "match": m.group(0),
            "context": prose_text[max(0, m.start()-15):min(len(prose_text), m.end()+15)].strip()
        })
        
    return {
        "total_issues": len(issues),
        "issues": issues,
        "status": "CLEAN" if len(issues) == 0 else "ATTENTION_NEEDED"
    }

# ---------------------------------------------------------------------------
# 2. Readability & Flow Metrics
# ---------------------------------------------------------------------------
def calculate_readability(text: str) -> Dict[str, Any]:
    # Strip LaTeX commands and math
    clean_text = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^\}]*\})*", " ", text)
    clean_text = re.sub(r"\$[^$]+\$", " ", clean_text)
    clean_text = re.sub(r"[{}\\%_#&~^]", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    
    sentences = [s.strip() for s in re.split(r"[.!?]+", clean_text) if len(s.strip()) > 3]
    words = [w for w in re.findall(r"\b[a-zA-Z]+\b", clean_text)]
    
    num_sentences = max(1, len(sentences))
    num_words = max(1, len(words))
    num_chars = sum(len(w) for w in words)
    
    # Syllable estimator
    def count_syllables(word: str) -> int:
        w = word.lower()
        if len(w) <= 3:
            return 1
        w = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", w)
        w = re.sub(r"^y", "", w)
        syl = len(re.findall(r"[aeiouy]{1,2}", w))
        return max(1, syl)
        
    num_syllables = sum(count_syllables(w) for w in words)
    
    # Flesch Reading Ease
    # 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    fre = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    
    # Flesch-Kincaid Grade Level
    # 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    fkgl = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
    
    # Coleman-Liau Index
    # 0.0588 * L - 0.296 * S - 15.8 (L = chars per 100 words, S = sentences per 100 words)
    L = (num_chars / num_words) * 100
    S = (num_sentences / num_words) * 100
    cli = 0.0588 * L - 0.296 * S - 15.8
    
    avg_sentence_len = num_words / num_sentences
    lexical_diversity = len(set(w.lower() for w in words)) / num_words
    
    return {
        "num_sentences": num_sentences,
        "num_words": num_words,
        "avg_sentence_length_words": round(avg_sentence_len, 2),
        "flesch_reading_ease": round(fre, 2),
        "flesch_kincaid_grade_level": round(fkgl, 2),
        "coleman_liau_index": round(cli, 2),
        "lexical_diversity_ttr": round(lexical_diversity, 4),
        "target_academic_level": "Graduate / Scholarly (Grade 13-18)" if fkgl >= 12 else "Undergraduate / General"
    }

# ---------------------------------------------------------------------------
# 3. Plagiarism & Text Overlap (MinHash / Jaccard via Datasketch)
# ---------------------------------------------------------------------------
def check_text_overlap(manuscript_text: str) -> Dict[str, Any]:
    from datasketch import MinHash
    
    # Extract 5-grams from manuscript
    words = re.findall(r"\b[a-zA-Z]+\b", manuscript_text.lower())
    ms_shingles = set(" ".join(words[i:i+5]) for i in range(len(words)-4))
    
    ms_minhash = MinHash(num_perm=128)
    for sh in ms_shingles:
        ms_minhash.update(sh.encode("utf-8"))
        
    # Check against known external texts / prior literature notes
    lit_path = PROJECT_ROOT / "paper" / "literature_review.md"
    max_sim = 0.0
    sim_source = "None"
    
    if lit_path.exists():
        lit_text = lit_path.read_text(encoding="utf-8")
        lit_words = re.findall(r"\b[a-zA-Z]+\b", lit_text.lower())
        lit_shingles = set(" ".join(lit_words[i:i+5]) for i in range(len(lit_words)-4))
        lit_minhash = MinHash(num_perm=128)
        for sh in lit_shingles:
            lit_minhash.update(sh.encode("utf-8"))
        sim = ms_minhash.jaccard(lit_minhash)
        if sim > max_sim:
            max_sim = sim
            sim_source = "Internal Literature Review Draft"
            
    return {
        "manuscript_5gram_shingles": len(ms_shingles),
        "internal_prior_overlap_jaccard": round(max_sim, 4),
        "highest_overlap_source": sim_source,
        "plagiarism_risk_level": "LOW (Clean original composition)" if max_sim < 0.25 else "MODERATE",
    }

# ---------------------------------------------------------------------------
# 4. Anti-AI Heuristics & Tone Analysis
# ---------------------------------------------------------------------------
def check_anti_ai_heuristics(text: str) -> Dict[str, Any]:
    # Hallmark AI cliches
    cliches = [
        "delve", "testament", "tapestry", "beacon", "pivotal", "paramount",
        "game-changer", "paradigm shift", "multifaceted", "plethora", "crucial role",
        "in conclusion, it is important to remember", "it is worth noting that",
        "serves as a testament", "plays a crucial role"
    ]
    detected_cliches = []
    for c in cliches:
        for m in re.finditer(rf"\b{c}\b", text, re.IGNORECASE):
            detected_cliches.append(m.group(0))
            
    # Passive voice heuristic (form of be + past participle)
    passive_matches = re.findall(r"\b(?:is|are|was|were|been|being)\s+([a-zA-Z]+ed|[a-zA-Z]+en)\b", text, re.IGNORECASE)
    
    # Nominalization heuristic (-tion, -ment, -ance, -ence)
    nominalizations = re.findall(r"\b[a-zA-Z]{5,}(?:tion|ment|ance|ence)\b", text, re.IGNORECASE)
    
    return {
        "ai_cliches_detected": detected_cliches,
        "ai_cliche_count": len(detected_cliches),
        "passive_voice_instances": len(passive_matches),
        "nominalization_instances": len(nominalizations),
        "academic_tone_assessment": "High-Quality Academic Prose" if len(detected_cliches) <= 3 else "Review Cliches"
    }

# ---------------------------------------------------------------------------
# 5. LaTeX Formatting & Consistency Linter
# ---------------------------------------------------------------------------
def check_latex_consistency(tex_content: str) -> Dict[str, Any]:
    issues = []
    
    # Check matching braces
    open_braces = tex_content.count("{")
    close_braces = tex_content.count("}")
    if open_braces != close_braces:
        issues.append(f"Mismatched curly braces: {open_braces} open vs {close_braces} close")
        
    # Check matching math delimiters $
    dollar_count = len(re.findall(r"(?<!\\)\$", tex_content))
    if dollar_count % 2 != 0:
        issues.append(f"Unmatched math dollar signs: total {dollar_count}")
        
    # Check labels and refs
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex_content))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex_content))
    missing_labels = refs - labels
    if missing_labels:
        issues.append(f"Undefined \\ref references: {missing_labels}")
        
    # Check citations vs extract_citations
    cited_keys = extract_citations(tex_content)
    verification = verify_citations(cited_keys)
    if not verification["all_citations_verified"]:
        issues.append(f"Unverified citation keys: {verification['invalid_keys']}")
        
    return {
        "formatting_issues": issues,
        "is_consistent": len(issues) == 0,
        "labels_defined": len(labels),
        "refs_used": len(refs),
        "citations_cited": len(cited_keys),
        "all_citations_verified": verification["all_citations_verified"]
    }

if __name__ == "__main__":
    tex_path = PROJECT_ROOT / "paper" / "manuscript.tex"
    content = tex_path.read_text(encoding="utf-8")
    
    print("=== Academic Quality Control Test ===")
    g = check_grammar_and_style(content)
    print(f"Grammar Issues: {g['total_issues']}")
    for iss in g["issues"]:
        print(f"  - [{iss['type']}] '{iss['match']}' in: ...{iss['context']}...")
    
    r = calculate_readability(content)
    print(f"Readability: FKGL Grade {r['flesch_kincaid_grade_level']}, FRE {r['flesch_reading_ease']}")
    
    o = check_text_overlap(content)
    print(f"Overlap: Jaccard {o['internal_prior_overlap_jaccard']} ({o['plagiarism_risk_level']})")
    
    a = check_anti_ai_heuristics(content)
    print(f"Anti-AI Cliches: {a['ai_cliche_count']}, Passive Voice: {a['passive_voice_instances']}")
    
    c = check_latex_consistency(content)
    print(f"LaTeX Consistency: {c['is_consistent']} (Issues: {c['formatting_issues']})")
