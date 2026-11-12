"""
src/tokenizer/morph_rules.py
Morphological boundary heuristics and rule sets for Amharic.

Amharic exhibits a non-concatenative root-and-pattern morphology combined with
frequent concatenative cliticization (prefixes and suffixes).

This module defines:
1. Validated candidate prefixes and suffixes.
2. Boundary detection functions that mark morpheme transition indices in a word.
3. Boundary-crossing predicate: b(p, q) in {0, 1} for candidate pair merges.
"""

from typing import List, Set, Tuple

# ── Verified Candidate Prefixes ──────────────────────────────────────────────
# Covers genitive/relative, prepositions, negation, conjunctions, aspectual markers
AMHARIC_PREFIXES: List[str] = [
    # Multi-character prefixes (longer prefixes checked first)
    "እንደ",  # əndä- (comparative / as, like)
    "ስለ",   # səlä- (causal / about, because of)
    "አል",   # al- (negative verbal prefix)
    "እየ",   # əyä- (continuous / progressive aspect)
    "እስከ",  # əskä- (terminative / until, up to)
    # Single-character prefixes
    "የ",    # yä- (genitive / relative marker)
    "በ",    # bä- (locative / instrumental / in, with)
    "ለ",    # lä- (dative / purposive / to, for)
    "ከ",    # kä- (ablative / from, than)
    "ወ",    # wä- (coordinating conjunction / and)
    "ስ",    # sə- (temporal / conditional / when, if)
    "እ",    # ə- (prothetic vowel prefix)
    "ያ",    # ya- (variant relativizer)
    "ይ",    # yə- (3ms imperfect prefix)
    "ት",    # tə- (3fs / 2ms imperfect prefix)
    "እን",   # ən- (1pl imperfect prefix)
    "አ",    # a- (1s imperfect / causative prefix)
]

# ── Verified Candidate Suffixes ──────────────────────────────────────────────
# Covers nominal plural, definite articles, pronominal objects, possessives, conjunctions
AMHARIC_SUFFIXES: List[str] = [
    # Multi-character suffixes
    "አላቸው", # -alatšäw (3pl indirect object)
    "አቸው",  # -atšäw (3pl object / possessive)
    "አችን",  # -atšən (1pl possessive)
    "አችሁ",  # -atšəhu (2pl possessive)
    "ዎች",   # -wočč (plural marker)
    "ኦች",   # -očč (plural marker)
    "ተኛ",   # -täňňa (agentive / ordinal)
    "ነት",   # -nät (abstract nominalizer)
    "ኛ",    # -ňňa (adjectival / language marker)
    # Single-character suffixes
    "ው",    # -w / -u (3ms definite article / object)
    "ዋ",    # -wa (3fs definite article / object)
    "ም",    # -əm (coordinating / negative auxiliary / also, even)
    "ና",    # -nna (conjunction / and)
    "ኡ",    # -u (3ms definite article / 3pl subject)
    "ኤ",    # -e (1s possessive)
    "ህ",    # -h (2ms possessive)
    "ሽ",    # -š (2fs possessive)
]

# Sort by decreasing length for greedy prefix/suffix matching
PREFIX_SET_SORTED = sorted(AMHARIC_PREFIXES, key=len, reverse=True)
SUFFIX_SET_SORTED = sorted(AMHARIC_SUFFIXES, key=len, reverse=True)


def identify_morphological_boundaries(word: str) -> Set[int]:
    """
    Identify potential morpheme boundary character offsets within a single word.
    A boundary index k means a boundary exists between word[:k] and word[k:].

    For example, in "የቤት" (yä-bet):
      word = "የቤት", boundary = {1}
    In "ልጆች" (ləǧ-očč):
      word = "ልጆች", boundary = {1} (after stem)
    In "አልነገረም" (al-nägär-əm):
      word = "አልነገረም", boundaries = {2, 5}

    Heuristics:
    - Min word length: 3 characters (avoid over-segmenting monosyllables).
    - Preserves a minimum stem length of at least 2 characters.
    """
    boundaries: Set[int] = set()
    n = len(word)
    if n < 3:
        return boundaries

    # Prefix detection: word[:k] in prefixes and remaining stem >= 2 chars
    for pfx in PREFIX_SET_SORTED:
        plen = len(pfx)
        if word.startswith(pfx) and (n - plen) >= 2:
            boundaries.add(plen)
            break  # Match longest prefix

    # Suffix detection: word[k:] in suffixes and preceding stem >= 2 chars
    for sfx in SUFFIX_SET_SORTED:
        slen = len(sfx)
        if word.endswith(sfx) and (n - slen) >= 2:
            boundaries.add(n - slen)
            break  # Match longest suffix

    return boundaries


def crosses_morphological_boundary(
    left_token: str,
    right_token: str,
    boundary_offsets: Set[int],
    token_start_offset: int,
) -> bool:
    """
    Determine if merging `left_token` and `right_token` would cross any
    identified morpheme boundary within the word.

    Args:
        left_token: The left subword string.
        right_token: The right subword string.
        boundary_offsets: Set of boundary character offsets in the word.
        token_start_offset: The character start offset of `left_token` in the word.

    Returns:
        True if the merge boundary (token_start_offset + len(left_token))
        coincides with an identified morpheme boundary.
    """
    if not boundary_offsets:
        return False

    merge_boundary = token_start_offset + len(left_token)
    return merge_boundary in boundary_offsets


if __name__ == "__main__":
    test_words = [
        "የቤት",         # yä-bet (boundary at 1)
        "በከተማ",        # bä-kätäma (boundary at 1)
        "አልነገረም",      # al-nägär-əm (boundary at 2 and 5)
        "ልጆች",         # ləǧ-očč (boundary at 1)
        "ኢትዮጵያዊነት",    # ityop'yawi-nät (boundary at 9)
        "ሰላም",         # root noun (no boundary)
    ]
    print("=== Morphological Boundary Detection Demo ===")
    for w in test_words:
        b = identify_morphological_boundaries(w)
        print(f"Word: {w:<12} Length: {len(w)} Boundaries: {b}")

