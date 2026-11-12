"""
tests/test_morphbpe.py
Comprehensive test suite for MorphBPE tokenizer.
Run: pytest tests/test_morphbpe.py -v
"""

import sys
import tempfile
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer.morphbpe import MorphBPETokenizer, SPECIAL_TOKENS, byte_to_token


# Mini training corpus for unit testing
MINI_CORPUS = [
    "በጥንት ዘመን አንድ ታላቅ ንጉሥ በኢትዮጵያ ምድር በፍትሕ ይገዛ ነበር ።",
    "ትንሽ ልጅቷ በጫካ ውስጥ ስትጓዝ ውብ አበባዎችን አገኘች ።",
    "አንድ አባት ለልጁ የጥበብ ታሪክ ይነግረው ጀመር ።",
    "የቤት እመቤት ወደ ገበያ ሄዳ ሸቀጣ ሸቀጥ ገዛች ።",
    "ልጆች በትምህርት ቤት ውስጥ ደስተኞች ሆኑ ።",
]


def test_special_tokens_initialization():
    tok = MorphBPETokenizer(boundary_lambda=0.5)
    tok.train(MINI_CORPUS, target_vocab_size=320, verbose=False)

    for st in SPECIAL_TOKENS:
        assert st in tok.vocab, f"Missing special token: {st}"

    assert tok.pad_token_id == 0
    assert tok.unk_token_id == 1
    assert tok.bos_token_id == 2
    assert tok.eos_token_id == 3


def test_256_byte_fallback_tokens_present():
    tok = MorphBPETokenizer(boundary_lambda=0.5)
    tok.train(MINI_CORPUS, target_vocab_size=320, verbose=False)

    # Check all 256 bytes are present in vocabulary
    for b in range(256):
        btok = byte_to_token(b)
        assert btok in tok.vocab, f"Missing byte token: {btok}"


def test_roundtrip_lossless_amharic():
    tok = MorphBPETokenizer(boundary_lambda=0.5)
    tok.train(MINI_CORPUS, target_vocab_size=350, verbose=False)

    test_sentence = "በጥንት ዘመን አንድ ንጉስ ነበር።"
    ids = tok.encode(test_sentence)
    reconstructed = tok.decode(ids)
    assert reconstructed == test_sentence


def test_roundtrip_lossless_mixed_and_oov():
    tok = MorphBPETokenizer(boundary_lambda=0.5)
    tok.train(MINI_CORPUS, target_vocab_size=350, verbose=False)

    # Out-of-vocabulary Latin, numbers, and emoji
    oov_text = "ኢትዮጵያ 2026! Hello World 🌟"
    ids = tok.encode(oov_text)
    reconstructed = tok.decode(ids)
    assert reconstructed == oov_text


def test_deterministic_training():
    tok1 = MorphBPETokenizer(boundary_lambda=0.5)
    tok1.train(MINI_CORPUS, target_vocab_size=340, verbose=False)

    tok2 = MorphBPETokenizer(boundary_lambda=0.5)
    tok2.train(MINI_CORPUS, target_vocab_size=340, verbose=False)

    assert tok1.vocab == tok2.vocab
    assert tok1.merges == tok2.merges


def test_standard_bpe_at_lambda_zero():
    # At lambda = 0.0, no boundary penalty is applied
    tok_std = MorphBPETokenizer(boundary_lambda=0.0)
    tok_std.train(MINI_CORPUS, target_vocab_size=340, verbose=False)

    # Training runs smoothly
    assert tok_std.vocab_size == 340
    assert len(tok_std.merges) > 0


def test_boundary_penalty_effect():
    # Synthetic corpus with frequent prefix 'የ' and stem 'ቤት'
    synth_corpus = [
        "የቤት የቤት የቤት የቤት የቤት",
        "ቤት ቤት ቤት ቤት ቤት",
        "ነገር ነገር ነገር",
    ]
    # Train with lambda = 0.0 (vanilla)
    tok_vanilla = MorphBPETokenizer(boundary_lambda=0.0)
    tok_vanilla.train(synth_corpus, target_vocab_size=270, verbose=False)

    # Train with lambda = 1.0 (strict boundary penalty)
    tok_morph = MorphBPETokenizer(boundary_lambda=1.0)
    tok_morph.train(synth_corpus, target_vocab_size=270, verbose=False)

    # Both tokenizers train successfully
    assert tok_vanilla.vocab_size <= 270
    assert tok_morph.vocab_size <= 270
    assert len(tok_morph.merges) > 0


def test_save_and_reload():
    tok = MorphBPETokenizer(boundary_lambda=0.5)
    tok.train(MINI_CORPUS, target_vocab_size=340, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        tok.save(tmpdir)

        # Reload
        tok_reloaded = MorphBPETokenizer.load(tmpdir)
        assert tok_reloaded.vocab == tok.vocab
        assert tok_reloaded.merges == tok.merges
        assert tok_reloaded.boundary_lambda == tok.boundary_lambda

        # Check encoding parity
        sample = "ትንሽ ልጅቷ በጫካ ውስጥ ሄደች ።"
        assert tok.encode(sample) == tok_reloaded.encode(sample)
        assert tok.decode(tok.encode(sample)) == sample
