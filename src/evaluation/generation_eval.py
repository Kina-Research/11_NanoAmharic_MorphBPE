"""
generation_eval.py — Text generation benchmarking, diversity metrics (Distinct-1, Distinct-2), and repetition analysis.
"""

from typing import List, Dict, Any, Set
import torch

from src.model.transformer import NanoAmharicLM
from src.tokenizer.morphbpe import MorphBPETokenizer


def compute_diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Computes Distinct-1, Distinct-2, and 3-gram repetition rate across generated texts.
    """
    total_unigrams = 0
    unique_unigrams: Set[str] = set()

    total_bigrams = 0
    unique_bigrams: Set[str] = set()

    total_trigrams = 0
    repeated_trigrams = 0
    trigram_counts: Dict[str, int] = {}

    total_length_words = 0

    for text in texts:
        words = text.split()
        total_length_words += len(words)

        # Unigrams
        for w in words:
            total_unigrams += 1
            unique_unigrams.add(w)

        # Bigrams
        for i in range(len(words) - 1):
            bg = f"{words[i]}_{words[i+1]}"
            total_bigrams += 1
            unique_bigrams.add(bg)

        # Trigrams
        doc_trigrams: Set[str] = set()
        for i in range(len(words) - 2):
            tg = f"{words[i]}_{words[i+1]}_{words[i+2]}"
            total_trigrams += 1
            if tg in doc_trigrams:
                repeated_trigrams += 1
            else:
                doc_trigrams.add(tg)

    distinct_1 = (len(unique_unigrams) / total_unigrams) if total_unigrams > 0 else 0.0
    distinct_2 = (len(unique_bigrams) / total_bigrams) if total_bigrams > 0 else 0.0
    repetition_rate = (repeated_trigrams / total_trigrams) if total_trigrams > 0 else 0.0
    avg_length = (total_length_words / len(texts)) if texts else 0.0

    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "repetition_3gram_rate": round(repetition_rate, 4),
        "average_length_words": round(avg_length, 1),
        "total_generated_words": total_length_words,
    }


def generate_continuations(
    model: NanoAmharicLM,
    tokenizer: MorphBPETokenizer,
    prompts: List[str],
    max_new_tokens: int = 40,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> List[Dict[str, str]]:
    """
    Generates continuations for a list of prompts.
    Returns list of dicts with 'prompt' and 'generated_continuation'.
    """
    model.eval()
    results = []

    for prompt in prompts:
        input_tokens = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_tokens], dtype=torch.long)

        with torch.no_grad():
            out_ids = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        gen_tokens = out_ids[0].tolist()
        continuation_tokens = gen_tokens[len(input_tokens):]
        continuation_text = tokenizer.decode(continuation_tokens)
        full_text = tokenizer.decode(gen_tokens)

        results.append({
            "prompt": prompt,
            "continuation": continuation_text.strip(),
            "full_text": full_text.strip(),
        })

    return results

