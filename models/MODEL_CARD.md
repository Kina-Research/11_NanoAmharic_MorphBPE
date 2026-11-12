# Model Card: NanoAmharic-MorphBPE

## Model Summary
**NanoAmharic-MorphBPE** is a lightweight, morphology-constrained decoder-only transformer language model designed specifically for low-resource Amharic text generation. Trained and evaluated exclusively on a commodity multicore CPU workstation with zero GPU requirements, NanoAmharic combines **MorphBPE** (a boundary-penalized subword tokenizer) with a modernized nano-scale causal transformer architecture.

- **Developer:** Research Project NanoAmharic
- **Model Type:** Causal Decoder-Only Transformer (Nano-scale)
- **Primary Language:** Amharic (`am` / Ethiopic script)
- **License:** MIT License
- **Hardware Requirement:** Commodity CPU (x86_64 or ARM, 4+ cores, >= 2 GB RAM)

---

## Model Architecture Specifications

| Parameter | Specification | Details / Rationale |
|---|---|---|
| **Parameters** | 5,745,920 (~5.75M) | Parameter budget strictly under 6.0M |
| **Layers ($N$)** | 6 | Balanced depth for CPU cache efficiency |
| **Hidden Dim ($d_{model}$)** | 256 | Embeddings & residual streams |
| **Attention Heads ($H$)** | 8 | Multi-Query / Multi-Head attention ($d_k = 32$) |
| **FFN Dimension ($d_{ffn}$)** | 1,024 | SwiGLU activation ($4 \times d_{model}$) |
| **Positional Encoding** | Rotary Embeddings (RoPE) | Relative position invariance |
| **Normalization** | RMSNorm ($\epsilon = 10^{-5}$) | Faster than LayerNorm on CPU |
| **Weight Tying** | Yes | Input embeddings tied to output LM head |
| **Context Length** | 128 tokens | Optimized for paragraph & dialogue contexts |
| **Vocabulary Size ($V$)** | 4,000 subwords | MorphBPE ($\lambda = 0.50$) + 256 byte fallback |

---

## Checkpoints and Artifacts

| Checkpoint Name | Precision | Disk Size | Location | Step / Val Loss |
|---|---|---|---|---|
| **Experimental MorphBPE Best** | FP32 | 65.82 MB | `eval/results/checkpoints/experimental_morphbpe_v4k_best.pt` | Step 500 / 3.5118 |
| **Baseline Standard BPE Best** | FP32 | 65.82 MB | `eval/results/checkpoints/baseline_standard_bpe_v4k_best.pt` | Step 500 / 3.5533 |
| **NanoAmharic INT8 Quantized** | INT8 Dynamic | 9.44 MB | `eval/results/checkpoints/nanoamharic_morphbpe_v4k_int8.pt` | 6.97× compression |
| **MorphBPE Tokenizer** | Vocab/Merges | ~130 KB | `eval/results/tokenizer/MorphBPE_v4k_l050/` | $\lambda=0.50$, $V=4k$ |

---

## Quickstart: Python Inference

```python
from src.app.inference_engine import NanoAmharicInferenceEngine

# Load FP32 model
engine = NanoAmharicInferenceEngine(use_int8=False)

# Or load INT8 dynamic quantized model (only 9.44 MB disk, ~120 MB RAM)
# engine = NanoAmharicInferenceEngine(use_int8=True)

# 1. Text Continuation
res = engine.generate("በጥንት ዘመን አንድ ንጉሥ", max_new_tokens=40)
print(res["full_text"])

# 2. Amharic Storytelling
story = engine.generate_story(topic="ጥበበኛው ገበሬ", max_tokens=60)
print(story["full_text"])

# 3. Conversational Dialogue
chat = engine.chat("ሰላም እንደምን ነህ?", max_tokens=30)
print(chat["assistant_response"])
```

---

## Tokenizer Details: MorphBPE

Standard BPE merges pairs strictly by raw co-occurrence frequency $f(p, q)$, which frequently shatters Semitic root morphemes (e.g. merging prefix `በ-` with the first consonant of root stem `መንግሥት`). 

MorphBPE penalizes merges that cross identified morphological boundaries:
$$\mathcal{S}(p, q) = f(p, q) \cdot (1 - \lambda \cdot b(p, q))$$

At $\lambda = 0.50$:
- **Morphological Boundary Consistency:** **25.10%** (vs 17.89% Standard BPE, +40.3% relative gain)
- **Linguistic Fertility:** 1.585 tokens/word (nearly identical to 1.575)
- **Root Stem Preservation:** **97.54%** (vs 94.96%)
- **Diagnostic Anomaly Reduction:** **-51.82%** relative reduction ($p < 0.001$)

---

## Quantization & CPU Deployment

The model utilizes dynamic range quantization (`torch.ao.quantization.quantize_dynamic` targeting `torch.nn.Linear` layers into `torch.qint8`):
- **Compression:** 65.82 MB $\rightarrow$ 9.44 MB (85.65% space reduction)
- **RAM Working Set:** $\sim 120$ MB RSS
- **Throughput:** $51.6$ tokens/sec on Intel Core i7 CPU
- **Portability:** Zero GPU / CUDA runtime dependencies.

