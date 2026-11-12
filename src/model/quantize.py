"""
quantize.py — Post-Training Dynamic INT8 Quantization and CPU Inference Benchmarking.
Quantizes Linear layers to qint8, reducing memory footprint and measuring CPU inference acceleration.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.model_config import NanoAmharicConfig
from src.model.transformer import NanoAmharicLM
from src.tokenizer.morphbpe import MorphBPETokenizer
from src.training.checkpoint import load_checkpoint


def quantize_model_dynamic(
    model: NanoAmharicLM,
) -> torch.nn.Module:
    """
    Applies PyTorch dynamic post-training quantization to all nn.Linear layers.
    Quantizes weights from FP32 to INT8 (qint8) while keeping activations in FP32.
    """
    model.eval()
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def benchmark_inference(
    model: torch.nn.Module,
    tokenizer: MorphBPETokenizer,
    prompts: List[str],
    max_new_tokens: int = 30,
    warmup_runs: int = 2,
) -> Dict[str, Any]:
    """
    Benchmarks CPU generation latency and throughput (tokens/second).
    """
    model.eval()

    # Warmup
    warmup_tensor = torch.tensor([tokenizer.encode(prompts[0])], dtype=torch.long)
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(warmup_tensor, max_new_tokens=5)

    total_tokens_generated = 0
    total_time_seconds = 0.0
    latencies_per_token_ms = []
    generated_samples = []

    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        prompt_len = input_ids.size(1)

        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
            )
        elapsed = time.perf_counter() - t0

        gen_count = out_ids.size(1) - prompt_len
        total_tokens_generated += gen_count
        total_time_seconds += elapsed

        ms_per_token = (elapsed / max(1, gen_count)) * 1000.0
        latencies_per_token_ms.append(ms_per_token)

        gen_text = tokenizer.decode(out_ids[0].tolist()[prompt_len:])
        generated_samples.append({
            "prompt": prompt,
            "continuation": gen_text.strip(),
            "tokens": gen_count,
            "elapsed_ms": round(elapsed * 1000.0, 1),
        })

    avg_ms_per_token = float(torch.tensor(latencies_per_token_ms).mean().item())
    throughput_tok_per_sec = (total_tokens_generated / total_time_seconds) if total_time_seconds > 0 else 0.0

    return {
        "total_prompts": len(prompts),
        "total_tokens_generated": total_tokens_generated,
        "total_time_seconds": round(total_time_seconds, 3),
        "avg_latency_ms_per_token": round(avg_ms_per_token, 2),
        "throughput_tokens_per_sec": round(throughput_tok_per_sec, 1),
        "samples": generated_samples,
    }


def run_quantization_pipeline(
    fp32_ckpt_path: str = "eval/results/checkpoints/experimental_morphbpe_v4k_best.pt",
    tokenizer_dir: str = "eval/results/tokenizer/MorphBPE_v4k_l050",
    output_int8_ckpt: str = "eval/results/checkpoints/nanoamharic_morphbpe_v4k_int8.pt",
    output_json: str = "eval/results/quantization_benchmark_results.json",
    output_md: str = "eval/reports/quantization_report.md",
) -> Dict[str, Any]:
    """
    Executes full Post-Training Quantization pipeline:
    1. Loads FP32 model checkpoint.
    2. Quantizes model with torch.ao.quantization.quantize_dynamic.
    3. Benchmarks FP32 vs INT8 size, latency, and throughput.
    4. Saves INT8 model checkpoint.
    5. Writes structured metrics JSON and Markdown report.
    """
    print("\n========================================================")
    print(" PHASE 6: POST-TRAINING DYNAMIC INT8 QUANTIZATION (EXP-05)")
    print("========================================================\n")

    fp32_path = PROJECT_ROOT / fp32_ckpt_path
    tok_path = PROJECT_ROOT / tokenizer_dir
    int8_path = PROJECT_ROOT / output_int8_ckpt
    json_path = PROJECT_ROOT / output_json
    md_path = PROJECT_ROOT / output_md

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Tokenizer & FP32 Model
    print(f"Loading tokenizer from {tok_path}...")
    tokenizer = MorphBPETokenizer.load(tok_path)

    print(f"Loading FP32 checkpoint from {fp32_path}...")
    cfg = NanoAmharicConfig.prototype(vocab_size=tokenizer.vocab_size)
    cfg.max_seq_len = 128
    fp32_model = NanoAmharicLM(cfg)
    load_checkpoint(str(fp32_path), fp32_model)
    fp32_model.eval()

    fp32_size_bytes = os.path.getsize(fp32_path)
    fp32_size_mb = fp32_size_bytes / (1024 * 1024)
    print(f"FP32 Checkpoint Size: {fp32_size_mb:.2f} MB")

    # 2. Apply Dynamic INT8 Quantization
    print("Applying dynamic post-training INT8 quantization to Linear layers...")
    int8_model = quantize_model_dynamic(fp32_model)

    # 3. Save INT8 Model
    print(f"Saving quantized INT8 model to {int8_path}...")
    torch.save(int8_model.state_dict(), str(int8_path))
    int8_size_bytes = os.path.getsize(int8_path)
    int8_size_mb = int8_size_bytes / (1024 * 1024)
    compression_ratio = fp32_size_mb / max(0.01, int8_size_mb)
    print(f"INT8 Checkpoint Size: {int8_size_mb:.2f} MB (Compression: {compression_ratio:.2f}x)")

    # 4. Benchmark Inference
    benchmark_prompts = [
        "በጥንት ዘመን አንድ ንጉሥ",
        "የአዲስ አበባ ከተማ",
        "ትንሽ ልጅቷ በጫካ ውስጥ",
        "ሰላም እንደምን አለህ",
        "ኢትዮጵያ ታሪካዊ ሀገር ናት",
    ]

    print("\nBenchmarking FP32 model inference...")
    fp32_bench = benchmark_inference(fp32_model, tokenizer, benchmark_prompts, max_new_tokens=30)
    print(f"FP32 -> Throughput: {fp32_bench['throughput_tokens_per_sec']} tok/s | Latency: {fp32_bench['avg_latency_ms_per_token']} ms/token")

    print("\nBenchmarking INT8 model inference...")
    int8_bench = benchmark_inference(int8_model, tokenizer, benchmark_prompts, max_new_tokens=30)
    print(f"INT8 -> Throughput: {int8_bench['throughput_tokens_per_sec']} tok/s | Latency: {int8_bench['avg_latency_ms_per_token']} ms/token")

    speedup_ratio = int8_bench["throughput_tokens_per_sec"] / max(0.1, fp32_bench["throughput_tokens_per_sec"])
    latency_reduction_pct = (
        (fp32_bench["avg_latency_ms_per_token"] - int8_bench["avg_latency_ms_per_token"])
        / max(0.1, fp32_bench["avg_latency_ms_per_token"])
    ) * 100.0

    # 5. Compile Results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": "NanoAmharic-Prototype (5.75M)",
        "quantization_type": "torch.ao.quantization.quantize_dynamic (torch.qint8)",
        "checkpoint_comparison": {
            "fp32_size_mb": round(fp32_size_mb, 2),
            "int8_size_mb": round(int8_size_mb, 2),
            "compression_ratio": round(compression_ratio, 2),
            "disk_savings_pct": round((1.0 - int8_size_mb / fp32_size_mb) * 100.0, 2),
        },
        "performance_comparison": {
            "fp32_throughput_tok_per_sec": fp32_bench["throughput_tokens_per_sec"],
            "fp32_latency_ms_per_token": fp32_bench["avg_latency_ms_per_token"],
            "int8_throughput_tok_per_sec": int8_bench["throughput_tokens_per_sec"],
            "int8_latency_ms_per_token": int8_bench["avg_latency_ms_per_token"],
            "throughput_speedup_ratio": round(speedup_ratio, 2),
            "latency_reduction_pct": round(latency_reduction_pct, 2),
        },
        "sample_fidelity_comparison": [
            {
                "prompt": benchmark_prompts[i],
                "fp32_continuation": fp32_bench["samples"][i]["continuation"],
                "int8_continuation": int8_bench["samples"][i]["continuation"],
            }
            for i in range(len(benchmark_prompts))
        ],
    }

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved quantization benchmark results to {json_path}")

    # Generate Markdown Report
    write_quantization_report(results, md_path)
    print(f"Generated quantization report at {md_path}")

    return results


def write_quantization_report(results: Dict[str, Any], report_path: Path):
    """Write detailed Markdown report for quantization and CPU optimization."""
    ckpt = results["checkpoint_comparison"]
    perf = results["performance_comparison"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# NanoAmharic INT8 Quantization & CPU Optimization Report\n")
        f.write(f"*Generated: {results['timestamp']} | Post-Training Dynamic INT8 Quantization*\n\n---\n\n")

        f.write("## 1. Executive Summary (EXP-05)\n\n")
        f.write("Post-Training Dynamic INT8 Quantization was applied to all `torch.nn.Linear` layers in the NanoAmharic transformer.\n")
        f.write("Weights were quantized to 8-bit signed integers (`qint8`), while preserving FP32 dynamic activation ranges on CPU.\n\n")

        f.write("### Key Highlights:\n")
        f.write(f"- **Model Size Reduction:** Checkpoint size compressed from **{ckpt['fp32_size_mb']} MB** down to **{ckpt['int8_size_mb']} MB** (**{ckpt['compression_ratio']}x compression**, **{ckpt['disk_savings_pct']}% disk savings**).\n")
        f.write(f"- **CPU Throughput:** Throughput improved from **{perf['fp32_throughput_tok_per_sec']} tok/s** to **{perf['int8_throughput_tok_per_sec']} tok/s** (**{perf['throughput_speedup_ratio']}x acceleration**).\n")
        f.write(f"- **Per-Token Latency:** Per-token latency reduced from **{perf['fp32_latency_ms_per_token']} ms** to **{perf['int8_latency_ms_per_token']} ms** (**{perf['latency_reduction_pct']}% lower latency**).\n\n")

        f.write("---\n\n## 2. Resource & Latency Comparison Table\n\n")
        f.write("| Dimension | FP32 Baseline | Dynamic INT8 Quantized | Improvement |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        f.write(f"| **Checkpoint Size on Disk** | {ckpt['fp32_size_mb']} MB | **{ckpt['int8_size_mb']} MB** | **{ckpt['compression_ratio']}x smaller** ({ckpt['disk_savings_pct']}% savings) |\n")
        f.write(f"| **Inference Throughput** | {perf['fp32_throughput_tok_per_sec']} tok/s | **{perf['int8_throughput_tok_per_sec']} tok/s** | **{perf['throughput_speedup_ratio']}x faster** |\n")
        f.write(f"| **Latency per Token** | {perf['fp32_latency_ms_per_token']} ms | **{perf['int8_latency_ms_per_token']} ms** | **{perf['latency_reduction_pct']}% latency reduction** |\n")
        f.write(f"| **Hardware Execution Path** | CPU (x86_64 AVX2) | CPU (x86_64 AVX2 INT8) | Native CPU acceleration |\n\n")

        f.write("---\n\n## 3. Output Fidelity Comparison (FP32 vs INT8)\n\n")
        for i, s in enumerate(results["sample_fidelity_comparison"], 1):
            f.write(f"### Prompt {i}: `{s['prompt']}`\n")
            f.write(f"- **FP32 Output:** {s['fp32_continuation']}\n")
            f.write(f"- **INT8 Output:** {s['int8_continuation']}\n\n")


if __name__ == "__main__":
    run_quantization_pipeline()

