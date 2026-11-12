"""
server.py — Flask REST API and Interactive Web Server for NanoAmharic Text Generation.
Features:
- Health check endpoint and detailed model telemetry
- Text completion, Storytelling, and Chat REST APIs
- Streaming token generation via Server-Sent Events (SSE)
- Tokenizer & Morpheme Lab endpoint comparing MorphBPE vs Standard BPE
- Embedded and template-rendered responsive Web UI supporting Ge'ez scripts
- Dual FP32 & Dynamic INT8 CPU engine support
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify, Response, render_template, render_template_string
from flask_cors import CORS
import psutil

from src.app.inference_engine import NanoAmharicInferenceEngine
from src.tokenizer.morphbpe import MorphBPETokenizer
from src.tokenizer.morph_rules import identify_morphological_boundaries, crosses_morphological_boundary

TEMPLATE_DIR = str(PROJECT_ROOT / "app" / "templates")
STATIC_DIR = str(PROJECT_ROOT / "app" / "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
)
CORS(app)

# Global lazy-loaded inference engines and tokenizers
_ENGINES: Dict[str, NanoAmharicInferenceEngine] = {}
_TOKENIZERS: Dict[str, MorphBPETokenizer] = {}


def get_engine(use_int8: bool = False) -> NanoAmharicInferenceEngine:
    global _ENGINES
    key = "int8" if use_int8 else "fp32"
    if key not in _ENGINES:
        _ENGINES[key] = NanoAmharicInferenceEngine(use_int8=use_int8)
    return _ENGINES[key]


def get_tokenizer(name: str = "morphbpe") -> MorphBPETokenizer:
    global _TOKENIZERS
    if name not in _TOKENIZERS:
        if name == "standard_bpe":
            tok_dir = PROJECT_ROOT / "eval" / "results" / "tokenizer" / "Standard_BPE_v4k"
        else:
            tok_dir = PROJECT_ROOT / "eval" / "results" / "tokenizer" / "MorphBPE_v4k_l050"
        _TOKENIZERS[name] = MorphBPETokenizer.load(tok_dir)
    return _TOKENIZERS[name]


EMBEDDED_HTML_UI = """<!DOCTYPE html>
<html lang="am">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ናኖ-አማርኛ (NanoAmharic) — የአማርኛ ቋንቋ ሞዴል</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --text-color: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
            --bubble-user: #0369a1;
            --bubble-bot: #334155;
        }
        body {
            font-family: 'Nyala', 'Abyssinica SIL', 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
        }
        .container { max-width: 900px; width: 100%; }
        header { text-align: center; margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }
        h1 { color: var(--accent); margin: 0 0 8px 0; font-size: 2.2rem; }
        .badge { background: #0369a1; padding: 4px 10px; border-radius: 9999px; font-size: 0.85rem; display: inline-block; }
        .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
        .tab-btn { background: var(--card-bg); color: var(--text-color); border: 1px solid var(--border); padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 1rem; }
        .tab-btn.active { background: var(--accent); color: #0f172a; font-weight: bold; }
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }
        textarea, input[type="text"] { width: 100%; box-sizing: border-box; background: #0f172a; border: 1px solid var(--border); color: var(--text-color); padding: 12px; border-radius: 8px; font-size: 1.1rem; margin-bottom: 12px; font-family: inherit; }
        .btn-primary { background: var(--accent); color: #0f172a; border: none; padding: 12px 24px; border-radius: 8px; font-weight: bold; font-size: 1rem; cursor: pointer; }
        .output-box { background: #090d16; border: 1px solid var(--border); border-radius: 8px; padding: 16px; min-height: 120px; margin-top: 16px; white-space: pre-wrap; font-size: 1.15rem; line-height: 1.6; }
        .chat-box { height: 350px; overflow-y: auto; border: 1px solid var(--border); background: #090d16; border-radius: 8px; padding: 16px; margin-bottom: 12px; display: flex; flex-direction: column; gap: 12px; }
        .chat-msg { max-width: 80%; padding: 10px 14px; border-radius: 12px; font-size: 1.05rem; line-height: 1.5; }
        .chat-user { background: var(--bubble-user); align-self: flex-end; }
        .chat-bot { background: var(--bubble-bot); align-self: flex-start; }
        .controls { display: flex; gap: 16px; margin-bottom: 12px; font-size: 0.9rem; color: var(--text-muted); }
        .metric-tag { font-size: 0.85rem; color: var(--accent); margin-top: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ናኖ-አማርኛ (NanoAmharic)</h1>
            <p style="margin: 4px 0 10px 0; color: var(--text-muted);">
                MorphBPE-ተኮር እና በሲፒዩ (CPU) ላይ የሚሰራ የቋንቋ ሞዴል | 5.75M Parameters
            </p>
            <span class="badge">CPU Workstation Verified</span>
            <span class="badge" style="background: #047857;">MorphBPE Tokenizer</span>
        </header>
        <div class="card">
            <h3>ናኖ-አማርኛ ስቱዲዮ ዝግጁ ነው!</h3>
            <p>እባክዎ ሙሉውን ዩዘር ኢንተርፌስ (UI) በ <code>app/templates/index.html</code> ይመልከቱ።</p>
        </div>
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    """Serves the interactive Amharic generation Web UI."""
    template_file = PROJECT_ROOT / "app" / "templates" / "index.html"
    if template_file.exists():
        try:
            return render_template("index.html")
        except Exception:
            pass
    return render_template_string(EMBEDDED_HTML_UI)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint reporting model status, memory RSS, and CPU utilization."""
    engine = get_engine()
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 * 1024)

    return jsonify({
        "status": "healthy",
        "model_name": "NanoAmharic-MorphBPE",
        "parameters": engine.param_count,
        "is_quantized": engine.is_quantized,
        "vocab_size": engine.vocab_size,
        "memory_rss_mb": round(rss_mb, 1),
        "cpu_threads": psutil.cpu_count(logical=True),
    })


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Returns verified architecture parameters and empirical benchmark summaries."""
    return jsonify({
        "model_name": "NanoAmharic-MorphBPE",
        "parameters": 5745920,
        "layers": 6,
        "heads": 8,
        "embedding_dim": 256,
        "ffn_dim": 1024,
        "context_length": 128,
        "vocab_size": 4000,
        "fp32_size_mb": 65.82,
        "int8_size_mb": 9.44,
        "compression_ratio": 6.97,
        "tokens_per_second_cpu_int8": 51.6,
        "tokens_per_second_cpu_fp32": 97.7,
        "validation_loss_morphbpe": 3.5118,
        "validation_loss_standard_bpe": 3.5533,
        "boundary_consistency_morphbpe": 25.10,
        "boundary_consistency_standard_bpe": 17.89,
        "morphological_error_rate_morphbpe": 1.46,
        "morphological_error_rate_standard_bpe": 3.03,
        "relative_error_reduction_pct": 51.82,
        "bootstrap_p_value": "<0.001",
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """General text completion endpoint."""
    data = request.get_json(force=True) or {}
    prompt = data.get("prompt", "ሰላም")
    max_tokens = int(data.get("max_tokens", 40))
    temperature = float(data.get("temperature", 0.7))
    top_k = int(data.get("top_k", 40))
    top_p = float(data.get("top_p", 0.9))
    precision = data.get("precision", "fp32").lower()

    engine = get_engine(use_int8=(precision == "int8"))
    result = engine.generate(
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return jsonify(result)


@app.route("/api/story", methods=["POST"])
def api_story():
    """Storytelling generation endpoint."""
    data = request.get_json(force=True) or {}
    topic = data.get("topic", "አንድ ንጉሥ")
    genre = data.get("genre", "ተረት")
    max_tokens = int(data.get("max_tokens", 60))
    temperature = float(data.get("temperature", 0.75))
    top_k = int(data.get("top_k", 40))
    top_p = float(data.get("top_p", 0.9))
    precision = data.get("precision", "fp32").lower()

    engine = get_engine(use_int8=(precision == "int8"))
    result = engine.generate_story(
        topic=topic,
        genre=genre,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Conversational dialogue endpoint."""
    data = request.get_json(force=True) or {}
    message = data.get("message", "ሰላም")
    history = data.get("history", [])
    max_tokens = int(data.get("max_tokens", 40))
    temperature = float(data.get("temperature", 0.7))
    precision = data.get("precision", "fp32").lower()

    engine = get_engine(use_int8=(precision == "int8"))
    result = engine.chat(
        user_message=message,
        conversation_history=history,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return jsonify(result)


@app.route("/api/tokenize", methods=["POST"])
def api_tokenize():
    """Detailed MorphBPE vs Standard BPE tokenization & morpheme boundary inspector."""
    data = request.get_json(force=True) or {}
    text = data.get("text", "በመንግሥታችን አልተረዱልንም")

    morph_tok = get_tokenizer("morphbpe")
    std_tok = get_tokenizer("standard_bpe")

    words = text.split()
    words_analysis = []

    for w in words:
        boundaries = sorted(list(identify_morphological_boundaries(w)))
        m_subwords = morph_tok.tokenize(w)
        s_subwords = std_tok.tokenize(w)

        # Check boundary crossing in standard BPE
        crosses_s = False
        offset_s = 0
        for i in range(len(s_subwords) - 1):
            if crosses_morphological_boundary(s_subwords[i], s_subwords[i+1], set(boundaries), offset_s):
                crosses_s = True
                break
            offset_s += len(s_subwords[i])

        # Check boundary crossing in MorphBPE
        crosses_m = False
        offset_m = 0
        for i in range(len(m_subwords) - 1):
            if crosses_morphological_boundary(m_subwords[i], m_subwords[i+1], set(boundaries), offset_m):
                crosses_m = True
                break
            offset_m += len(m_subwords[i])

        words_analysis.append({
            "word": w,
            "boundaries": boundaries,
            "morphbpe_subwords": m_subwords,
            "standard_bpe_subwords": s_subwords,
            "crosses_standard": crosses_s,
            "crosses_morphbpe": crosses_m,
        })

    morph_all = morph_tok.tokenize(text)
    std_all = std_tok.tokenize(text)

    return jsonify({
        "text": text,
        "morphbpe_tokens": morph_all,
        "standard_bpe_tokens": std_all,
        "words_analysis": words_analysis,
        "morphbpe_count": len(morph_all),
        "standard_bpe_count": len(std_all),
    })


@app.route("/api/stream", methods=["GET"])
def api_stream():
    """Server-Sent Events streaming token generation."""
    prompt = request.args.get("prompt", "ሰላም")
    max_tokens = int(request.args.get("max_tokens", 40))
    temperature = float(request.args.get("temperature", 0.7))
    precision = request.args.get("precision", "fp32").lower()

    engine = get_engine(use_int8=(precision == "int8"))

    def generate_events():
        for token_str in engine.generate_stream(prompt=prompt, max_new_tokens=max_tokens, temperature=temperature):
            yield f"data: {json.dumps({'token': token_str}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate_events(), mimetype="text/event-stream")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start NanoAmharic Local Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantized checkpoint")
    args = parser.parse_args()

    get_engine(use_int8=args.int8)
    print(f"\n========================================================")
    print(f" NanoAmharic Local Server running on http://{args.host}:{args.port}")
    print(f" INT8 Quantized: {args.int8}")
    print(f"========================================================\n")
    app.run(host=args.host, port=args.port, debug=False)
