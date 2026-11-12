"""
test_deployment.py — Unit tests for optimization, quantization, inference engine, and Flask server endpoints.
"""

import pytest
import json
import torch
from src.model.model_config import NanoAmharicConfig
from src.model.transformer import NanoAmharicLM
from src.model.quantize import quantize_model_dynamic
from src.app.inference_engine import NanoAmharicInferenceEngine
from src.app.server import app, get_engine


def test_quantize_dynamic_cpu_forward():
    """Verify that INT8 dynamic quantization executes forward pass cleanly on CPU."""
    cfg = NanoAmharicConfig.prototype(vocab_size=100)
    cfg.embedding_dim = 64
    cfg.num_layers = 2
    cfg.num_heads = 4
    cfg.num_kv_heads = 4
    cfg.head_dim = 16
    cfg.ffn_dim = 128
    cfg.max_seq_len = 32

    model = NanoAmharicLM(cfg)
    model.eval()

    q_model = quantize_model_dynamic(model)

    inp = torch.randint(0, 100, (1, 8), dtype=torch.long)
    with torch.no_grad():
        logits, _ = q_model(inp)

    assert logits.shape == (1, 8, 100)
    assert not torch.isnan(logits).any()


def test_inference_engine_modes():
    """Verify inference engine generation, storytelling, chat, and streaming."""
    engine = NanoAmharicInferenceEngine(use_int8=False)

    # 1. Synchronous generation
    gen_res = engine.generate("ሰላም", max_new_tokens=5)
    assert "prompt" in gen_res
    assert "continuation" in gen_res
    assert gen_res["tokens_generated"] > 0
    assert gen_res["tokens_per_second"] > 0.0

    # 2. Storytelling
    story_res = engine.generate_story(topic="አንድ ንጉሥ", max_tokens=10)
    assert "full_text" in story_res
    assert "ንጉሥ" in story_res["full_text"]

    # 3. Chat
    chat_res = engine.chat(user_message="እንደምን አለህ", max_tokens=10)
    assert "assistant_response" in chat_res
    assert len(chat_res["assistant_response"]) >= 0

    # 4. Streaming
    stream_chunks = list(engine.generate_stream("ሰላም", max_new_tokens=5))
    assert len(stream_chunks) > 0
    assert all(isinstance(c, str) for c in stream_chunks)


def test_flask_api_endpoints():
    """Verify Flask REST API endpoints and web UI."""
    client = app.test_client()

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = json.loads(res_health.data)
    assert data_health["status"] == "healthy"
    assert data_health["model_name"] == "NanoAmharic-MorphBPE"

    # 2. Generate API
    res_gen = client.post("/api/generate", json={"prompt": "ኢትዮጵያ", "max_tokens": 5})
    assert res_gen.status_code == 200
    data_gen = json.loads(res_gen.data)
    assert "continuation" in data_gen

    # 3. Story API
    res_story = client.post("/api/story", json={"topic": "አበባ", "max_tokens": 5})
    assert res_story.status_code == 200
    data_story = json.loads(res_story.data)
    assert "full_text" in data_story

    # 4. Chat API
    res_chat = client.post("/api/chat", json={"message": "ሰላም", "max_tokens": 5})
    assert res_chat.status_code == 200
    data_chat = json.loads(res_chat.data)
    assert "assistant_response" in data_chat

    # 5. Web UI
    res_ui = client.get("/")
    assert res_ui.status_code == 200
    assert "ናኖ-አማርኛ" in res_ui.data.decode("utf-8")

    # 6. Tokenize API
    res_tok = client.post("/api/tokenize", json={"text": "በመንግሥታችን አልተረዱልንም"})
    assert res_tok.status_code == 200
    data_tok = json.loads(res_tok.data)
    assert "morphbpe_tokens" in data_tok
    assert "standard_bpe_tokens" in data_tok
    assert "words_analysis" in data_tok
    assert len(data_tok["words_analysis"]) == 2

    # 7. Model Stats API
    res_stats = client.get("/api/stats")
    assert res_stats.status_code == 200
    data_stats = json.loads(res_stats.data)
    assert data_stats["model_name"] == "NanoAmharic-MorphBPE"
    assert data_stats["parameters"] == 5745920


