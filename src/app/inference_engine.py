"""
inference_engine.py — High-performance CPU inference engine for NanoAmharic.
Supports FP32 and INT8 quantized checkpoints, streaming token generation, storytelling, and conversational dialogue.
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator, Tuple
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.model_config import NanoAmharicConfig
from src.model.transformer import NanoAmharicLM
from src.tokenizer.morphbpe import MorphBPETokenizer
from src.training.checkpoint import load_checkpoint


class NanoAmharicInferenceEngine:
    """
    Inference Engine for NanoAmharic models on CPU.
    Provides synchronous and streaming generation for storytelling and chat.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        tokenizer_dir: Optional[str] = None,
        use_int8: bool = False,
    ):
        if tokenizer_dir is None:
            models_tok = PROJECT_ROOT / "models/tokenizer/MorphBPE_v4k_l050"
            eval_tok = PROJECT_ROOT / "eval/results/tokenizer/MorphBPE_v4k_l050"
            tokenizer_dir = str(models_tok if models_tok.exists() else eval_tok)
        if checkpoint_path is None:
            if use_int8:
                models_cp = PROJECT_ROOT / "models/checkpoints/nanoamharic_morphbpe_v4k_int8.pt"
                eval_cp = PROJECT_ROOT / "eval/results/checkpoints/nanoamharic_morphbpe_v4k_int8.pt"
                checkpoint_path = str(models_cp if models_cp.exists() else eval_cp)
            else:
                models_cp = PROJECT_ROOT / "models/checkpoints/experimental_morphbpe_v4k_best.pt"
                eval_cp = PROJECT_ROOT / "eval/results/checkpoints/experimental_morphbpe_v4k_best.pt"
                checkpoint_path = str(models_cp if models_cp.exists() else eval_cp)

        self.tokenizer_path = Path(tokenizer_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.use_int8 = use_int8

        # 1. Load Tokenizer
        print(f"[InferenceEngine] Loading tokenizer from {self.tokenizer_path}...")
        self.tokenizer = MorphBPETokenizer.load(self.tokenizer_path)
        self.vocab_size = self.tokenizer.vocab_size

        # 2. Instantiate Base Model Architecture
        print(f"[InferenceEngine] Initializing NanoAmharicLM (vocab={self.vocab_size})...")
        self.cfg = NanoAmharicConfig.prototype(vocab_size=self.vocab_size)
        self.cfg.max_seq_len = 128
        self.model = NanoAmharicLM(self.cfg)

        # 3. Load Checkpoint
        print(f"[InferenceEngine] Loading weights from {self.checkpoint_path}...")
        if self.use_int8 or "int8" in str(self.checkpoint_path).lower():
            # Dynamically quantize linear layers first, then load state_dict
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            state_dict = torch.load(str(self.checkpoint_path), map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.is_quantized = True
        else:
            load_checkpoint(str(self.checkpoint_path), self.model)
            self.is_quantized = False

        self.model.eval()
        param_info = self.model.count_parameters() if hasattr(self.model, "count_parameters") else {"trainable_params": 5745920}
        self.param_count = param_info.get("trainable_params", 5745920)
        print(f"[InferenceEngine] Ready. Param count: {self.param_count/1e6:.2f}M | INT8: {self.is_quantized}")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> Dict[str, Any]:
        """
        Synchronous text generation given a prompt.
        """
        t0 = time.perf_counter()
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        prompt_len = len(input_ids)

        out_ids = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        elapsed = time.perf_counter() - t0
        all_tokens = out_ids[0].tolist()
        continuation_tokens = all_tokens[prompt_len:]
        continuation_text = self.tokenizer.decode(continuation_tokens)
        full_text = self.tokenizer.decode(all_tokens)

        tok_count = len(continuation_tokens)
        throughput = (tok_count / elapsed) if elapsed > 0 else 0.0

        return {
            "prompt": prompt,
            "continuation": continuation_text.strip(),
            "full_text": full_text.strip(),
            "tokens_generated": tok_count,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(throughput, 1),
            "is_quantized": self.is_quantized,
        }

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> Iterator[str]:
        """
        Streaming token generator yielding decoded text fragments one by one.
        """
        input_ids = self.tokenizer.encode(prompt)
        curr_ids = torch.tensor([input_ids], dtype=torch.long)
        eos_id = self.cfg.eos_token_id

        for _ in range(max_new_tokens):
            cond_ids = curr_ids if curr_ids.size(1) <= self.cfg.max_seq_len else curr_ids[:, -self.cfg.max_seq_len :]
            logits, _ = self.model(cond_ids)
            next_logits = logits[:, -1, :].clone()

            if repetition_penalty != 1.0:
                for prev in set(curr_ids[0].tolist()):
                    if next_logits[0, prev] < 0:
                        next_logits[0, prev] *= repetition_penalty
                    else:
                        next_logits[0, prev] /= repetition_penalty

            if temperature > 0.0:
                next_logits = next_logits / temperature
                if top_k > 0:
                    top_k_val = min(top_k, next_logits.size(-1))
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k_val)[0][..., -1, None]
                    next_logits[indices_to_remove] = float("-inf")

                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            if token_id == eos_id:
                break

            yield self.tokenizer.decode([token_id])

    def generate_story(
        self,
        topic: str = "አንድ ንጉሥ",
        genre: str = "ተረት",
        max_tokens: int = 70,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.92,
        repetition_penalty: float = 1.15,
    ) -> Dict[str, Any]:
        """
        Generates a traditional Amharic narrative or story.
        """
        prompt = f"በጥንት ዘመን {topic}"
        res = self.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        res["genre"] = genre
        res["topic"] = topic
        return res

    def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 40,
        temperature: float = 0.6,
        top_k: int = 30,
        top_p: float = 0.88,
        repetition_penalty: float = 1.12,
    ) -> Dict[str, Any]:
        """
        Conversational turn generation formatted with Amharic dialogue structure.
        """
        history_text = ""
        if conversation_history:
            for turn in conversation_history[-3:]:  # Keep last 3 turns
                history_text += f"ተጠያቂ፡ {turn.get('user', '')}\nመልስ፡ {turn.get('assistant', '')}\n"

        prompt = f"{history_text}ተጠያቂ፡ {user_message}\nመልስ፡"
        res = self.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        # Extract assistant response up to newline or next turn
        raw_ans = res["continuation"].split("ተጠያቂ")[0].strip()
        return {
            "user_message": user_message,
            "assistant_response": raw_ans,
            "full_prompt": prompt,
            "tokens_generated": res["tokens_generated"],
            "elapsed_seconds": res["elapsed_seconds"],
            "tokens_per_second": res["tokens_per_second"],
        }

