"""
cli.py — Interactive Command-Line Interface for NanoAmharic.
Supports conversational dialogue, storytelling generation, and single prompt completion on CPU.
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.inference_engine import NanoAmharicInferenceEngine


def run_chat_loop(engine: NanoAmharicInferenceEngine):
    """Interactive conversational chat in the terminal."""
    print("\n========================================================")
    print(" ናኖ-አማርኛ (NanoAmharic) — በውይይት ሁነታ (Interactive Chat)")
    print(" ለመውጣት 'exit' ወይም 'quit' ብለው ይጻፉ።")
    print("========================================================\n")

    history = []
    while True:
        try:
            user_input = input("\nተጠያቂ> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("ውይይቱ ተጠናቋል። ደህና ሁኑ!")
                break

            print("ረዳት> ", end="", flush=True)
            res = engine.chat(user_input, conversation_history=history, max_tokens=40)
            print(res["assistant_response"])
            print(f"[{res['tokens_generated']} tokens | {res['tokens_per_second']} tok/s]")
            history.append({"user": user_input, "assistant": res["assistant_response"]})

        except (KeyboardInterrupt, EOFError):
            print("\nውይይቱ ተጠናቋል። ደህና ሁኑ!")
            break


def run_story_prompt(engine: NanoAmharicInferenceEngine, topic: str, tokens: int):
    """Generate and display an Amharic story."""
    print(f"\n[ናኖ-አማርኛ ተረት ማፍለቂያ] ርዕስ፡ {topic}")
    print("-" * 50)
    res = engine.generate_story(topic=topic, max_tokens=tokens)
    print(res["full_text"])
    print("-" * 50)
    print(f"⏱ {res['elapsed_seconds']}s | ⚡ {res['tokens_per_second']} tok/s | 📝 {res['tokens_generated']} tokens\n")


def main():
    parser = argparse.ArgumentParser(description="NanoAmharic Interactive CLI")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat session")
    parser.add_argument("--story", type=str, default=None, help="Generate story for the given topic")
    parser.add_argument("--prompt", type=str, default=None, help="Complete a single prompt")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantized checkpoint")
    parser.add_argument("--tokens", type=int, default=50, help="Maximum new tokens to generate")
    args = parser.parse_args()

    engine = NanoAmharicInferenceEngine(use_int8=args.int8)

    if args.chat:
        run_chat_loop(engine)
    elif args.story:
        run_story_prompt(engine, args.story, args.tokens)
    elif args.prompt:
        res = engine.generate(args.prompt, max_new_tokens=args.tokens)
        print(f"\n{res['full_text']}\n")
    else:
        # Default to interactive chat
        run_chat_loop(engine)


if __name__ == "__main__":
    main()

