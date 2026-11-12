"""
app.py — Direct entry point for NanoAmharic interactive studio and Flask application.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.server import app, get_engine

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NanoAmharic Web Application")
    parser.add_argument("--port", type=int, default=5000, help="Port to serve on (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--int8", action="store_true", help="Launch with INT8 dynamic quantization")
    args = parser.parse_args()

    get_engine(use_int8=args.int8)
    print("\n========================================================")
    print(f" NanoAmharic Studio running on http://{args.host}:{args.port}")
    print(f" INT8 Quantization: {args.int8}")
    print("========================================================\n")
    app.run(host=args.host, port=args.port, debug=False)
