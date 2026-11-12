import importlib.util
import sys

pkgs = [
    "language_tool_python",
    "copydetect",
    "datasketch",
    "textstat",
    "spacy",
    "transformers",
    "onnxruntime",
    "torch"
]

print("=== QA Tool Package Inspection ===")
for p in pkgs:
    spec = importlib.util.find_spec(p)
    status = "AVAILABLE" if spec is not None else "NOT INSTALLED"
    print(f"  {p:22s}: {status}")

