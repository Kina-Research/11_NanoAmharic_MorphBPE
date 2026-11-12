"""
Phase 1 Setup Verification Script
Verifies the CPU-only environment and essential package imports.
Run: python tests/test_phase1_setup.py
"""
import sys
import os
import pathlib
import importlib
import json

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"


def check_python_version():
    major, minor = sys.version_info.major, sys.version_info.minor
    ok = (major == 3 and minor >= 10)
    status = PASS if ok else FAIL
    print(f"{status} Python {major}.{minor}.{sys.version_info.micro} (required >=3.10)")
    return ok


def check_no_cuda():
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            print(f"{FAIL} CUDA is available — this violates the CPU-only constraint!")
            return False
        print(f"{PASS} CUDA not available (CPU-only confirmed)")
        print(f"       PyTorch {torch.__version__}")
        print(f"       CPU threads: {torch.get_num_threads()}")
        return True
    except ImportError:
        print(f"{FAIL} PyTorch not installed")
        return False


def check_cpu_resources():
    try:
        import psutil
        import multiprocessing
        phys = psutil.cpu_count(logical=False)
        logic = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        total_gb = mem.total / 1e9
        avail_gb = mem.available / 1e9
        print(f"{PASS} CPU: {phys} physical / {logic} logical cores")
        print(f"       RAM total: {total_gb:.1f} GB | available: {avail_gb:.1f} GB")
        if avail_gb < 2.0:
            print(f"{WARN} Available RAM ({avail_gb:.1f} GB) is below 2 GB — close other applications before training")
        return True
    except Exception as e:
        print(f"{WARN} Resource check failed: {e}")
        return False


def check_packages():
    required = [
        ("torch", "2.0.0"),
        ("numpy", "1.24.0"),
        ("transformers", "4.30.0"),
        ("datasets", "2.0.0"),
        ("tokenizers", "0.13.0"),
        ("sentencepiece", "0.1.99"),
        ("requests", "2.28.0"),
        ("tqdm", "4.65.0"),
        ("regex", "2023.0.0"),
        ("psutil", "5.9.0"),
    ]
    optional = [
        "streamlit", "textstat", "spacy", "datasketch",
        "language_tool_python", "sentence_transformers", "onnxruntime", "optimum",
        "jsonlines", "yaml", "pytest"
    ]
    all_ok = True
    print("\n--- Required packages ---")
    for pkg, min_ver in required:
        try:
            m = importlib.import_module(pkg)
            ver = getattr(m, "__version__", "unknown")
            print(f"{PASS}  {pkg} == {ver}")
        except ImportError:
            print(f"{FAIL}  {pkg} NOT FOUND (required)")
            all_ok = False
    print("\n--- Optional packages ---")
    for pkg in optional:
        try:
            m = importlib.import_module(pkg)
            ver = getattr(m, "__version__", "installed")
            print(f"{PASS}  {pkg} == {ver}")
        except ImportError:
            print(f"{WARN}  {pkg} not installed (install before corresponding phase)")
    return all_ok


def check_directory_structure():
    expected = [
        "configs", "data/raw", "data/intermediate", "data/processed",
        "data/reference_corpus", "src/data", "src/tokenizer", "src/model",
        "src/training", "src/evaluation", "src/writing_assistant", "src/app",
        "eval/benchmarks", "eval/baselines", "eval/results", "eval/reports",
        "tests", "paper/figures", "paper/tables", "vendor", "docker",
    ]
    all_ok = True
    for d in expected:
        p = PROJECT_ROOT / d
        if p.exists():
            print(f"{PASS}  {d}/")
        else:
            print(f"{FAIL}  {d}/ MISSING")
            all_ok = False
    return all_ok


def check_config_files():
    expected_files = [
        "configs/system.yaml",
        "configs/tokenizer.yaml",
        "configs/model.yaml",
        "configs/training.yaml",
        "configs/evaluation.yaml",
        "requirements.txt",
        "requirements-cpu.txt",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        ".gitignore",
        "data/DATASET_CARD.md",
        "paper/material_passport.json",
    ]
    all_ok = True
    for f in expected_files:
        p = PROJECT_ROOT / f
        if p.exists():
            print(f"{PASS}  {f}")
        else:
            print(f"{FAIL}  {f} MISSING")
            all_ok = False
    return all_ok


def check_material_passport():
    passport_path = PROJECT_ROOT / "paper" / "material_passport.json"
    try:
        with open(passport_path) as f:
            passport = json.load(f)
        exps = passport.get("experiments", {})
        for exp_id, exp in exps.items():
            status = exp.get("status", "UNKNOWN")
            assert status in ("PLANNED", "RUNNING", "COMPLETED", "FAILED", "VERIFIED"), \
                f"Invalid status: {status}"
            # Confirm no fabricated results for PLANNED experiments
            if status == "PLANNED":
                assert exp.get("metrics") is None, \
                    f"{exp_id} is PLANNED but has non-null metrics — possible fabrication"
        print(f"{PASS}  material_passport.json valid ({len(exps)} experiments declared as PLANNED)")
        return True
    except FileNotFoundError:
        print(f"{FAIL}  material_passport.json not found")
        return False
    except (json.JSONDecodeError, AssertionError) as e:
        print(f"{FAIL}  material_passport.json invalid: {e}")
        return False


def main():
    print("=" * 60)
    print("NanoAmharic-MorphBPE — Phase 1 Setup Verification")
    print("=" * 60)

    results = {}

    print("\n[1] Python Version")
    results["python"] = check_python_version()

    print("\n[2] CPU-Only Constraint (CUDA=False)")
    results["no_cuda"] = check_no_cuda()

    print("\n[3] CPU and Memory Resources")
    results["resources"] = check_cpu_resources()

    print("\n[4] Python Packages")
    results["packages"] = check_packages()

    print("\n[5] Project Directory Structure")
    results["directories"] = check_directory_structure()

    print("\n[6] Configuration and Project Files")
    results["files"] = check_config_files()

    print("\n[7] Material Passport Integrity")
    results["passport"] = check_material_passport()

    print("\n" + "=" * 60)
    passed = sum(results.values())
    total = len(results)
    print(f"Phase 1 Verification: {passed}/{total} checks passed")
    if passed == total:
        print("✅ ALL CHECKS PASSED — Phase 1 complete")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"❌ FAILED: {', '.join(failed)}")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

