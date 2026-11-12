# NanoAmharic-MorphBPE Dockerfile
# CPU-optimized minimal container for Amharic text generation and REST API

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /workspace

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for efficient caching
COPY requirements-cpu.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-cpu.txt

# Copy application source code and configurations
COPY configs/ configs/
COPY src/ src/
COPY app/ app/
COPY eval/results/tokenizer/ eval/results/tokenizer/
COPY eval/results/checkpoints/nanoamharic_morphbpe_v4k_int8.pt eval/results/checkpoints/

# Expose server port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default launch command: starts Flask server with INT8 quantization
CMD ["python", "src/app/server.py", "--host", "0.0.0.0", "--port", "5000", "--int8"]

