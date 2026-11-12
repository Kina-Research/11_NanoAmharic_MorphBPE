# Docker Deployment for NanoAmharic-MorphBPE

This directory contains containerization files for building and serving NanoAmharic on commodity CPU hardware without CUDA or GPU dependencies.

## Quickstart

### 1. Build the Docker Image
From the repository root:
```bash
docker build -t nanoamharic:latest -f docker/Dockerfile .
```

### 2. Run the Container
```bash
docker run -p 5000:5000 --rm -it nanoamharic:latest
```

The server will be available at `http://localhost:5000` with the web UI and REST API.

### 3. Using Docker Compose
```bash
cd docker
docker compose up -d
```
Check health:
```bash
curl http://localhost:5000/health
```

