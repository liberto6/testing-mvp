# GCP GPU-Optimized Dockerfile
# For Google Cloud GPU VMs (T4, V100, A100)

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    GCP_ENV=true

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements-gpu.txt .
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir -r requirements-gpu.txt

# Copy application
COPY src/ ./src/
COPY configs/ ./configs/
COPY static/ ./static/

# Auto-detect GPU configuration
ENV CONFIG_PATH=/app/configs/auto

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
