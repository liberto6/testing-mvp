# RunPod-Optimized Dockerfile
# Tuned for RunPod GPU instances (RTX 4090, A100)

FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    RUNPOD_ENV=true

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Copy application
COPY src/ ./src/
COPY configs/ ./configs/
COPY static/ ./static/

# Use RunPod-optimized config
ENV CONFIG_PATH=/workspace/configs/runpod_optimized.yaml

EXPOSE 8000

# RunPod start script
COPY deployment/runpod/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
