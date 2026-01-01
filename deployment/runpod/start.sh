#!/bin/bash
# RunPod startup script

echo "🚀 Starting Pipecat Voice Pipeline on RunPod..."

# GPU detection
nvidia-smi

# Start server
python3 -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
