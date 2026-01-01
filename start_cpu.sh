#!/bin/bash
# Start server with CPU STT configuration (for cuDNN issues)

echo "🚀 Starting Pipecat Voice Pipeline with CPU STT..."
echo ""
echo "Configuration: STT on CPU, TTS on GPU"
echo ""

export CONFIG_PATH=configs/runpod_cpu_stt.yaml

python run.py
