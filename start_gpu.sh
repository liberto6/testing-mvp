#!/bin/bash
# Start server with full GPU configuration

echo "🚀 Starting Pipecat Voice Pipeline with Full GPU..."
echo ""
echo "Configuration: STT on GPU, TTS on GPU"
echo ""
echo "⚠️  This requires cuDNN to be working properly"
echo ""

export CONFIG_PATH=configs/runpod_optimized.yaml

python run.py
