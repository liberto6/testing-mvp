#!/bin/bash
# Quick Start Script for Pipecat Voice Pipeline

set -e

echo "🚀 Pipecat Voice Pipeline - Quick Start"
echo "========================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python: $python_version"

# Check CUDA availability
echo ""
echo "🎮 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "   ⚠️  No GPU detected, will use CPU mode"
fi

# Check if .env exists
echo ""
echo "⚙️  Checking configuration..."
if [ ! -f .env ]; then
    echo "   Creating .env from template..."
    cp .env.example .env
    echo "   ⚠️  Please edit .env and add your GROQ_API_KEY"
    echo "   nano .env"
    read -p "   Press Enter after setting GROQ_API_KEY..."
fi

# Check GROQ_API_KEY
source .env
if [ -z "$GROQ_API_KEY" ] || [ "$GROQ_API_KEY" = "your_groq_api_key_here" ]; then
    echo "   ❌ GROQ_API_KEY not set in .env"
    echo "   Please get your API key from: https://console.groq.com"
    exit 1
fi
echo "   ✅ GROQ_API_KEY found"

# Check if requirements are installed
echo ""
echo "📦 Checking dependencies..."
if python3 -c "import pipecat" 2>/dev/null; then
    echo "   ✅ Pipecat installed"
else
    echo "   Installing dependencies..."
    pip3 install -r requirements-gpu.txt

    # Install PyTorch with CUDA
    echo "   Installing PyTorch with CUDA support..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Test GPU detection
echo ""
echo "🧪 Testing GPU detection..."
python3 -c "
from src.utils.gpu_utils import gpu_manager
from src.utils.config import ConfigManager

config_manager = ConfigManager(auto_optimize=True)
config_manager.print_config()
" || echo "   ⚠️  Warning: GPU detection test failed, but will continue..."

# Start server
echo ""
echo "🚀 Starting Pipecat Voice Pipeline..."
echo ""
echo "   Server will start on: http://0.0.0.0:8000"
echo "   Health check: http://localhost:8000/health"
echo "   Config: http://localhost:8000/config"
echo "   WebSocket: ws://localhost:8000/ws"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

# Run server (using uvicorn module to avoid import issues)
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8000
