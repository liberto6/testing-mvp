#!/bin/bash
# Fix cuDNN issues for faster-whisper on RunPod

echo "🔧 Fixing cuDNN for faster-whisper..."

# Check current cuDNN
echo "📋 Current library status:"
ldconfig -p | grep cudnn || echo "   cuDNN libraries not found in ldconfig"

# Option 1: Install cuDNN via conda-forge (recommended for RunPod)
echo ""
echo "🔄 Installing cuDNN via pip..."
pip install nvidia-cudnn-cu12

# Option 2: Set LD_LIBRARY_PATH to find cuDNN
echo ""
echo "🔍 Searching for cuDNN libraries..."
find /usr -name "libcudnn*.so*" 2>/dev/null | head -5

# Try to find cuDNN in common locations
CUDNN_PATHS=(
    "/usr/local/cuda/lib64"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/local/lib"
    "$CONDA_PREFIX/lib"
)

for path in "${CUDNN_PATHS[@]}"; do
    if [ -d "$path" ] && ls "$path"/libcudnn*.so* >/dev/null 2>&1; then
        echo "✅ Found cuDNN in: $path"
        export LD_LIBRARY_PATH="$path:$LD_LIBRARY_PATH"
    fi
done

echo ""
echo "📝 Add this to your ~/.bashrc or .env:"
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"

echo ""
echo "✅ cuDNN fix attempt complete"
echo ""
echo "To test, run:"
echo "  python -c 'from faster_whisper import WhisperModel; print(\"OK\")'"
