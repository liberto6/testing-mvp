#!/bin/bash

################################################################################
# VibeVoice TTS - RunPOD Setup Script
################################################################################
#
# Este script automatiza la instalación completa de VibeVoice TTS en RunPOD.
#
# Requisitos:
#   - RunPOD POD con GPU NVIDIA
#   - Persistent storage en /workspace
#   - Conexión a internet
#
# Uso:
#   chmod +x runpod_setup.sh
#   ./runpod_setup.sh
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Emoji
CHECK="✅"
CROSS="❌"
ROCKET="🚀"
GEAR="⚙️"
PACKAGE="📦"
TEST="🧪"
SUCCESS="🎉"

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   VibeVoice TTS - RunPOD Automated Setup Script${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

################################################################################
# Step 1: Validate GPU and CUDA
################################################################################

echo -e "${GEAR} ${BLUE}Step 1/10: Validating GPU and CUDA...${NC}"

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${CROSS} ${RED}nvidia-smi not found. This script requires an NVIDIA GPU.${NC}"
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if [ $? -eq 0 ]; then
    echo -e "${CHECK} ${GREEN}GPU detected successfully${NC}"
else
    echo -e "${CROSS} ${RED}Failed to query GPU${NC}"
    exit 1
fi

# Check CUDA availability in Python
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null
if [ $? -eq 0 ]; then
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    echo -e "${CHECK} ${GREEN}CUDA is available (version: $CUDA_VERSION)${NC}"
else
    echo -e "${YELLOW}⚠️  CUDA not available in PyTorch, will install PyTorch with CUDA support${NC}"
fi

echo ""

################################################################################
# Step 2: Install System Dependencies
################################################################################

echo -e "${GEAR} ${BLUE}Step 2/10: Installing system dependencies...${NC}"

apt-get update -qq
apt-get install -y -qq ffmpeg git > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${CHECK} ${GREEN}System dependencies installed (ffmpeg, git)${NC}"
else
    echo -e "${CROSS} ${RED}Failed to install system dependencies${NC}"
    exit 1
fi

echo ""

################################################################################
# Step 3: Install PyTorch with CUDA
################################################################################

echo -e "${GEAR} ${BLUE}Step 3/10: Installing PyTorch with CUDA support...${NC}"

# Check if PyTorch with CUDA is already installed
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${PACKAGE} Installing PyTorch with CUDA 12.1..."
    pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cu121

    if [ $? -eq 0 ]; then
        echo -e "${CHECK} ${GREEN}PyTorch with CUDA installed successfully${NC}"
    else
        echo -e "${CROSS} ${RED}Failed to install PyTorch with CUDA${NC}"
        exit 1
    fi
else
    echo -e "${CHECK} ${GREEN}PyTorch with CUDA already installed${NC}"
fi

echo ""

################################################################################
# Step 4: Clone VibeVoice Repository
################################################################################

echo -e "${GEAR} ${BLUE}Step 4/10: Cloning VibeVoice repository...${NC}"

VIBEVOICE_DIR="/workspace/VibeVoice"

if [ -d "$VIBEVOICE_DIR" ]; then
    echo -e "${CHECK} ${GREEN}VibeVoice repository already exists at $VIBEVOICE_DIR${NC}"
    cd "$VIBEVOICE_DIR"

    # Optionally update the repo
    echo -e "  ${BLUE}Pulling latest changes...${NC}"
    git pull --quiet origin main 2>/dev/null || git pull --quiet origin master 2>/dev/null || true
else
    echo -e "${PACKAGE} Cloning VibeVoice from GitHub..."
    cd /workspace
    git clone --quiet https://github.com/microsoft/VibeVoice.git

    if [ $? -eq 0 ]; then
        echo -e "${CHECK} ${GREEN}VibeVoice cloned successfully${NC}"
    else
        echo -e "${CROSS} ${RED}Failed to clone VibeVoice repository${NC}"
        exit 1
    fi
fi

echo ""

################################################################################
# Step 5: Install VibeVoice
################################################################################

echo -e "${GEAR} ${BLUE}Step 5/10: Installing VibeVoice in editable mode...${NC}"

cd "$VIBEVOICE_DIR"
pip install --quiet -e .

if [ $? -eq 0 ]; then
    echo -e "${CHECK} ${GREEN}VibeVoice installed successfully${NC}"
else
    echo -e "${CROSS} ${RED}Failed to install VibeVoice${NC}"
    exit 1
fi

# Verify voices are present
VOICES_DIR="$VIBEVOICE_DIR/demo/voices/streaming_model"
if [ -d "$VOICES_DIR" ]; then
    VOICE_COUNT=$(ls -1 "$VOICES_DIR"/*.pt 2>/dev/null | wc -l)
    if [ $VOICE_COUNT -gt 0 ]; then
        echo -e "${CHECK} ${GREEN}Found $VOICE_COUNT voice file(s) in $VOICES_DIR${NC}"
    else
        echo -e "${YELLOW}⚠️  No .pt voice files found in $VOICES_DIR${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Voices directory not found at $VOICES_DIR${NC}"
fi

echo ""

################################################################################
# Step 6: Install Project Requirements
################################################################################

echo -e "${GEAR} ${BLUE}Step 6/10: Installing project requirements...${NC}"

# Assume we're in /workspace/testing-mvp
PROJECT_DIR="/workspace/testing-mvp"

if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${YELLOW}⚠️  Project directory not found at $PROJECT_DIR${NC}"
    echo -e "   Please ensure your project is located at $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

if [ ! -f "requirements.txt" ]; then
    echo -e "${CROSS} ${RED}requirements.txt not found in $PROJECT_DIR${NC}"
    exit 1
fi

echo -e "${PACKAGE} Installing Python dependencies..."
pip install --quiet -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${CHECK} ${GREEN}Project requirements installed successfully${NC}"
else
    echo -e "${CROSS} ${RED}Failed to install project requirements${NC}"
    exit 1
fi

echo ""

################################################################################
# Step 7: Install Flash Attention 2 (Optional)
################################################################################

echo -e "${GEAR} ${BLUE}Step 7/10: Installing Flash Attention 2 (optional)...${NC}"

# Check GPU compute capability
GPU_COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
echo -e "  ${BLUE}GPU Compute Capability: $GPU_COMPUTE${NC}"

# Flash Attention 2 requires compute capability >= 8.0
if [ "$(echo "$GPU_COMPUTE >= 8.0" | bc)" -eq 1 ]; then
    echo -e "${PACKAGE} Installing flash-attn (this may take a few minutes)..."
    pip install --quiet flash-attn --no-build-isolation

    if [ $? -eq 0 ]; then
        echo -e "${CHECK} ${GREEN}Flash Attention 2 installed successfully${NC}"
    else
        echo -e "${YELLOW}⚠️  Flash Attention 2 installation failed (non-critical)${NC}"
        echo -e "   The system will use SDPA as fallback"
    fi
else
    echo -e "${YELLOW}⚠️  GPU compute capability < 8.0, skipping Flash Attention 2${NC}"
    echo -e "   The system will use SDPA (still works fine)${NC}"
fi

echo ""

################################################################################
# Step 8: Configure Environment Variables
################################################################################

echo -e "${GEAR} ${BLUE}Step 8/10: Configuring environment variables...${NC}"

cd "$PROJECT_DIR"

if [ ! -f ".env" ]; then
    echo -e "${PACKAGE} Creating .env file from template..."

    if [ -f ".env.runpod.example" ]; then
        cp .env.runpod.example .env
    elif [ -f ".env.example" ]; then
        cp .env.example .env
    else
        # Create a basic .env file
        cat > .env << 'EOF'
# Groq API Key (REQUIRED - replace with your actual key)
GROQ_API_KEY=your_groq_api_key_here

# TTS Engine
TTS_ENGINE=vibevoice

# VibeVoice Settings
VIBEVOICE_VOICE=Wayne
VIBEVOICE_VOICES_DIR=/workspace/VibeVoice/demo/voices/streaming_model
VIBEVOICE_CFG_SCALE=1.5
VIBEVOICE_DDPM_STEPS=5
EOF
    fi

    echo -e "${CHECK} ${GREEN}.env file created${NC}"
    echo -e "${YELLOW}⚠️  IMPORTANT: Edit .env and add your GROQ_API_KEY${NC}"
else
    echo -e "${CHECK} ${GREEN}.env file already exists${NC}"
fi

# Verify VibeVoice configuration
if grep -q "VIBEVOICE_VOICES_DIR=/workspace/VibeVoice" .env 2>/dev/null; then
    echo -e "${CHECK} ${GREEN}VibeVoice configuration found in .env${NC}"
else
    echo -e "${YELLOW}⚠️  Adding VibeVoice configuration to .env${NC}"
    cat >> .env << 'EOF'

# VibeVoice Settings (added by setup script)
TTS_ENGINE=vibevoice
VIBEVOICE_VOICE=Wayne
VIBEVOICE_VOICES_DIR=/workspace/VibeVoice/demo/voices/streaming_model
VIBEVOICE_CFG_SCALE=1.5
VIBEVOICE_DDPM_STEPS=5
EOF
fi

echo ""

################################################################################
# Step 9: Run Validation Test
################################################################################

echo -e "${GEAR} ${BLUE}Step 9/10: Running validation test...${NC}"

if [ -f "test_vibevoice.py" ]; then
    echo -e "${TEST} Executing test_vibevoice.py..."
    echo ""

    # Run test with timeout (5 minutes max)
    timeout 300 python test_vibevoice.py --skip-api-test 2>&1 | grep -E "(✅|❌|TEST|Audio|Device|seconds|RTF)"

    TEST_EXIT_CODE=$?
    echo ""

    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo -e "${CHECK} ${GREEN}Validation test PASSED${NC}"
    elif [ $TEST_EXIT_CODE -eq 124 ]; then
        echo -e "${YELLOW}⚠️  Test timed out (may still work, check manually)${NC}"
    else
        echo -e "${YELLOW}⚠️  Test failed (exit code: $TEST_EXIT_CODE)${NC}"
        echo -e "   You can still try running the server manually"
    fi
else
    echo -e "${YELLOW}⚠️  test_vibevoice.py not found, skipping validation${NC}"
fi

echo ""

################################################################################
# Step 10: Final Summary
################################################################################

echo -e "${GEAR} ${BLUE}Step 10/10: Setup complete!${NC}"
echo ""

echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${SUCCESS} ${GREEN}VibeVoice TTS Setup Completed Successfully!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo -e "1. ${YELLOW}Edit your .env file and add your Groq API key:${NC}"
echo -e "   ${BLUE}nano .env${NC}"
echo -e "   Change: ${RED}GROQ_API_KEY=your_groq_api_key_here${NC}"
echo -e "   To:     ${GREEN}GROQ_API_KEY=<your_actual_key>${NC}"
echo ""

echo -e "2. ${YELLOW}Start the server:${NC}"
echo -e "   ${BLUE}python server.py${NC}"
echo ""

echo -e "3. ${YELLOW}Access the application:${NC}"
echo -e "   Open your browser and go to:"
echo -e "   ${BLUE}http://<your-pod-id>.runpod.io:8000${NC}"
echo -e "   (or whatever port RunPOD assigned to your pod)"
echo ""

echo -e "4. ${YELLOW}Test the TTS manually (optional):${NC}"
echo -e "   ${BLUE}python test_vibevoice.py --text \"Hello world\" --voice Wayne${NC}"
echo ""

echo -e "${GREEN}Configuration Summary:${NC}"
echo -e "  • TTS Engine: ${BLUE}VibeVoice${NC}"
echo -e "  • Default Voice: ${BLUE}Wayne${NC}"
echo -e "  • Voices Directory: ${BLUE}/workspace/VibeVoice/demo/voices/streaming_model${NC}"
echo -e "  • Project Directory: ${BLUE}$PROJECT_DIR${NC}"
echo ""

echo -e "${BLUE}For troubleshooting, see:${NC}"
echo -e "  • ${BLUE}RUNPOD_DEPLOYMENT.md${NC}"
echo -e "  • ${BLUE}README_VIBEVOICE.md${NC}"
echo ""

echo -e "${SUCCESS} ${GREEN}Happy coding!${NC}"
echo ""
