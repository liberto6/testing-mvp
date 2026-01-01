#!/usr/bin/env python3
"""
Quick setup checker for Pipecat Voice Pipeline
Verifies all dependencies and configurations
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_python_version():
    """Check Python version"""
    print("🐍 Python Version:")
    print(f"   {sys.version}")
    if sys.version_info < (3, 10):
        print("   ⚠️  Warning: Python 3.10+ recommended")
    else:
        print("   ✅ OK")
    print()

def check_cuda():
    """Check CUDA availability"""
    print("🎮 CUDA Check:")
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Device count: {torch.cuda.device_count()}")
            print("   ✅ CUDA OK")
        else:
            print("   ⚠️  No CUDA GPU detected (will use CPU)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()

def check_dependencies():
    """Check required packages"""
    print("📦 Dependencies Check:")

    packages = [
        ("pipecat", "pipecat"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("faster_whisper", "faster-whisper"),
        ("groq", "groq"),
        ("kokoro", "kokoro"),
        ("edge_tts", "edge-tts"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
    ]

    all_ok = True
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - NOT INSTALLED")
            all_ok = False

    if not all_ok:
        print("\n   Install missing packages with:")
        print("   pip install -r requirements-gpu.txt")
    print()

def check_env_vars():
    """Check environment variables"""
    print("⚙️  Environment Variables:")

    import os
    from dotenv import load_dotenv

    # Try to load .env
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"   ✅ .env file found")
    else:
        print(f"   ⚠️  .env file not found")

    # Check GROQ_API_KEY
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and groq_key != "your_groq_api_key_here":
        print(f"   ✅ GROQ_API_KEY is set")
    else:
        print(f"   ❌ GROQ_API_KEY not set or invalid")
        print(f"      Get your key from: https://console.groq.com")
    print()

def check_gpu_utils():
    """Check GPU utilities"""
    print("🔧 GPU Utilities Check:")
    try:
        from src.utils.gpu_utils import gpu_manager

        print(f"   Device: {gpu_manager.device}")

        if gpu_manager.device.type == "cuda":
            print(f"   GPU: {gpu_manager.capabilities.name}")
            print(f"   VRAM: {gpu_manager.capabilities.total_memory_gb:.1f}GB")
            print(f"   Recommended Whisper: {gpu_manager.get_optimal_whisper_model()}")
            print("   ✅ GPU utilities OK")
        else:
            print("   ℹ️  Running in CPU mode")

    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()

def check_config():
    """Check configuration"""
    print("📋 Configuration Check:")
    try:
        from src.utils.config import ConfigManager

        config_manager = ConfigManager(auto_optimize=True)
        config = config_manager.get_config()

        print(f"   STT Model: {config.stt.model}")
        print(f"   STT Device: {config.stt.device}")
        print(f"   LLM Provider: {config.llm.provider}")
        print(f"   TTS Provider: {config.tts.provider}")
        print("   ✅ Configuration OK")

    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()

def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("🔍 PIPECAT VOICE PIPELINE - SETUP CHECK")
    print("="*60 + "\n")

    check_python_version()
    check_cuda()
    check_dependencies()
    check_env_vars()
    check_gpu_utils()
    check_config()

    print("="*60)
    print("✅ Setup check complete!")
    print("\nTo start the server:")
    print("  python run.py")
    print("  OR")
    print("  python -m uvicorn src.main:app --host 0.0.0.0 --port 8000")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
