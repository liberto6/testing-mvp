"""
Configuration Management for Pipecat Voice Pipeline
Supports YAML configs, environment variables, and GPU auto-configuration
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
import logging

from .gpu_utils import get_gpu_config

logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class STTConfig:
    """Speech-to-Text configuration"""
    provider: str = "whisper"
    model: str = "medium"
    language: str = "en"
    device: str = "cuda"
    compute_type: str = "float16"
    batch_size: int = 16
    beam_size: int = 5
    vad_enabled: bool = True
    vad_device: str = "cuda"
    vad_threshold: float = 0.5
    enable_flash_attention: bool = False


@dataclass
class LLMConfig:
    """Large Language Model configuration"""
    provider: str = "groq"
    model: str = "llama-3.1-8b-instant"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 150
    stream: bool = True
    # Fallback for local GPU LLM
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None


@dataclass
class TTSConfig:
    """Text-to-Speech configuration"""
    provider: str = "kokoro"  # Options: kokoro, edge, azure
    device: str = "cuda"
    voice: str = "af_sarah"
    speed: float = 1.0
    quality: str = "high"
    # Fallback TTS
    fallback_provider: str = "edge"
    fallback_voice: str = "en-US-JennyNeural"
    # Azure specific
    azure_api_key: Optional[str] = None
    azure_region: Optional[str] = None


@dataclass
class TransportConfig:
    """Transport layer configuration"""
    type: str = "websocket"  # Options: websocket, daily, local
    # WebSocket config
    host: str = "0.0.0.0"
    port: int = 8000
    # Daily.co config
    daily_api_key: Optional[str] = None
    daily_room_url: Optional[str] = None
    # Audio config
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class GPUConfig:
    """GPU optimization configuration"""
    enabled: bool = True
    device_id: int = 0
    memory_fraction: float = 0.8
    allow_growth: bool = True
    mixed_precision: bool = True
    use_cuda_graphs: bool = False
    optimize_for_latency: bool = True


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)

    # System prompt
    system_prompt: str = field(default_factory=lambda: """
ROLE: You are Sarah, an expert conversational English tutor for Verba.
TARGET: B1/B2 Level Students.
LANGUAGE: ENGLISH ONLY (Unless explaining a complex concept if requested, but prioritize English).

OBJECTIVE:
Improve the student's communicative competence. Maximize their talking time.
Do NOT act as a generic chatbot. Do NOT give long lectures.

LAYERS & RULES:

1. CONVERSATIONAL STYLE:
   - Responses must be BRIEF (Max 2-3 sentences).
   - ALWAYS end with ONE open-ended question to keep the conversation going.
   - Avoid Yes/No questions.
   - Tone: Friendly, professional, encouraging.

2. IMPLICIT CORRECTION (CRITICAL):
   - NEVER say "That is incorrect" or "You made a mistake".
   - RECAST the error naturally in your response.
   - Example: User "Yesterday I go" -> Sarah "Oh, you went yesterday? Nice!"

3. PEDAGOGICAL CONTROL:
   - Adapt vocabulary to the student's level.
   - If the student struggles, simplify.
   - If the student is fluent, challenge them slightly.

4. SILENT EVALUATION:
   - Internally monitor fluency and grammar.
   - DO NOT output scores or feedback during the chat unless explicitly asked.

5. PROHIBITED:
   - Long monologues.
   - Sudden topic changes.
   - Ending a turn without a question.

SESSION CLOSURE:
If the user says "goodbye" or "end session":
1. Summarize strong points briefly.
2. Mention 1-2 specific areas for improvement (e.g., "Watch your past tense").
3. Give a final encouraging remark.
""".strip())


class ConfigManager:
    """Manages pipeline configuration with auto-optimization"""

    def __init__(self, config_path: Optional[str] = None, auto_optimize: bool = True):
        self.config_path = config_path
        self.config = PipelineConfig()
        self.from_yaml = False  # Track if loaded from YAML

        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
            self.from_yaml = True
        else:
            # Load from environment variables
            self.load_from_env()

        # Auto-optimize based on GPU if enabled AND not from YAML
        # If loaded from YAML, respect the explicit config
        if auto_optimize and not self.from_yaml:
            self.auto_optimize_for_gpu()

        logger.info("✅ Configuration loaded successfully")

    def load_from_yaml(self, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if data:
            self._update_config_from_dict(data)
            logger.info(f"📄 Loaded config from: {path}")

    def _update_config_from_dict(self, data: Dict[str, Any]):
        """Update config from dictionary"""
        if 'stt' in data:
            for key, value in data['stt'].items():
                if hasattr(self.config.stt, key):
                    setattr(self.config.stt, key, value)

        if 'llm' in data:
            for key, value in data['llm'].items():
                if hasattr(self.config.llm, key):
                    setattr(self.config.llm, key, value)

        if 'tts' in data:
            for key, value in data['tts'].items():
                if hasattr(self.config.tts, key):
                    setattr(self.config.tts, key, value)

        if 'transport' in data:
            for key, value in data['transport'].items():
                if hasattr(self.config.transport, key):
                    setattr(self.config.transport, key, value)

        if 'gpu' in data:
            for key, value in data['gpu'].items():
                if hasattr(self.config.gpu, key):
                    setattr(self.config.gpu, key, value)

        if 'system_prompt' in data:
            self.config.system_prompt = data['system_prompt']

    def load_from_env(self):
        """Load configuration from environment variables"""
        # API Keys
        self.config.llm.api_key = os.getenv("GROQ_API_KEY")
        self.config.tts.azure_api_key = os.getenv("AZURE_SPEECH_KEY")
        self.config.tts.azure_region = os.getenv("AZURE_SPEECH_REGION")
        self.config.transport.daily_api_key = os.getenv("DAILY_API_KEY")

        # TTS Engine
        tts_engine = os.getenv("TTS_ENGINE", "kokoro").lower()
        self.config.tts.provider = tts_engine

        # Kokoro voice
        self.config.tts.voice = os.getenv("KOKORO_VOICE", "af_sarah")

        # Transport
        self.config.transport.host = os.getenv("HOST", "0.0.0.0")
        self.config.transport.port = int(os.getenv("PORT", "8000"))

        logger.info("📋 Loaded config from environment variables")

    def auto_optimize_for_gpu(self):
        """Auto-optimize configuration based on detected GPU"""
        gpu_config = get_gpu_config()

        # Update STT config
        self.config.stt.model = gpu_config["whisper_model"]
        self.config.stt.compute_type = gpu_config["compute_type"]
        self.config.stt.batch_size = gpu_config["batch_size"]
        self.config.stt.enable_flash_attention = gpu_config["enable_flash_attention"]
        self.config.stt.device = gpu_config["device"]

        # Update TTS device
        if gpu_config["device"] == "cuda":
            self.config.tts.device = "cuda"
        else:
            self.config.tts.device = "cpu"
            # Fallback to Edge TTS on CPU for better performance
            if self.config.tts.provider == "kokoro":
                logger.info("⚠️ GPU not available, switching TTS to Edge for CPU efficiency")
                self.config.tts.provider = "edge"

        logger.info("🎯 Configuration auto-optimized for GPU")

    def save_to_yaml(self, path: str):
        """Save current configuration to YAML file"""
        config_dict = {
            'stt': asdict(self.config.stt),
            'llm': asdict(self.config.llm),
            'tts': asdict(self.config.tts),
            'transport': asdict(self.config.transport),
            'gpu': asdict(self.config.gpu),
            'system_prompt': self.config.system_prompt
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"💾 Saved config to: {path}")

    def get_config(self) -> PipelineConfig:
        """Get the current configuration"""
        return self.config

    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("🔧 PIPECAT PIPELINE CONFIGURATION")
        print("="*60)

        print(f"\n📝 STT (Speech-to-Text):")
        print(f"   Provider: {self.config.stt.provider}")
        print(f"   Model: {self.config.stt.model}")
        print(f"   Device: {self.config.stt.device}")
        print(f"   Compute Type: {self.config.stt.compute_type}")
        print(f"   Batch Size: {self.config.stt.batch_size}")
        print(f"   VAD Enabled: {self.config.stt.vad_enabled}")

        print(f"\n🤖 LLM (Language Model):")
        print(f"   Provider: {self.config.llm.provider}")
        print(f"   Model: {self.config.llm.model}")
        print(f"   Temperature: {self.config.llm.temperature}")
        print(f"   Max Tokens: {self.config.llm.max_tokens}")

        print(f"\n🔊 TTS (Text-to-Speech):")
        print(f"   Provider: {self.config.tts.provider}")
        print(f"   Device: {self.config.tts.device}")
        print(f"   Voice: {self.config.tts.voice}")
        print(f"   Quality: {self.config.tts.quality}")

        print(f"\n🌐 Transport:")
        print(f"   Type: {self.config.transport.type}")
        print(f"   Host: {self.config.transport.host}")
        print(f"   Port: {self.config.transport.port}")

        print(f"\n🎮 GPU:")
        print(f"   Enabled: {self.config.gpu.enabled}")
        print(f"   Mixed Precision: {self.config.gpu.mixed_precision}")
        print(f"   Optimize for Latency: {self.config.gpu.optimize_for_latency}")

        print("\n" + "="*60 + "\n")


# Global config manager
config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None, auto_optimize: bool = True) -> ConfigManager:
    """Get or create global config manager"""
    global config_manager

    if config_manager is None:
        config_manager = ConfigManager(config_path, auto_optimize)

    return config_manager


if __name__ == "__main__":
    # Test configuration
    logging.basicConfig(level=logging.INFO)

    print("\n🧪 Testing Configuration Manager...\n")

    manager = ConfigManager(auto_optimize=True)
    manager.print_config()

    # Save example config
    example_path = "configs/auto_generated.yaml"
    manager.save_to_yaml(example_path)
    print(f"✅ Example config saved to: {example_path}")
