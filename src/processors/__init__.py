"""Pipecat processors for voice pipeline"""

from .stt_whisper_gpu import WhisperGPUProcessor, WhisperGPUConfig, WhisperGPUProcessorWithVAD
from .llm_groq import GroqLLMProcessor, GroqConfig, GroqLLMService
from .llm_local_gpu import LocalLLMProcessor, LocalLLMConfig
from .tts_kokoro import KokoroTTSProcessor, KokoroConfig
from .tts_edge import EdgeTTSProcessor, EdgeTTSConfig, EdgeTTSSimple
from .tts_azure import AzureTTSProcessor, AzureTTSConfig

__all__ = [
    # STT
    'WhisperGPUProcessor',
    'WhisperGPUConfig',
    'WhisperGPUProcessorWithVAD',
    # LLM
    'GroqLLMProcessor',
    'GroqConfig',
    'GroqLLMService',
    'LocalLLMProcessor',
    'LocalLLMConfig',
    # TTS
    'KokoroTTSProcessor',
    'KokoroConfig',
    'EdgeTTSProcessor',
    'EdgeTTSConfig',
    'EdgeTTSSimple',
    'AzureTTSProcessor',
    'AzureTTSConfig',
]
