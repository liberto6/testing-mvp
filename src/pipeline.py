"""
Main Pipecat Voice Pipeline
Orchestrates STT -> LLM -> TTS with GPU optimizations
"""

import asyncio
from typing import Optional
import logging

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

from src.utils.config import ConfigManager, PipelineConfig
from src.utils.gpu_utils import gpu_manager
from src.processors.stt_whisper_gpu import WhisperGPUProcessor, WhisperGPUConfig
from src.processors.llm_groq import GroqLLMProcessor, GroqConfig
from src.processors.tts_kokoro import KokoroTTSProcessor, KokoroConfig
from src.processors.tts_edge import EdgeTTSProcessor, EdgeTTSConfig

logger = logging.getLogger(__name__)


class VoicePipeline:
    """
    GPU-Optimized Voice Conversation Pipeline

    Architecture:
    Browser/Client -> Transport -> STT (Whisper GPU) -> LLM (Groq) -> TTS (Kokoro GPU) -> Transport -> Browser/Client

    Features:
    - GPU-accelerated STT and TTS
    - Ultra-low latency LLM with Groq
    - Automatic GPU optimization
    - Conversation history management
    - Metrics and monitoring
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        config_manager: Optional[ConfigManager] = None
    ):
        # Load configuration
        if config_manager:
            self.config_manager = config_manager
            self.config = config_manager.get_config()
        elif config:
            self.config = config
        else:
            self.config_manager = ConfigManager(auto_optimize=True)
            self.config = self.config_manager.get_config()

        # Pipeline components
        self.stt_processor: Optional[WhisperGPUProcessor] = None
        self.llm_processor: Optional[GroqLLMProcessor] = None
        self.tts_processor = None  # Will be KokoroTTSProcessor or EdgeTTSProcessor

        self.pipeline: Optional[Pipeline] = None
        self.runner: Optional[PipelineRunner] = None

        # Initialize components
        self._initialize_processors()

    def _initialize_processors(self):
        """Initialize pipeline processors"""

        logger.info("🔧 Initializing pipeline processors...")

        # 1. STT Processor (Whisper GPU)
        stt_config = WhisperGPUConfig(
            model_size=self.config.stt.model,
            device=self.config.stt.device,
            compute_type=self.config.stt.compute_type,
            language=self.config.stt.language,
            beam_size=self.config.stt.beam_size,
            batch_size=self.config.stt.batch_size,
            vad_enabled=self.config.stt.vad_enabled,
            vad_threshold=self.config.stt.vad_threshold,
            enable_flash_attention=self.config.stt.enable_flash_attention
        )

        self.stt_processor = WhisperGPUProcessor(config=stt_config)
        logger.info(f"✅ STT: Whisper {self.config.stt.model} on {self.config.stt.device}")

        # 2. LLM Processor (Groq)
        llm_config = GroqConfig(
            api_key=self.config.llm.api_key,
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens
        )

        self.llm_processor = GroqLLMProcessor(
            config=llm_config,
            system_prompt=self.config.system_prompt
        )
        logger.info(f"✅ LLM: {self.config.llm.model} via Groq")

        # 3. TTS Processor (Kokoro GPU or Edge fallback)
        if self.config.tts.provider == "kokoro":
            tts_config = KokoroConfig(
                voice=self.config.tts.voice,
                device=self.config.tts.device,
                speed=self.config.tts.speed
            )
            self.tts_processor = KokoroTTSProcessor(config=tts_config)
            logger.info(f"✅ TTS: Kokoro on {self.config.tts.device}")

        elif self.config.tts.provider == "edge":
            tts_config = EdgeTTSConfig(
                voice=self.config.tts.fallback_voice
            )
            self.tts_processor = EdgeTTSProcessor(config=tts_config)
            logger.info("✅ TTS: Microsoft Edge")

        else:
            raise ValueError(f"Unsupported TTS provider: {self.config.tts.provider}")

    def create_pipeline(self) -> Pipeline:
        """Create Pipecat pipeline"""

        logger.info("🔨 Building Pipecat pipeline...")

        # Create pipeline with processors in order
        # Note: Simplified version - actual Pipecat pipeline uses more sophisticated routing

        processors = [
            self.stt_processor,
            self.llm_processor,
            self.tts_processor
        ]

        # Create pipeline
        # In real Pipecat, you'd use Pipeline.create() with proper frame routing
        self.pipeline = Pipeline(processors=processors)

        logger.info("✅ Pipeline created")

        return self.pipeline

    async def run_with_transport(self, transport):
        """Run pipeline with a transport"""

        logger.info("🚀 Starting voice pipeline...")

        # Create pipeline if not already created
        if not self.pipeline:
            self.create_pipeline()

        try:
            # Create pipeline task with transport
            task = PipelineTask(
                pipeline=self.pipeline,
                transport=transport
            )

            # Create runner
            self.runner = PipelineRunner()

            # Run pipeline
            await self.runner.run(task)

            logger.info("✅ Pipeline completed")

        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            raise

    def get_metrics(self) -> dict:
        """Get metrics from all processors"""

        metrics = {}

        if self.stt_processor:
            metrics['stt'] = self.stt_processor.get_metrics()

        if self.llm_processor:
            metrics['llm'] = self.llm_processor.get_metrics()

        if self.tts_processor:
            metrics['tts'] = self.tts_processor.get_metrics()

        if gpu_manager.device.type == "cuda":
            metrics['gpu'] = gpu_manager.get_memory_stats()

        return metrics

    async def cleanup(self):
        """Cleanup pipeline resources"""

        logger.info("🧹 Cleaning up pipeline...")

        # Print final metrics
        metrics = self.get_metrics()
        logger.info(f"📊 Final Pipeline Metrics:\n{metrics}")

        # Cleanup processors
        if self.stt_processor:
            await self.stt_processor.cleanup()

        if self.tts_processor:
            await self.tts_processor.cleanup()

        # Clear GPU cache
        gpu_manager.clear_cache()

        logger.info("✅ Pipeline cleanup complete")


class SimplifiedVoicePipeline:
    """
    Simplified pipeline for migration from existing code

    This version mimics the original WebSocket-based architecture
    while using Pipecat processors internally
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or ConfigManager(auto_optimize=True).get_config()

        # Initialize processors
        self.stt_processor = WhisperGPUProcessor(
            config=WhisperGPUConfig(
                model_size=self.config.stt.model,
                device=self.config.stt.device,
                compute_type=self.config.stt.compute_type,
                language=self.config.stt.language
            )
        )

        self.llm_processor = GroqLLMProcessor(
            config=GroqConfig(
                api_key=self.config.llm.api_key,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens
            ),
            system_prompt=self.config.system_prompt
        )

        # TTS processor
        if self.config.tts.provider == "kokoro":
            self.tts_processor = KokoroTTSProcessor(
                config=KokoroConfig(
                    voice=self.config.tts.voice,
                    device=self.config.tts.device
                )
            )
        else:
            self.tts_processor = EdgeTTSProcessor(
                config=EdgeTTSConfig(voice=self.config.tts.fallback_voice)
            )

    async def process_audio(self, audio_bytes: bytes) -> str:
        """Process audio to text (STT)"""

        from pipecat.frames.frames import AudioRawFrame

        # Create audio frame
        audio_frame = AudioRawFrame(
            audio=audio_bytes,
            sample_rate=16000,
            num_channels=1
        )

        # Process through STT
        transcription = ""
        async for frame in self.stt_processor.process_frame(audio_frame):
            if hasattr(frame, 'text'):
                transcription = frame.text

        return transcription

    async def process_text_to_speech(self, text: str) -> bytes:
        """Process text to speech (TTS)"""

        from pipecat.frames.frames import TextFrame

        # Create text frame
        text_frame = TextFrame(text=text)

        # Process through TTS
        audio_data = None
        async for frame in self.tts_processor.process_frame(text_frame):
            if hasattr(frame, 'audio'):
                audio_data = frame.audio

        return audio_data

    async def stream_llm_response(self, user_text: str):
        """Stream LLM response chunks"""

        from pipecat.frames.frames import TextFrame

        # Create text frame
        text_frame = TextFrame(text=user_text)

        # Stream LLM response
        async for frame in self.llm_processor.process_frame(text_frame):
            if hasattr(frame, 'text'):
                yield frame.text


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async def test_pipeline():
        """Test pipeline creation"""

        print("\n🧪 Testing Voice Pipeline...\n")

        # Create pipeline
        pipeline = VoicePipeline()

        print("\n✅ Pipeline initialized successfully")
        print("\n📊 Configuration:")
        pipeline.config_manager.print_config()

        print("\n📈 Initial GPU Stats:")
        if gpu_manager.device.type == "cuda":
            print(gpu_manager.get_memory_stats())
        else:
            print("Running on CPU")

    # Run test
    asyncio.run(test_pipeline())
