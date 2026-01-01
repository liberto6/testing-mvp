"""
Kokoro TTS Processor for Pipecat
GPU-accelerated neural TTS with Kokoro
"""

import asyncio
import time
import io
import numpy as np
import scipy.io.wavfile as wavfile
from typing import AsyncGenerator, Optional
from dataclasses import dataclass
import logging

from pipecat.frames.frames import Frame, TextFrame, AudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


@dataclass
class KokoroConfig:
    """Configuration for Kokoro TTS"""
    voice: str = "af_sarah"
    lang_code: str = "a"  # 'a' for American English
    repo_id: str = "hexgrad/Kokoro-82M"
    device: str = "cuda"
    speed: float = 1.0
    sample_rate: int = 24000
    quality: str = "high"


class KokoroTTSProcessor(FrameProcessor):
    """
    Kokoro TTS Processor with GPU acceleration

    Features:
    - GPU-accelerated inference
    - Streaming audio generation
    - High-quality neural TTS
    - Low latency with proper GPU
    """

    def __init__(
        self,
        config: KokoroConfig,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.config = config
        self.pipeline = None

        # Metrics
        self.synthesis_count = 0
        self.total_chars = 0
        self.total_synthesis_time = 0.0

        # Initialize pipeline
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize Kokoro pipeline"""
        try:
            from kokoro import KPipeline
            import torch

            logger.info(f"🔄 Loading Kokoro TTS on {self.config.device}...")

            start_time = time.time()

            # Initialize pipeline
            self.pipeline = KPipeline(
                lang_code=self.config.lang_code,
                repo_id=self.config.repo_id
            )

            # Move to GPU if specified
            if self.config.device == "cuda" and torch.cuda.is_available():
                # Kokoro models are automatically on GPU if PyTorch is CUDA-enabled
                logger.info("✅ Kokoro using CUDA")
            elif self.config.device == "cuda":
                logger.warning("⚠️ CUDA requested but not available, using CPU")

            load_time = time.time() - start_time
            logger.info(f"✅ Kokoro TTS loaded in {load_time:.2f}s")

        except ImportError:
            logger.error("❌ Kokoro not installed. Install with: pip install kokoro>=0.3.4")
            raise

        except Exception as e:
            logger.error(f"❌ Failed to initialize Kokoro: {e}")
            raise

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Process incoming text frames and generate audio"""

        if isinstance(frame, TextFrame):
            text = frame.text

            if text.strip():
                # Generate audio
                audio_frame = await self._synthesize_speech(text)

                if audio_frame:
                    yield audio_frame

        else:
            # Pass through non-text frames
            yield frame

    async def _synthesize_speech(self, text: str) -> Optional[AudioRawFrame]:
        """Synthesize speech from text"""

        start_time = time.time()

        try:
            # Run synthesis in executor to avoid blocking
            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(
                None,
                self._execute_synthesis,
                text
            )

            if audio_bytes:
                synthesis_time = time.time() - start_time

                # Update metrics
                self.synthesis_count += 1
                self.total_chars += len(text)
                self.total_synthesis_time += synthesis_time

                # Calculate speed
                chars_per_second = len(text) / synthesis_time if synthesis_time > 0 else 0

                logger.info(
                    f"🔊 Synthesized: '{text[:30]}...' "
                    f"({synthesis_time:.2f}s, {chars_per_second:.0f} chars/s)"
                )

                # Create audio frame
                # Note: Pipecat expects raw PCM audio
                audio_frame = AudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=self.config.sample_rate,
                    num_channels=1
                )

                return audio_frame

            return None

        except Exception as e:
            logger.error(f"❌ Synthesis error: {e}")
            return None

    def _execute_synthesis(self, text: str) -> Optional[bytes]:
        """Execute Kokoro synthesis (blocking)"""

        try:
            # Generate audio using Kokoro pipeline
            generator = self.pipeline(
                text,
                voice=self.config.voice,
                speed=self.config.speed,
                split_pattern=r'\n+'  # Split on newlines for better quality
            )

            # Collect all audio segments
            audio_segments = []

            for graphemes, phonemes, audio_tensor in generator:
                audio_segments.append(audio_tensor)

            if not audio_segments:
                logger.warning(f"⚠️ No audio generated for text: '{text}'")
                return None

            # Concatenate audio segments
            audio_np = np.concatenate(audio_segments)

            # Convert float32 [-1, 1] to int16
            audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

            # Return raw PCM bytes (not WAV)
            return audio_int16.tobytes()

        except Exception as e:
            logger.error(f"❌ Kokoro execution error: {e}")
            return None

    def get_wav_bytes(self, text: str) -> Optional[bytes]:
        """
        Synchronous method to get WAV bytes
        Useful for non-streaming use cases
        """

        try:
            generator = self.pipeline(
                text,
                voice=self.config.voice,
                speed=self.config.speed,
                split_pattern=r'\n+'
            )

            # Collect audio
            audio_segments = []
            for _, _, audio_tensor in generator:
                audio_segments.append(audio_tensor)

            if not audio_segments:
                return None

            # Concatenate and convert
            audio_np = np.concatenate(audio_segments)
            audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

            # Write to WAV format
            byte_io = io.BytesIO()
            wavfile.write(byte_io, self.config.sample_rate, audio_int16)

            return byte_io.getvalue()

        except Exception as e:
            logger.error(f"❌ WAV generation error: {e}")
            return None

    def get_metrics(self) -> dict:
        """Get performance metrics"""

        avg_synthesis_time = (
            self.total_synthesis_time / self.synthesis_count
            if self.synthesis_count > 0
            else 0
        )

        avg_chars_per_synthesis = (
            self.total_chars / self.synthesis_count
            if self.synthesis_count > 0
            else 0
        )

        chars_per_second = (
            self.total_chars / self.total_synthesis_time
            if self.total_synthesis_time > 0
            else 0
        )

        return {
            "synthesis_count": self.synthesis_count,
            "total_chars": self.total_chars,
            "total_synthesis_time": self.total_synthesis_time,
            "avg_synthesis_time": avg_synthesis_time,
            "avg_chars_per_synthesis": avg_chars_per_synthesis,
            "chars_per_second": chars_per_second,
        }

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up Kokoro processor...")

        # Log final metrics
        metrics = self.get_metrics()
        logger.info(f"📊 Final TTS Metrics: {metrics}")

        await super().cleanup()


if __name__ == "__main__":
    import torch

    logging.basicConfig(level=logging.INFO)

    async def test_kokoro():
        """Test Kokoro processor"""
        print("\n🧪 Testing Kokoro TTS Processor...\n")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        config = KokoroConfig(
            voice="af_sarah",
            device=device,
            speed=1.0
        )

        processor = KokoroTTSProcessor(config=config)

        # Test synthesis
        test_text = "Hello! This is a test of the Kokoro text to speech system."
        print(f"🎯 Synthesizing: '{test_text}'\n")

        text_frame = TextFrame(text=test_text)

        async for output_frame in processor.process_frame(text_frame):
            if isinstance(output_frame, AudioRawFrame):
                print(f"✅ Generated audio: {len(output_frame.audio)} bytes")
                print(f"   Sample rate: {output_frame.sample_rate} Hz")

        print("\n📊 Metrics:", processor.get_metrics())

    # Run test
    asyncio.run(test_kokoro())
