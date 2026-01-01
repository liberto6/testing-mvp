"""
OpenAI Whisper STT Processor (Alternative to faster-whisper)
Use this if faster-whisper has cuDNN issues
More compatible but slightly slower than faster-whisper
"""

import asyncio
import time
import numpy as np
import torch
from typing import AsyncGenerator, Optional
from dataclasses import dataclass
import logging

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


@dataclass
class OpenAIWhisperConfig:
    """Configuration for OpenAI Whisper"""
    model_size: str = "base"
    device: str = "cuda"
    language: str = "en"
    vad_enabled: bool = True
    vad_threshold: float = 0.5


class OpenAIWhisperProcessor(FrameProcessor):
    """
    OpenAI Whisper STT Processor

    Alternative to faster-whisper that's more compatible
    but slightly slower. Use if you have cuDNN issues.
    """

    def __init__(
        self,
        config: OpenAIWhisperConfig,
        sample_rate: int = 16000,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.config = config
        self.sample_rate = sample_rate
        self.model = None
        self.audio_buffer = []
        self.is_processing = False

        # Performance metrics
        self.total_audio_duration = 0.0
        self.total_inference_time = 0.0
        self.transcription_count = 0

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize OpenAI Whisper model"""
        try:
            import whisper

            logger.info(f"🔄 Loading OpenAI Whisper {self.config.model_size} on {self.config.device}...")

            start_time = time.time()

            # Load model
            self.model = whisper.load_model(
                self.config.model_size,
                device=self.config.device
            )

            load_time = time.time() - start_time

            logger.info(f"✅ OpenAI Whisper loaded in {load_time:.2f}s")

            # Log VRAM if GPU
            if self.config.device == "cuda" and torch.cuda.is_available():
                vram_allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"📊 VRAM: {vram_allocated:.2f}GB allocated")

        except ImportError:
            logger.error("❌ openai-whisper not installed")
            logger.error("   Install with: pip install openai-whisper")
            raise

        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            raise

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Process incoming audio frames"""

        if not isinstance(frame, AudioRawFrame):
            yield frame
            return

        # Buffer audio
        audio_data = np.frombuffer(frame.audio, dtype=np.int16)
        self.audio_buffer.append(audio_data)

        # Check buffer duration
        buffer_duration = len(self.audio_buffer) * len(audio_data) / self.sample_rate

        # Process when we have enough audio
        if buffer_duration >= 1.0:
            transcription = await self._transcribe_buffer()

            if transcription:
                yield TranscriptionFrame(
                    text=transcription,
                    user_id="user",
                    timestamp=time.time()
                )

            # Clear buffer
            self.audio_buffer = []

        yield frame

    async def _transcribe_buffer(self) -> Optional[str]:
        """Transcribe buffered audio"""
        if not self.audio_buffer or self.is_processing:
            return None

        self.is_processing = True

        try:
            # Concatenate buffer
            audio_np = np.concatenate(self.audio_buffer)

            # Convert to float32 [-1, 1]
            audio_float = audio_np.astype(np.float32) / 32768.0

            # VAD check
            if self.config.vad_enabled:
                rms = np.sqrt(np.mean(audio_float ** 2))
                if rms < self.config.vad_threshold * 0.01:
                    logger.debug("🔇 Silence detected")
                    return None

            # Audio duration
            audio_duration = len(audio_float) / self.sample_rate
            self.total_audio_duration += audio_duration

            # Transcribe in executor
            start_time = time.time()
            transcription = await asyncio.get_event_loop().run_in_executor(
                None,
                self._execute_transcription,
                audio_float
            )
            inference_time = time.time() - start_time

            # Update metrics
            self.total_inference_time += inference_time
            self.transcription_count += 1

            if transcription:
                rtf = inference_time / audio_duration
                logger.info(
                    f"🎤 Transcribed: '{transcription[:50]}...' "
                    f"({inference_time:.2f}s, RTF: {rtf:.2f}x)"
                )

            return transcription

        except torch.cuda.OutOfMemoryError:
            logger.error("❌ CUDA OOM!")
            torch.cuda.empty_cache()
            return None

        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return None

        finally:
            self.is_processing = False

    def _execute_transcription(self, audio: np.ndarray) -> str:
        """Execute Whisper transcription (blocking)"""
        import whisper

        # Transcribe
        result = self.model.transcribe(
            audio,
            language=self.config.language,
            fp16=(self.config.device == "cuda")
        )

        return result["text"].strip()

    def get_metrics(self) -> dict:
        """Get performance metrics"""
        avg_rtf = (
            self.total_inference_time / self.total_audio_duration
            if self.total_audio_duration > 0
            else 0
        )

        return {
            "total_audio_duration": self.total_audio_duration,
            "total_inference_time": self.total_inference_time,
            "transcription_count": self.transcription_count,
            "average_rtf": avg_rtf,
        }

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up OpenAI Whisper processor...")

        metrics = self.get_metrics()
        logger.info(f"📊 Final Metrics: {metrics}")

        if self.config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        await super().cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n🧪 Testing OpenAI Whisper Processor...\n")

    config = OpenAIWhisperConfig(
        model_size="tiny",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    processor = OpenAIWhisperProcessor(config=config)

    print(f"\n✅ Processor initialized")
    print(f"Device: {config.device}")
    print(f"Model: {config.model_size}")
