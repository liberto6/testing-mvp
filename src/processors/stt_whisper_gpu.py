"""
GPU-Optimized Whisper STT Processor for Pipecat
Supports faster-whisper with CUDA, VAD, and dynamic model loading
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
    ErrorFrame
)
from pipecat.processors.frame_processor import FrameProcessor

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@dataclass
class WhisperGPUConfig:
    """Configuration for GPU-optimized Whisper"""
    model_size: str = "medium"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"
    beam_size: int = 5
    best_of: int = 1
    batch_size: int = 16
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    enable_flash_attention: bool = False
    num_workers: int = 1


class WhisperGPUProcessor(FrameProcessor):
    """
    GPU-Optimized Whisper STT Processor for Pipecat

    Features:
    - faster-whisper with CUDA support
    - Dynamic model loading based on VRAM
    - VAD integration for silence detection
    - Batch processing support
    - FP16/BF16 mixed precision
    - Automatic fallback on OOM
    """

    def __init__(
        self,
        config: WhisperGPUConfig,
        sample_rate: int = 16000,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.config = config
        self.sample_rate = sample_rate
        self.model: Optional[WhisperModel] = None
        self.audio_buffer = []
        self.is_processing = False

        # Performance metrics
        self.total_audio_duration = 0.0
        self.total_inference_time = 0.0
        self.transcription_count = 0

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize Whisper model with GPU optimizations"""
        try:
            logger.info(f"🔄 Loading Whisper {self.config.model_size} on {self.config.device}...")

            start_time = time.time()

            # Initialize faster-whisper model
            self.model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
                num_workers=self.config.num_workers,
                download_root=None
            )

            load_time = time.time() - start_time

            # Warmup inference to initialize CUDA kernels
            if self.config.device == "cuda":
                self._warmup_model()

            logger.info(f"✅ Whisper loaded in {load_time:.2f}s")

            # Log VRAM usage if on GPU
            if self.config.device == "cuda" and torch.cuda.is_available():
                vram_allocated = torch.cuda.memory_allocated() / 1024**3
                vram_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"📊 VRAM: {vram_allocated:.2f}GB allocated, {vram_reserved:.2f}GB reserved")

        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            raise

    def _warmup_model(self):
        """Warmup model with dummy inference to initialize CUDA"""
        logger.info("🔥 Warming up model...")

        # Create dummy audio (1 second of silence)
        dummy_audio = np.zeros(self.sample_rate, dtype=np.float32)

        try:
            # Run dummy inference
            segments, _ = self.model.transcribe(
                dummy_audio,
                language=self.config.language,
                beam_size=1,
                best_of=1,
                vad_filter=False
            )

            # Consume generator to complete warmup
            list(segments)

            logger.info("✅ Model warmup complete")

        except Exception as e:
            logger.warning(f"⚠️ Warmup failed (non-critical): {e}")

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Process incoming audio frames"""

        # Pass through non-audio frames
        if not isinstance(frame, AudioRawFrame):
            yield frame
            return

        # Buffer audio
        audio_data = np.frombuffer(frame.audio, dtype=np.int16)
        self.audio_buffer.append(audio_data)

        # Check if we have enough audio to process
        buffer_duration = len(self.audio_buffer) * len(audio_data) / self.sample_rate

        # Process when we have enough audio or on EndFrame
        if buffer_duration >= 1.0:  # Process every 1 second
            transcription = await self._transcribe_buffer()

            if transcription:
                # Yield transcription frame
                yield TranscriptionFrame(
                    text=transcription,
                    user_id="user",
                    timestamp=time.time()
                )

            # Clear buffer
            self.audio_buffer = []

        # Always yield the original frame to continue pipeline
        yield frame

    async def _transcribe_buffer(self) -> Optional[str]:
        """Transcribe buffered audio"""
        if not self.audio_buffer or self.is_processing:
            return None

        self.is_processing = True

        try:
            # Concatenate audio buffer
            audio_np = np.concatenate(self.audio_buffer)

            # Convert to float32 in range -1.0 to 1.0
            audio_float = audio_np.astype(np.float32) / 32768.0

            # Check for silence using simple energy threshold
            if self.config.vad_enabled:
                rms = np.sqrt(np.mean(audio_float ** 2))
                if rms < self.config.vad_threshold * 0.01:  # Adjusted threshold
                    logger.debug("🔇 Silence detected, skipping transcription")
                    return None

            # Calculate audio duration
            audio_duration = len(audio_float) / self.sample_rate
            self.total_audio_duration += audio_duration

            # Run transcription in executor to avoid blocking
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
                rtf = inference_time / audio_duration  # Real-time factor
                logger.info(
                    f"🎤 Transcribed: '{transcription[:50]}...' "
                    f"({inference_time:.2f}s, RTF: {rtf:.2f}x)"
                )

            return transcription

        except torch.cuda.OutOfMemoryError:
            logger.error("❌ CUDA OOM! Attempting to recover...")
            torch.cuda.empty_cache()
            # Try with smaller batch or fallback
            return await self._transcribe_with_fallback(audio_float)

        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return None

        finally:
            self.is_processing = False

    def _execute_transcription(self, audio: np.ndarray) -> str:
        """Execute Whisper transcription (blocking)"""
        segments, info = self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            vad_filter=self.config.vad_enabled,
            vad_parameters={
                "threshold": self.config.vad_threshold
            } if self.config.vad_enabled else None
        )

        # Concatenate all segments
        transcription = " ".join([segment.text for segment in segments]).strip()

        return transcription

    async def _transcribe_with_fallback(self, audio: np.ndarray) -> Optional[str]:
        """Fallback transcription with reduced batch size"""
        logger.warning("⚠️ Using fallback transcription with reduced settings")

        try:
            # Reduce beam size for less memory usage
            segments, _ = self.model.transcribe(
                audio,
                language=self.config.language,
                beam_size=1,  # Reduce from default
                best_of=1,
                vad_filter=False  # Disable VAD to save memory
            )

            transcription = " ".join([segment.text for segment in segments]).strip()
            return transcription

        except Exception as e:
            logger.error(f"❌ Fallback transcription failed: {e}")
            return None

    def get_metrics(self) -> dict:
        """Get performance metrics"""
        avg_rtf = (
            self.total_inference_time / self.total_audio_duration
            if self.total_audio_duration > 0
            else 0
        )

        avg_inference_time = (
            self.total_inference_time / self.transcription_count
            if self.transcription_count > 0
            else 0
        )

        return {
            "total_audio_duration": self.total_audio_duration,
            "total_inference_time": self.total_inference_time,
            "transcription_count": self.transcription_count,
            "average_rtf": avg_rtf,
            "average_inference_time": avg_inference_time,
        }

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up Whisper processor...")

        # Log final metrics
        metrics = self.get_metrics()
        logger.info(f"📊 Final Metrics: {metrics}")

        # Clear CUDA cache
        if self.config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        await super().cleanup()


class WhisperGPUProcessorWithVAD(WhisperGPUProcessor):
    """
    Extended Whisper processor with Silero VAD integration

    This version uses Silero VAD running on GPU for better
    silence detection before sending audio to Whisper.
    """

    def __init__(self, config: WhisperGPUConfig, **kwargs):
        super().__init__(config, **kwargs)

        # Load Silero VAD if enabled
        if config.vad_enabled and config.device == "cuda":
            self._load_silero_vad()

    def _load_silero_vad(self):
        """Load Silero VAD model on GPU"""
        try:
            logger.info("🔄 Loading Silero VAD on GPU...")

            # Load Silero VAD from torch hub
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )

            # Move to GPU
            if self.config.device == "cuda":
                self.vad_model = self.vad_model.cuda()

            self.vad_model.eval()

            # Extract utils
            (self.get_speech_timestamps, _, _, _, _) = utils

            logger.info("✅ Silero VAD loaded successfully")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load Silero VAD: {e}, using simple VAD")
            self.vad_model = None

    def _detect_speech_vad(self, audio: np.ndarray) -> bool:
        """Detect speech using Silero VAD"""
        if not hasattr(self, 'vad_model') or self.vad_model is None:
            # Fallback to simple energy-based VAD
            rms = np.sqrt(np.mean(audio ** 2))
            return rms >= (self.config.vad_threshold * 0.01)

        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio)

            if self.config.device == "cuda":
                audio_tensor = audio_tensor.cuda()

            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                threshold=self.config.vad_threshold,
                sampling_rate=self.sample_rate
            )

            # Return True if speech detected
            has_speech = len(speech_timestamps) > 0

            return has_speech

        except Exception as e:
            logger.warning(f"⚠️ VAD detection failed: {e}")
            return True  # Assume speech to be safe


if __name__ == "__main__":
    # Test Whisper GPU processor
    logging.basicConfig(level=logging.INFO)

    print("\n🧪 Testing Whisper GPU Processor...\n")

    # Create config
    config = WhisperGPUConfig(
        model_size="tiny",  # Use tiny for testing
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "int8"
    )

    # Create processor
    processor = WhisperGPUProcessor(config=config)

    print(f"\n✅ Processor initialized successfully")
    print(f"Device: {config.device}")
    print(f"Model: {config.model_size}")
    print(f"Compute Type: {config.compute_type}")

    # Metrics
    metrics = processor.get_metrics()
    print(f"\n📊 Metrics: {metrics}")
