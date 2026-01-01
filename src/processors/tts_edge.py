"""
Microsoft Edge TTS Processor for Pipecat
Free, CPU-efficient TTS using Edge browser voices
"""

import asyncio
import time
from typing import AsyncGenerator, Optional
from dataclasses import dataclass
import logging

from pipecat.frames.frames import Frame, TextFrame, AudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


@dataclass
class EdgeTTSConfig:
    """Configuration for Edge TTS"""
    voice: str = "en-US-JennyNeural"
    rate: str = "+0%"  # Speech rate: -50% to +100%
    volume: str = "+0%"  # Volume: -50% to +50%
    pitch: str = "+0Hz"  # Pitch: -50Hz to +50Hz


class EdgeTTSProcessor(FrameProcessor):
    """
    Microsoft Edge TTS Processor

    Features:
    - Free, no API key required
    - High-quality neural voices
    - CPU-efficient
    - Good fallback for GPU TTS
    """

    def __init__(
        self,
        config: EdgeTTSConfig,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.config = config

        # Metrics
        self.synthesis_count = 0
        self.total_chars = 0
        self.total_synthesis_time = 0.0

        # Verify edge-tts is installed
        self._verify_installation()

    def _verify_installation(self):
        """Verify edge-tts is installed"""
        try:
            import edge_tts
            logger.info("✅ Edge TTS available")
        except ImportError:
            logger.error("❌ edge-tts not installed. Install with: pip install edge-tts")
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
            import edge_tts
            import io

            # Create communicate instance
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.config.voice,
                rate=self.config.rate,
                volume=self.config.volume,
                pitch=self.config.pitch
            )

            # Collect audio chunks
            audio_chunks = []

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])

            if not audio_chunks:
                logger.warning(f"⚠️ No audio generated for text: '{text}'")
                return None

            # Concatenate audio (MP3 format from Edge)
            audio_bytes = b"".join(audio_chunks)

            # Convert MP3 to PCM using pydub
            audio_pcm = await self._convert_mp3_to_pcm(audio_bytes)

            if not audio_pcm:
                return None

            synthesis_time = time.time() - start_time

            # Update metrics
            self.synthesis_count += 1
            self.total_chars += len(text)
            self.total_synthesis_time += synthesis_time

            chars_per_second = len(text) / synthesis_time if synthesis_time > 0 else 0

            logger.info(
                f"🔊 Edge TTS: '{text[:30]}...' "
                f"({synthesis_time:.2f}s, {chars_per_second:.0f} chars/s)"
            )

            # Create audio frame
            # Edge TTS outputs at 24kHz
            audio_frame = AudioRawFrame(
                audio=audio_pcm,
                sample_rate=24000,
                num_channels=1
            )

            return audio_frame

        except Exception as e:
            logger.error(f"❌ Edge TTS error: {e}")
            return None

    async def _convert_mp3_to_pcm(self, mp3_bytes: bytes) -> Optional[bytes]:
        """Convert MP3 audio to raw PCM"""

        try:
            from pydub import AudioSegment
            import io

            # Load MP3 from bytes
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))

            # Convert to mono if needed
            if audio.channels > 1:
                audio = audio.set_channels(1)

            # Convert to 24kHz (Edge TTS native rate)
            if audio.frame_rate != 24000:
                audio = audio.set_frame_rate(24000)

            # Convert to 16-bit PCM
            audio = audio.set_sample_width(2)  # 2 bytes = 16 bits

            # Get raw PCM data
            pcm_bytes = audio.raw_data

            return pcm_bytes

        except ImportError:
            logger.error("❌ pydub not installed. Install with: pip install pydub")
            logger.error("   Also need ffmpeg: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
            return None

        except Exception as e:
            logger.error(f"❌ MP3 conversion error: {e}")
            return None

    def get_metrics(self) -> dict:
        """Get performance metrics"""

        avg_synthesis_time = (
            self.total_synthesis_time / self.synthesis_count
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
            "chars_per_second": chars_per_second,
        }

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up Edge TTS processor...")

        metrics = self.get_metrics()
        logger.info(f"📊 Final Edge TTS Metrics: {metrics}")

        await super().cleanup()


class EdgeTTSSimple:
    """
    Simplified Edge TTS without Pipecat integration
    Useful for standalone usage or migration
    """

    def __init__(self, voice: str = "en-US-JennyNeural"):
        self.voice = voice

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to MP3 bytes"""

        try:
            import edge_tts

            communicate = edge_tts.Communicate(text=text, voice=self.voice)

            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])

            if audio_chunks:
                return b"".join(audio_chunks)

            return None

        except Exception as e:
            logger.error(f"❌ Edge TTS error: {e}")
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def test_edge_tts():
        """Test Edge TTS processor"""
        print("\n🧪 Testing Edge TTS Processor...\n")

        config = EdgeTTSConfig(
            voice="en-US-JennyNeural",
            rate="+0%"
        )

        processor = EdgeTTSProcessor(config=config)

        # Test synthesis
        test_text = "Hello! This is a test of Microsoft Edge text to speech."
        print(f"🎯 Synthesizing: '{test_text}'\n")

        text_frame = TextFrame(text=test_text)

        async for output_frame in processor.process_frame(text_frame):
            if isinstance(output_frame, AudioRawFrame):
                print(f"✅ Generated audio: {len(output_frame.audio)} bytes")
                print(f"   Sample rate: {output_frame.sample_rate} Hz")

        print("\n📊 Metrics:", processor.get_metrics())

    # Run test
    asyncio.run(test_edge_tts())
