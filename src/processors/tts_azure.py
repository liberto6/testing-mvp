"""
Azure Cognitive Services TTS Processor for Pipecat
Enterprise-grade TTS with Azure Neural Voices
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
class AzureTTSConfig:
    """Configuration for Azure TTS"""
    api_key: str
    region: str  # e.g., "eastus"
    voice: str = "en-US-JennyNeural"
    language: str = "en-US"
    rate: str = "0%"  # -50% to +50%
    pitch: str = "0%"  # -50% to +50%


class AzureTTSProcessor(FrameProcessor):
    """
    Azure Cognitive Services TTS Processor

    Features:
    - Enterprise-grade neural voices
    - Low latency
    - High quality
    - Multiple language support
    """

    def __init__(
        self,
        config: AzureTTSConfig,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.config = config
        self.speech_synthesizer = None

        # Metrics
        self.synthesis_count = 0
        self.total_chars = 0
        self.total_synthesis_time = 0.0

        # Initialize Azure Speech SDK
        self._initialize_synthesizer()

    def _initialize_synthesizer(self):
        """Initialize Azure Speech synthesizer"""

        try:
            import azure.cognitiveservices.speech as speechsdk

            logger.info(f"🔄 Initializing Azure TTS ({self.config.region})...")

            # Configure speech
            speech_config = speechsdk.SpeechConfig(
                subscription=self.config.api_key,
                region=self.config.region
            )

            # Set voice
            speech_config.speech_synthesis_voice_name = self.config.voice

            # Set output format to raw PCM
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
            )

            # Create synthesizer (in-memory output)
            self.speech_synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=None  # None = in-memory
            )

            logger.info("✅ Azure TTS initialized")

        except ImportError:
            logger.error("❌ Azure Speech SDK not installed")
            logger.error("   Install with: pip install azure-cognitiveservices-speech")
            raise

        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure TTS: {e}")
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
            yield frame

    async def _synthesize_speech(self, text: str) -> Optional[AudioRawFrame]:
        """Synthesize speech from text"""

        start_time = time.time()

        try:
            # Build SSML for better control
            ssml = self._build_ssml(text)

            # Run synthesis in executor (SDK is blocking)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.speech_synthesizer.speak_ssml,
                ssml
            )

            # Check result
            import azure.cognitiveservices.speech as speechsdk

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_data = result.audio_data

                synthesis_time = time.time() - start_time

                # Update metrics
                self.synthesis_count += 1
                self.total_chars += len(text)
                self.total_synthesis_time += synthesis_time

                chars_per_second = len(text) / synthesis_time if synthesis_time > 0 else 0

                logger.info(
                    f"🔊 Azure TTS: '{text[:30]}...' "
                    f"({synthesis_time:.2f}s, {chars_per_second:.0f} chars/s)"
                )

                # Create audio frame
                # Azure outputs at 16kHz with our config
                audio_frame = AudioRawFrame(
                    audio=audio_data,
                    sample_rate=16000,
                    num_channels=1
                )

                return audio_frame

            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                logger.error(f"❌ Azure TTS canceled: {cancellation.reason}")
                if cancellation.error_details:
                    logger.error(f"   Error: {cancellation.error_details}")

                return None

        except Exception as e:
            logger.error(f"❌ Azure TTS error: {e}")
            return None

    def _build_ssml(self, text: str) -> str:
        """Build SSML for synthesis with prosody control"""

        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self.config.language}">
            <voice name="{self.config.voice}">
                <prosody rate="{self.config.rate}" pitch="{self.config.pitch}">
                    {text}
                </prosody>
            </voice>
        </speak>
        """

        return ssml.strip()

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
        logger.info("🧹 Cleaning up Azure TTS processor...")

        metrics = self.get_metrics()
        logger.info(f"📊 Final Azure TTS Metrics: {metrics}")

        await super().cleanup()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    async def test_azure_tts():
        """Test Azure TTS processor"""
        print("\n🧪 Testing Azure TTS Processor...\n")

        api_key = os.getenv("AZURE_SPEECH_KEY")
        region = os.getenv("AZURE_SPEECH_REGION", "eastus")

        if not api_key:
            print("❌ AZURE_SPEECH_KEY not found in environment")
            print("   Set it in .env file to test Azure TTS")
            return

        config = AzureTTSConfig(
            api_key=api_key,
            region=region,
            voice="en-US-JennyNeural"
        )

        processor = AzureTTSProcessor(config=config)

        # Test synthesis
        test_text = "Hello! This is a test of Azure Cognitive Services text to speech."
        print(f"🎯 Synthesizing: '{test_text}'\n")

        text_frame = TextFrame(text=test_text)

        async for output_frame in processor.process_frame(text_frame):
            if isinstance(output_frame, AudioRawFrame):
                print(f"✅ Generated audio: {len(output_frame.audio)} bytes")
                print(f"   Sample rate: {output_frame.sample_rate} Hz")

        print("\n📊 Metrics:", processor.get_metrics())

    # Run test
    asyncio.run(test_azure_tts())
