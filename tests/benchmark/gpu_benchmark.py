"""
GPU Benchmarking Suite for Pipecat Voice Pipeline
Comprehensive performance testing and optimization validation
"""

import asyncio
import time
import torch
import numpy as np
import logging
from typing import Dict, List
from dataclasses import dataclass, asdict

from src.utils.gpu_utils import gpu_manager
from src.processors.stt_whisper_gpu import WhisperGPUProcessor, WhisperGPUConfig
from src.processors.tts_kokoro import KokoroTTSProcessor, KokoroConfig
from pipecat.frames.frames import AudioRawFrame, TextFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result data"""
    test_name: str
    model: str
    device: str
    compute_type: str
    duration_seconds: float
    throughput: float
    rtf: float  # Real-time factor
    vram_used_gb: float
    peak_vram_gb: float


class GPUBenchmark:
    """GPU performance benchmarking"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    async def benchmark_whisper_models(self):
        """Benchmark different Whisper model sizes"""

        logger.info("\n" + "="*60)
        logger.info("🎤 WHISPER MODEL BENCHMARK")
        logger.info("="*60 + "\n")

        models = ["tiny", "base", "small", "medium"]

        if gpu_manager.capabilities and gpu_manager.capabilities.available_memory_gb > 12:
            models.append("large-v3")

        # Generate test audio (10 seconds of speech-like signal)
        sample_rate = 16000
        duration = 10.0
        test_audio = self._generate_test_audio(sample_rate, duration)

        for model_size in models:
            logger.info(f"\n📊 Testing Whisper {model_size}...")

            try:
                # Create processor
                config = WhisperGPUConfig(
                    model_size=model_size,
                    device="cuda" if gpu_manager.device.type == "cuda" else "cpu",
                    compute_type="float16" if gpu_manager.device.type == "cuda" else "int8",
                    vad_enabled=False  # Disable for benchmark consistency
                )

                processor = WhisperGPUProcessor(config=config)

                # Warm up
                logger.info("🔥 Warming up...")
                await self._run_stt_inference(processor, test_audio[:sample_rate * 2])

                # Clear cache
                if gpu_manager.device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                # Benchmark
                logger.info("⏱️ Running benchmark...")

                start_time = time.time()
                transcription = await self._run_stt_inference(processor, test_audio)
                duration = time.time() - start_time

                # Calculate metrics
                rtf = duration / (len(test_audio) / sample_rate)
                throughput = (len(test_audio) / sample_rate) / duration

                # GPU stats
                vram_used = 0
                peak_vram = 0
                if gpu_manager.device.type == "cuda":
                    vram_used = torch.cuda.memory_allocated() / 1024**3
                    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

                # Store result
                result = BenchmarkResult(
                    test_name=f"Whisper {model_size}",
                    model=model_size,
                    device=config.device,
                    compute_type=config.compute_type,
                    duration_seconds=duration,
                    throughput=throughput,
                    rtf=rtf,
                    vram_used_gb=vram_used,
                    peak_vram_gb=peak_vram
                )

                self.results.append(result)

                logger.info(f"✅ {model_size}: {duration:.2f}s (RTF: {rtf:.2f}x, VRAM: {peak_vram:.2f}GB)")

                # Cleanup
                await processor.cleanup()
                del processor

                if gpu_manager.device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"❌ {model_size} failed: {e}")

        self._print_whisper_comparison()

    async def benchmark_batch_sizes(self):
        """Benchmark different batch sizes for Whisper"""

        logger.info("\n" + "="*60)
        logger.info("📦 BATCH SIZE BENCHMARK")
        logger.info("="*60 + "\n")

        if gpu_manager.device.type != "cuda":
            logger.warning("⚠️ Batch size benchmark requires GPU")
            return

        model_size = "medium"
        batch_sizes = [1, 4, 8, 16, 32]

        sample_rate = 16000
        test_audio = self._generate_test_audio(sample_rate, 5.0)

        for batch_size in batch_sizes:
            logger.info(f"\n📊 Testing batch size {batch_size}...")

            try:
                config = WhisperGPUConfig(
                    model_size=model_size,
                    device="cuda",
                    compute_type="float16",
                    batch_size=batch_size,
                    vad_enabled=False
                )

                processor = WhisperGPUProcessor(config=config)

                # Warmup
                await self._run_stt_inference(processor, test_audio[:sample_rate])

                # Benchmark
                torch.cuda.reset_peak_memory_stats()

                start_time = time.time()
                await self._run_stt_inference(processor, test_audio)
                duration = time.time() - start_time

                peak_vram = torch.cuda.max_memory_allocated() / 1024**3

                logger.info(f"✅ Batch {batch_size}: {duration:.2f}s, Peak VRAM: {peak_vram:.2f}GB")

                await processor.cleanup()
                del processor
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                logger.error(f"❌ Batch size {batch_size}: OOM")
                torch.cuda.empty_cache()
                break

            except Exception as e:
                logger.error(f"❌ Batch size {batch_size}: {e}")

    async def benchmark_tts_providers(self):
        """Benchmark TTS providers"""

        logger.info("\n" + "="*60)
        logger.info("🔊 TTS BENCHMARK")
        logger.info("="*60 + "\n")

        test_texts = [
            "Hello, this is a short test.",
            "The quick brown fox jumps over the lazy dog.",
            "This is a longer sentence to test the text-to-speech system performance with more characters and complexity."
        ]

        # Benchmark Kokoro
        if gpu_manager.device.type == "cuda":
            logger.info("\n📊 Testing Kokoro TTS (GPU)...")

            try:
                config = KokoroConfig(
                    voice="af_sarah",
                    device="cuda"
                )

                processor = KokoroTTSProcessor(config=config)

                for text in test_texts:
                    start_time = time.time()

                    text_frame = TextFrame(text=text)
                    async for frame in processor.process_frame(text_frame):
                        if hasattr(frame, 'audio'):
                            pass  # Audio generated

                    duration = time.time() - start_time
                    chars_per_sec = len(text) / duration

                    logger.info(f"   '{text[:30]}...': {duration:.2f}s ({chars_per_sec:.0f} chars/s)")

                await processor.cleanup()

            except Exception as e:
                logger.error(f"❌ Kokoro failed: {e}")

    def _generate_test_audio(self, sample_rate: int, duration: float) -> np.ndarray:
        """Generate test audio signal"""

        # Generate speech-like signal (multiple frequencies)
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Mix of frequencies typical in speech (200-3000 Hz)
        audio = (
            0.3 * np.sin(2 * np.pi * 200 * t) +
            0.3 * np.sin(2 * np.pi * 500 * t) +
            0.2 * np.sin(2 * np.pi * 1000 * t) +
            0.2 * np.sin(2 * np.pi * 2000 * t)
        )

        # Add some noise
        audio += 0.05 * np.random.randn(len(audio))

        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8

        return audio.astype(np.float32)

    async def _run_stt_inference(self, processor: WhisperGPUProcessor, audio: np.ndarray) -> str:
        """Run STT inference"""

        # Convert to int16 bytes
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Create frame
        audio_frame = AudioRawFrame(
            audio=audio_bytes,
            sample_rate=16000,
            num_channels=1
        )

        # Process
        transcription = ""
        async for frame in processor.process_frame(audio_frame):
            if hasattr(frame, 'text'):
                transcription = frame.text

        return transcription

    def _print_whisper_comparison(self):
        """Print comparison table of Whisper models"""

        logger.info("\n" + "="*80)
        logger.info("📊 WHISPER MODEL COMPARISON")
        logger.info("="*80)

        logger.info(f"\n{'Model':<12} | {'Device':<8} | {'Duration':<10} | {'RTF':<8} | {'VRAM':<10}")
        logger.info("-" * 80)

        for result in self.results:
            if "Whisper" in result.test_name:
                logger.info(
                    f"{result.model:<12} | "
                    f"{result.device:<8} | "
                    f"{result.duration_seconds:<10.2f} | "
                    f"{result.rtf:<8.2f} | "
                    f"{result.peak_vram_gb:<10.2f}"
                )

        logger.info("="*80 + "\n")

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON"""

        import json

        results_dict = [asdict(r) for r in self.results]

        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"💾 Results saved to {filename}")


async def main():
    """Run all benchmarks"""

    logger.info("\n🚀 Starting GPU Benchmark Suite...\n")

    # Print GPU info
    if gpu_manager.device.type == "cuda":
        logger.info("🎮 GPU Information:")
        logger.info(f"   Name: {gpu_manager.capabilities.name}")
        logger.info(f"   Memory: {gpu_manager.capabilities.total_memory_gb:.1f}GB")
        logger.info(f"   Compute: {gpu_manager.capabilities.compute_capability}")
        logger.info("")
    else:
        logger.warning("⚠️ No GPU detected, running CPU benchmarks\n")

    # Run benchmarks
    benchmark = GPUBenchmark()

    await benchmark.benchmark_whisper_models()
    await benchmark.benchmark_batch_sizes()
    await benchmark.benchmark_tts_providers()

    # Save results
    benchmark.save_results()

    logger.info("\n✅ Benchmark complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
