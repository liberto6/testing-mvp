"""
GPU Utilities for Pipecat Voice Pipeline
Auto-detection, optimization, and monitoring for CUDA GPUs
"""

import os
import torch
import subprocess
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GPUCapabilities:
    """GPU hardware capabilities"""
    name: str
    compute_capability: Tuple[int, int]
    total_memory_gb: float
    available_memory_gb: float
    cuda_cores: int
    supports_fp16: bool
    supports_bf16: bool
    supports_int8: bool
    supports_flash_attention: bool
    tensor_cores: bool
    multi_gpu: bool
    gpu_count: int


class GPUManager:
    """Manages GPU resources and provides optimization recommendations"""

    def __init__(self):
        self.capabilities: Optional[GPUCapabilities] = None
        self.device = self._detect_device()

        if self.device.type == "cuda":
            self.capabilities = self._detect_capabilities()
            self._setup_cuda_optimizations()

    def _detect_device(self) -> torch.device:
        """Detect available compute device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")  # faster-whisper needs "cuda" not "cuda:0"
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            return device
        else:
            logger.warning("⚠️ CUDA not available, falling back to CPU")
            return torch.device("cpu")

    def _detect_capabilities(self) -> GPUCapabilities:
        """Detect detailed GPU capabilities"""
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / (1024**3)  # GB

        # Get available memory
        torch.cuda.empty_cache()
        available_mem = torch.cuda.mem_get_info()[0] / (1024**3)  # GB

        # Compute capability determines feature support
        compute_cap = props.major, props.minor

        # Feature detection
        supports_fp16 = compute_cap >= (5, 3)  # Maxwell and newer
        supports_bf16 = compute_cap >= (8, 0)  # Ampere and newer
        supports_int8 = compute_cap >= (6, 1)  # Pascal and newer
        tensor_cores = compute_cap >= (7, 0)   # Volta and newer
        supports_flash_attention = compute_cap >= (8, 0)  # Ampere and newer

        # Estimate CUDA cores (approximate)
        cuda_cores = self._estimate_cuda_cores(props.name, props.multi_processor_count)

        capabilities = GPUCapabilities(
            name=props.name,
            compute_capability=compute_cap,
            total_memory_gb=total_mem,
            available_memory_gb=available_mem,
            cuda_cores=cuda_cores,
            supports_fp16=supports_fp16,
            supports_bf16=supports_bf16,
            supports_int8=supports_int8,
            supports_flash_attention=supports_flash_attention,
            tensor_cores=tensor_cores,
            multi_gpu=torch.cuda.device_count() > 1,
            gpu_count=torch.cuda.device_count()
        )

        logger.info(f"""
🎮 GPU Capabilities Detected:
   Name: {capabilities.name}
   Compute: {capabilities.compute_capability[0]}.{capabilities.compute_capability[1]}
   Memory: {capabilities.available_memory_gb:.1f}GB / {capabilities.total_memory_gb:.1f}GB
   CUDA Cores: ~{capabilities.cuda_cores}
   Tensor Cores: {'✅' if capabilities.tensor_cores else '❌'}
   FP16: {'✅' if capabilities.supports_fp16 else '❌'}
   BF16: {'✅' if capabilities.supports_bf16 else '❌'}
   INT8: {'✅' if capabilities.supports_int8 else '❌'}
   Flash Attention: {'✅' if capabilities.supports_flash_attention else '❌'}
   Multi-GPU: {capabilities.gpu_count} GPUs
        """)

        return capabilities

    def _estimate_cuda_cores(self, gpu_name: str, sm_count: int) -> int:
        """Estimate CUDA cores based on GPU architecture"""
        cores_per_sm = {
            "RTX 4090": 128,
            "RTX 4080": 128,
            "A100": 64,
            "A10": 128,
            "T4": 64,
            "V100": 64,
            "RTX 3090": 128,
            "RTX 3080": 128,
        }

        for key, cores in cores_per_sm.items():
            if key in gpu_name:
                return sm_count * cores

        # Default estimate for unknown GPUs
        return sm_count * 64

    def _setup_cuda_optimizations(self):
        """Apply CUDA optimizations"""
        if self.device.type != "cuda":
            return

        # Enable TF32 for faster matmul on Ampere GPUs
        if self.capabilities.compute_capability >= (8, 0):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✅ TF32 enabled for Ampere GPU")

        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        logger.info("✅ cuDNN auto-tuner enabled")

        # Set memory allocator settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    def get_optimal_whisper_model(self) -> str:
        """Recommend optimal Whisper model based on VRAM"""
        if self.device.type == "cpu":
            return "tiny"

        vram = self.capabilities.available_memory_gb

        # Model VRAM requirements (approximate with FP16)
        # tiny: ~1GB, base: ~1.5GB, small: ~2GB, medium: ~5GB, large-v3: ~10GB
        if vram >= 20:
            return "large-v3"
        elif vram >= 8:
            return "medium"
        elif vram >= 4:
            return "small"
        elif vram >= 2:
            return "base"
        else:
            return "tiny"

    def get_optimal_compute_type(self) -> str:
        """Get optimal compute type for inference"""
        if self.device.type == "cpu":
            return "int8"

        if self.capabilities.supports_bf16:
            return "bfloat16"
        elif self.capabilities.supports_fp16:
            return "float16"
        else:
            return "float32"

    def get_optimal_batch_size(self, model_size: str = "medium") -> int:
        """Calculate optimal batch size for Whisper"""
        if self.device.type == "cpu":
            return 1

        vram = self.capabilities.available_memory_gb

        # Estimates based on model size and available VRAM
        batch_sizes = {
            "tiny": min(64, int(vram * 20)),
            "base": min(48, int(vram * 15)),
            "small": min(32, int(vram * 10)),
            "medium": min(16, int(vram * 4)),
            "large-v3": min(8, int(vram * 2)),
        }

        return max(1, batch_sizes.get(model_size, 8))

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics"""
        if self.device.type == "cpu":
            return {}

        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free, total = torch.cuda.mem_get_info(0)
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": free_gb,
            "total_gb": total_gb,
            "utilization_percent": (allocated / total_gb) * 100
        }

    def get_nvidia_smi_stats(self) -> Optional[Dict]:
        """Get stats from nvidia-smi"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                return {
                    "gpu_utilization_percent": float(values[0]),
                    "memory_utilization_percent": float(values[1]),
                    "temperature_celsius": float(values[2]),
                    "power_draw_watts": float(values[3])
                }
        except Exception as e:
            logger.debug(f"Could not get nvidia-smi stats: {e}")

        return None

    def optimize_for_latency(self):
        """Apply optimizations specifically for low latency"""
        if self.device.type != "cuda":
            return

        # Disable profiling overhead
        torch.autograd.set_detect_anomaly(False)

        # Use CUDA graphs for reduced overhead (requires warmup)
        logger.info("✅ Low-latency optimizations applied")

    def clear_cache(self):
        """Clear GPU cache"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")


# Global GPU manager instance
gpu_manager = GPUManager()


def get_gpu_config() -> Dict:
    """Get recommended GPU configuration for Pipecat pipeline"""
    manager = gpu_manager

    config = {
        "device": str(manager.device),
        "whisper_model": manager.get_optimal_whisper_model(),
        "compute_type": manager.get_optimal_compute_type(),
        "batch_size": manager.get_optimal_batch_size(),
        "enable_flash_attention": manager.capabilities.supports_flash_attention if manager.capabilities else False,
        "use_tensor_cores": manager.capabilities.tensor_cores if manager.capabilities else False,
    }

    logger.info(f"📋 Recommended GPU Config: {config}")
    return config


if __name__ == "__main__":
    # Test GPU detection
    logging.basicConfig(level=logging.INFO)

    print("\n🔍 Testing GPU Detection...\n")
    config = get_gpu_config()

    if gpu_manager.device.type == "cuda":
        print("\n📊 Memory Stats:")
        print(gpu_manager.get_memory_stats())

        print("\n🎯 NVIDIA-SMI Stats:")
        print(gpu_manager.get_nvidia_smi_stats())
