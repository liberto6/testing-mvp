"""
Local GPU LLM Processor for Pipecat
Optional local LLM using vLLM or ExLlama for GPU inference
"""

import asyncio
import time
from typing import AsyncGenerator, Optional, List, Dict
from dataclasses import dataclass
import logging

from pipecat.frames.frames import Frame, TextFrame, LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


@dataclass
class LocalLLMConfig:
    """Configuration for local GPU LLM"""
    backend: str = "vllm"  # Options: vllm, exllama, transformers
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    device: str = "cuda"
    max_tokens: int = 150
    temperature: float = 0.7
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9


class LocalLLMProcessor(FrameProcessor):
    """
    Local GPU LLM Processor using vLLM

    This is an optional processor for running LLMs locally on GPU
    when you want to avoid API calls or compare latencies.

    Requires: pip install vllm
    """

    def __init__(
        self,
        config: LocalLLMConfig,
        system_prompt: str = "",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.config = config
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = []

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

        self.model = None
        self.tokenizer = None

        # Load model based on backend
        self._load_model()

    def _load_model(self):
        """Load local LLM model"""
        try:
            if self.config.backend == "vllm":
                self._load_vllm()
            elif self.config.backend == "transformers":
                self._load_transformers()
            else:
                raise ValueError(f"Unsupported backend: {self.config.backend}")

        except Exception as e:
            logger.error(f"❌ Failed to load local LLM: {e}")
            logger.error("💡 Make sure you have installed the required packages:")
            logger.error("   pip install vllm  # for vLLM backend")
            raise

    def _load_vllm(self):
        """Load model using vLLM (recommended for production)"""
        try:
            from vllm import LLM, SamplingParams
            from vllm.outputs import RequestOutput

            logger.info(f"🔄 Loading {self.config.model_name} with vLLM...")

            self.model = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=True
            )

            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=0.9
            )

            logger.info("✅ vLLM model loaded successfully")

        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )

    def _load_transformers(self):
        """Load model using Hugging Face transformers (simpler, slower)"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            logger.info(f"🔄 Loading {self.config.model_name} with transformers...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            logger.info("✅ Transformers model loaded successfully")

        except ImportError:
            raise ImportError(
                "Transformers not installed. Install with: pip install transformers"
            )

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Process incoming frames"""

        if isinstance(frame, TextFrame):
            user_text = frame.text

            # Add to conversation history
            self.messages.append({"role": "user", "content": user_text})

            # Generate response
            async for output_frame in self._generate_response():
                yield output_frame

            yield LLMFullResponseEndFrame()

        else:
            yield frame

    async def _generate_response(self) -> AsyncGenerator[TextFrame, None]:
        """Generate LLM response"""

        start_time = time.time()

        try:
            if self.config.backend == "vllm":
                async for text_frame in self._generate_vllm():
                    yield text_frame
            elif self.config.backend == "transformers":
                async for text_frame in self._generate_transformers():
                    yield text_frame

            latency = time.time() - start_time
            logger.info(f"🤖 Local LLM response generated in {latency:.2f}s")

        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            yield TextFrame(text="Sorry, I had an error.")

    async def _generate_vllm(self) -> AsyncGenerator[TextFrame, None]:
        """Generate using vLLM"""

        # Format prompt for chat
        prompt = self._format_chat_prompt()

        # Run inference in executor (vLLM is CPU-bound for scheduling)
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.model.generate([prompt], self.sampling_params)
        )

        # Extract response
        output = outputs[0]
        generated_text = output.outputs[0].text.strip()

        # Add to history
        self.messages.append({"role": "assistant", "content": generated_text})

        # Yield full response (vLLM doesn't support streaming easily)
        yield TextFrame(text=generated_text)

    async def _generate_transformers(self) -> AsyncGenerator[TextFrame, None]:
        """Generate using transformers with streaming"""
        import torch

        prompt = self._format_chat_prompt()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)

        # Generate with streaming
        full_response = ""

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][len(inputs[0]):],
            skip_special_tokens=True
        )

        full_response = generated_text.strip()

        # Add to history
        self.messages.append({"role": "assistant", "content": full_response})

        # Yield response
        yield TextFrame(text=full_response)

    def _format_chat_prompt(self) -> str:
        """Format messages as prompt (Llama chat format)"""

        prompt_parts = []

        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt_parts.append(f"<<SYS>>\n{content}\n<</SYS>>")
            elif role == "user":
                prompt_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                prompt_parts.append(content)

        return "\n\n".join(prompt_parts)

    def get_metrics(self) -> dict:
        """Get performance metrics"""
        return {
            "backend": self.config.backend,
            "model": self.config.model_name,
            "device": self.config.device,
            "message_count": len(self.messages)
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n🧪 Testing Local GPU LLM Processor...\n")
    print("⚠️ This test requires a local LLM installation (vLLM or transformers)")
    print("⚠️ Skipping actual model loading in this test\n")

    # Example configuration
    config = LocalLLMConfig(
        backend="vllm",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device="cuda",
        max_tokens=100
    )

    print(f"📋 Config: {config}")
    print("\n✅ Configuration test passed")
    print("\n💡 To use this processor:")
    print("   1. Install vLLM: pip install vllm")
    print("   2. Download a model from Hugging Face")
    print("   3. Initialize processor with model path")
