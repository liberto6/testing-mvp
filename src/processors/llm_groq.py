"""
Groq LLM Processor for Pipecat
Ultra-low latency LLM with Llama models via Groq API
"""

import asyncio
import time
import re
from typing import AsyncGenerator, Optional, List, Dict
from dataclasses import dataclass
import logging

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMMessagesFrame,
    LLMFullResponseEndFrame
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.ai_services import LLMService

from groq import AsyncGroq

logger = logging.getLogger(__name__)


@dataclass
class GroqConfig:
    """Configuration for Groq LLM"""
    api_key: str
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.7
    max_tokens: int = 150
    stream: bool = True
    # Streaming optimization
    min_chunk_length: int = 20  # Minimum characters before yielding on weak delimiter


class GroqLLMProcessor(FrameProcessor):
    """
    Groq LLM Processor with aggressive streaming

    Features:
    - Ultra-low latency with Groq API
    - Aggressive sentence chunking for fast TTS pipeline
    - Smart punctuation-based streaming
    - Conversation history management
    """

    def __init__(
        self,
        config: GroqConfig,
        system_prompt: str = "",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.config = config
        self.system_prompt = system_prompt
        self.client = AsyncGroq(api_key=config.api_key)

        # Conversation history
        self.messages: List[Dict[str, str]] = []

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

        # Metrics
        self.request_count = 0
        self.total_tokens = 0
        self.total_latency = 0.0

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Process incoming frames"""

        # Handle transcription/text input
        if isinstance(frame, TextFrame):
            user_text = frame.text

            # Add to conversation history
            self.messages.append({"role": "user", "content": user_text})

            # Stream LLM response
            async for output_frame in self._stream_llm_response():
                yield output_frame

            # Yield end marker
            yield LLMFullResponseEndFrame()

        else:
            # Pass through other frames
            yield frame

    async def _stream_llm_response(self) -> AsyncGenerator[TextFrame, None]:
        """Stream LLM response with aggressive chunking"""

        full_response = ""
        start_time = time.time()

        try:
            # Create streaming completion
            completion = await self.client.chat.completions.create(
                messages=self.messages,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=self.config.stream
            )

            buffer = ""
            first_token_latency = None

            async for chunk in completion:
                content = chunk.choices[0].delta.content

                if content:
                    # Record first token latency
                    if first_token_latency is None:
                        first_token_latency = time.time() - start_time
                        logger.debug(f"⚡ First token: {first_token_latency*1000:.0f}ms")

                    buffer += content
                    full_response += content

                    # Try to yield complete sentences/phrases
                    while True:
                        # Look for delimiters (strong: .!? weak: ,;:\n)
                        match = re.search(r'([.?!])|([,;:\—\n])', buffer)

                        if not match:
                            break

                        delimiter = match.group(0)
                        end_idx = match.end()
                        candidate = buffer[:end_idx].strip()

                        # Skip empty candidates
                        if not candidate:
                            buffer = buffer[end_idx:]
                            continue

                        # Determine if we should yield
                        is_strong = delimiter in ".!?"
                        is_long_enough = len(candidate) >= self.config.min_chunk_length

                        if is_strong or is_long_enough:
                            # Yield text chunk
                            yield TextFrame(text=candidate)

                            buffer = buffer[end_idx:]
                        else:
                            # Wait for more text
                            break

            # Yield remaining buffer
            if buffer.strip():
                yield TextFrame(text=buffer.strip())

            # Update conversation history
            self.messages.append({"role": "assistant", "content": full_response.strip()})

            # Update metrics
            self.request_count += 1
            latency = time.time() - start_time
            self.total_latency += latency

            logger.info(
                f"🤖 LLM Response: '{full_response[:50]}...' "
                f"({latency:.2f}s, TTFT: {first_token_latency*1000:.0f}ms)"
            )

        except Exception as e:
            logger.error(f"❌ Groq API error: {e}")
            yield TextFrame(text="Sorry, I had an error processing your request.")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.messages.copy()

    def clear_history(self, keep_system_prompt: bool = True):
        """Clear conversation history"""
        if keep_system_prompt and self.system_prompt:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        else:
            self.messages = []

        logger.info("🗑️ Conversation history cleared")

    def get_metrics(self) -> dict:
        """Get performance metrics"""
        avg_latency = (
            self.total_latency / self.request_count
            if self.request_count > 0
            else 0
        )

        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "total_latency": self.total_latency,
            "average_latency": avg_latency,
        }


class GroqLLMService(LLMService):
    """
    Pipecat LLMService implementation for Groq

    This is a more integrated version that works with Pipecat's
    built-in LLM frame handling.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    async def _stream_chat_completions(
        self,
        messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Stream chat completions"""

        completion = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=True
        )

        async for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Process frames using Pipecat's LLM patterns"""

        if isinstance(frame, LLMMessagesFrame):
            # Stream response
            async for text_chunk in self._stream_chat_completions(frame.messages):
                yield TextFrame(text=text_chunk)

            yield LLMFullResponseEndFrame()

        else:
            yield frame


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    async def test_groq():
        """Test Groq processor"""
        print("\n🧪 Testing Groq LLM Processor...\n")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ GROQ_API_KEY not found in environment")
            return

        config = GroqConfig(
            api_key=api_key,
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=100
        )

        processor = GroqLLMProcessor(
            config=config,
            system_prompt="You are a helpful assistant. Be brief and friendly."
        )

        # Simulate user input
        user_frame = TextFrame(text="Hello! How are you?")

        print("🎯 Sending: 'Hello! How are you?'\n")
        print("🤖 Response chunks:")

        async for output_frame in processor.process_frame(user_frame):
            if isinstance(output_frame, TextFrame):
                print(f"   📝 '{output_frame.text}'")

        print("\n📊 Metrics:", processor.get_metrics())

    # Run test
    asyncio.run(test_groq())
