"""
WebSocket Transport for Pipecat
Custom WebSocket transport for direct browser connections
"""

import asyncio
import json
import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

from fastapi import WebSocket, WebSocketDisconnect
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TextFrame,
    TranscriptionFrame
)
from pipecat.transports.base_transport import BaseTransport

logger = logging.getLogger(__name__)


@dataclass
class WebSocketTransportConfig:
    """Configuration for WebSocket transport"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 8192


class WebSocketTransport(BaseTransport):
    """
    WebSocket transport for Pipecat

    Handles bidirectional audio/text communication over WebSocket
    Compatible with existing WebSocket frontend
    """

    def __init__(
        self,
        websocket: WebSocket,
        config: WebSocketTransportConfig,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.websocket = websocket
        self.config = config
        self.connected = False

        # Task for receiving WebSocket messages
        self.receive_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start WebSocket transport"""

        try:
            # Accept WebSocket connection
            await self.websocket.accept()
            self.connected = True

            logger.info("🔌 WebSocket transport connected")

            # Start receiving task
            self.receive_task = asyncio.create_task(self._receive_loop())

            await super().start()

        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket transport: {e}")
            raise

    async def stop(self):
        """Stop WebSocket transport"""

        logger.info("🛑 Stopping WebSocket transport...")

        self.connected = False

        # Cancel receive task
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass

        await super().stop()

    async def _receive_loop(self):
        """Receive messages from WebSocket"""

        try:
            while self.connected:
                # Receive message
                message = await self.websocket.receive()

                # Handle different message types
                if "text" in message:
                    # Text transcription from client (Web Speech API)
                    data = json.loads(message["text"])

                    if "text" in data:
                        user_text = data["text"]
                        logger.info(f"📨 Received text: '{user_text}'")

                        # Create transcription frame
                        frame = TranscriptionFrame(
                            text=user_text,
                            user_id="user",
                            timestamp=asyncio.get_event_loop().time()
                        )

                        # Push to pipeline
                        await self.push_frame(frame)

                elif "bytes" in message:
                    # Audio bytes from client
                    audio_bytes = message["bytes"]

                    if len(audio_bytes) > 0:
                        # Convert to numpy array (assuming int16 PCM)
                        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

                        # Check for silence
                        amplitude = np.max(np.abs(audio_np))
                        if amplitude < 320:  # ~0.01 in float32
                            continue

                        logger.debug(f"🎤 Received audio: {len(audio_bytes)} bytes")

                        # Create audio frame
                        frame = AudioRawFrame(
                            audio=audio_bytes,
                            sample_rate=self.config.sample_rate,
                            num_channels=self.config.channels
                        )

                        # Push to pipeline
                        await self.push_frame(frame)

        except WebSocketDisconnect:
            logger.info("🔌 WebSocket disconnected")
            self.connected = False

        except Exception as e:
            logger.error(f"❌ WebSocket receive error: {e}")
            self.connected = False

    async def send_audio(self, audio_frame: AudioRawFrame):
        """Send audio frame to client"""

        if not self.connected:
            return

        try:
            # Send raw audio bytes
            await self.websocket.send_bytes(audio_frame.audio)

            logger.debug(f"📤 Sent audio: {len(audio_frame.audio)} bytes")

        except Exception as e:
            logger.error(f"❌ Failed to send audio: {e}")

    async def send_message(self, message: dict):
        """Send JSON message to client"""

        if not self.connected:
            return

        try:
            await self.websocket.send_json(message)

        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")

    async def write_frame_to_transport(self, frame: Frame):
        """Write frame to WebSocket transport"""

        # Handle audio output
        if isinstance(frame, AudioRawFrame):
            await self.send_audio(frame)

        # Handle text output (for debugging or UI updates)
        elif isinstance(frame, TextFrame):
            await self.send_message({
                "type": "text",
                "text": frame.text
            })

        # Handle transcription (echo back user input if needed)
        elif isinstance(frame, TranscriptionFrame):
            await self.send_message({
                "type": "transcription",
                "text": frame.text,
                "user_id": frame.user_id
            })


class WebSocketServerTransport:
    """
    WebSocket server transport manager

    Manages WebSocket connections and creates transport instances
    """

    def __init__(self, config: Optional[WebSocketTransportConfig] = None):
        self.config = config or WebSocketTransportConfig()

    async def create_transport(self, websocket: WebSocket) -> WebSocketTransport:
        """Create transport for a WebSocket connection"""

        transport = WebSocketTransport(
            websocket=websocket,
            config=self.config
        )

        await transport.start()

        return transport


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n🧪 WebSocket Transport Module\n")
    print("✅ This module provides WebSocket transport for Pipecat")
    print("📝 Use with FastAPI WebSocket endpoints")
    print("\nExample usage:")
    print("""
    from fastapi import FastAPI, WebSocket
    from src.transports.websocket_transport import WebSocketServerTransport

    app = FastAPI()
    transport_manager = WebSocketServerTransport()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        transport = await transport_manager.create_transport(websocket)
        # Use transport with Pipecat pipeline
    """)
