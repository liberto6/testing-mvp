"""
Main Entry Point for Pipecat Voice Pipeline
FastAPI server with WebSocket support
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.utils.config import ConfigManager, get_config_manager
from src.utils.gpu_utils import gpu_manager
from src.pipeline import SimplifiedVoicePipeline
from src.processors.stt_whisper_gpu import WhisperGPUConfig
from src.processors.llm_groq import GroqConfig
from src.processors.tts_kokoro import KokoroConfig
from src.processors.tts_edge import EdgeTTSConfig

from pipecat.frames.frames import AudioRawFrame, TextFrame

logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: SimplifiedVoicePipeline = None
config_manager: ConfigManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""

    global pipeline, config_manager

    # --- Startup ---
    logger.info("🚀 Starting Pipecat Voice Pipeline Server...")

    # Load configuration
    config_path = None  # Auto-detect or use environment
    config_manager = get_config_manager(config_path=config_path, auto_optimize=True)
    config_manager.print_config()

    # Initialize pipeline
    pipeline = SimplifiedVoicePipeline(config=config_manager.get_config())

    logger.info("✅ Pipeline ready")

    # Print GPU info
    if gpu_manager.device.type == "cuda":
        stats = gpu_manager.get_memory_stats()
        logger.info(f"🎮 GPU Stats: {stats}")

    yield

    # --- Shutdown ---
    logger.info("🛑 Shutting down server...")

    # Cleanup
    if gpu_manager.device.type == "cuda":
        gpu_manager.clear_cache()

    logger.info("✅ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Pipecat Voice Pipeline",
    description="GPU-Optimized Voice Conversation Pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Pipecat Voice Pipeline",
        "version": "1.0.0",
        "status": "running",
        "gpu_enabled": gpu_manager.device.type == "cuda"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""

    health_status = {
        "status": "healthy",
        "gpu": {
            "available": gpu_manager.device.type == "cuda",
            "device": str(gpu_manager.device)
        }
    }

    if gpu_manager.device.type == "cuda":
        health_status["gpu"]["memory"] = gpu_manager.get_memory_stats()

    return health_status


@app.get("/config")
async def get_config():
    """Get current configuration"""

    if config_manager:
        return {
            "stt": {
                "model": config_manager.config.stt.model,
                "device": config_manager.config.stt.device,
                "compute_type": config_manager.config.stt.compute_type
            },
            "llm": {
                "provider": config_manager.config.llm.provider,
                "model": config_manager.config.llm.model
            },
            "tts": {
                "provider": config_manager.config.tts.provider,
                "device": config_manager.config.tts.device,
                "voice": config_manager.config.tts.voice
            }
        }

    return {"error": "Configuration not loaded"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for voice conversation

    Maintains compatibility with existing frontend
    """

    await websocket.accept()
    logger.info("🔌 WebSocket connected")

    # Track current processing task for interruption handling
    current_task = None

    try:
        while True:
            # Receive message from client
            message = await websocket.receive()

            # Cancel previous task if still running (barge-in)
            if current_task and not current_task.done():
                logger.info("✋ Interruption detected, canceling previous response...")
                current_task.cancel()
                try:
                    await current_task
                except asyncio.CancelledError:
                    pass

            # Create new processing task
            current_task = asyncio.create_task(
                process_message(websocket, message)
            )

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")
        if current_task:
            current_task.cancel()

    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}", exc_info=True)
        if current_task:
            current_task.cancel()


async def process_message(websocket: WebSocket, message: dict):
    """Process a single message through the pipeline"""

    import time
    import numpy as np

    try:
        user_text = ""
        t_start = time.time()

        # Handle text input (Web Speech API)
        if "text" in message:
            user_text = message["text"]
            logger.info(f"📨 Received text: '{user_text}'")

        # Handle audio input (Whisper STT)
        elif "bytes" in message:
            audio_bytes = message["bytes"]

            if len(audio_bytes) == 0:
                return

            # Convert to numpy for silence detection
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Check for silence
            amplitude = np.max(np.abs(audio_np))
            if amplitude < 0.01:
                logger.debug("🔇 Silence detected, ignoring")
                return

            logger.info("🎤 Processing audio...")

            # Run STT
            t_stt_start = time.time()
            user_text = await pipeline.process_audio(audio_bytes)
            t_stt = time.time() - t_stt_start

            if not user_text:
                logger.warning("⚠️ STT returned no text")
                return

            logger.info(f"📝 Transcribed: '{user_text}' ({t_stt:.2f}s)")

        if not user_text:
            return

        # Send response start marker
        await websocket.send_json({"type": "response_start"})

        # Stream LLM -> TTS pipeline
        first_audio_sent = False
        t_first_byte = 0

        async for sentence in pipeline.stream_llm_response(user_text):
            logger.info(f"🤖 LLM: '{sentence[:30]}...'")

            # Generate TTS audio
            t_tts_start = time.time()
            audio_bytes = await pipeline.process_text_to_speech(sentence)
            t_tts = time.time() - t_tts_start

            if audio_bytes:
                # Send audio to client
                await websocket.send_bytes(audio_bytes)

                if not first_audio_sent:
                    t_first_byte = time.time() - t_start
                    logger.info(f"⚡ Time to first audio: {t_first_byte:.2f}s")
                    first_audio_sent = True

                logger.info(f"🔊 TTS: {len(sentence)} chars in {t_tts:.2f}s")

        # Total latency
        t_total = time.time() - t_start
        logger.info(f"✅ Total latency: {t_total:.2f}s")

    except asyncio.CancelledError:
        logger.info("🛑 Processing canceled (user interrupted)")
        raise

    except Exception as e:
        logger.error(f"❌ Processing error: {e}", exc_info=True)


# Serve static files (for frontend)
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Serve frontend files
    @app.get("/index.html")
    async def serve_index():
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run server
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload in production
        log_level="info"
    )
