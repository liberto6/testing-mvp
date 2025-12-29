import asyncio
from faster_whisper import WhisperModel
from app.core.executor import executor
from app.core.logging import logger

print(f"Cargando Whisper (STT) en CPU...")
stt_model = WhisperModel("small", device="cuda", compute_type="float16")

async def run_stt(audio_np):
    """Ejecuta Whisper en un hilo separado para no bloquear el loop de eventos."""
    loop = asyncio.get_running_loop()
    # Whisper transcribe es bloqueante
    result = await loop.run_in_executor(executor, _execute_whisper, audio_np)
    return result

def _execute_whisper(audio_np):
    segments, _ = stt_model.transcribe(audio_np, language="en", beam_size=1, best_of=1) # Optimización Fase 2
    return " ".join([s.text for s in segments]).strip()
