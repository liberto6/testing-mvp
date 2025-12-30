import asyncio
from app.core.config import TTS_ENGINE, VIBEVOICE_VOICE
from app.core.executor import executor

# --- CACHÉ TTS (Fase 2) ---
TTS_CACHE = {}
MAX_CACHE_SIZE = 1000

async def run_tts(text):
    """
    Ejecuta el motor TTS configurado en un hilo separado con Caché.
    Motor actual: {TTS_ENGINE}

    Motores disponibles:
    - f5-tts: F5-TTS (síntesis natural, requiere audio de referencia)
    - kokoro: Kokoro-82M (rápido, 24kHz)
    - vibevoice: Microsoft VibeVoice Realtime (streaming, 24kHz, alta calidad)
    """
    if not text.strip():
        return None

    # Check Caché
    if text in TTS_CACHE:
        return TTS_CACHE[text]

    # Seleccionar función generadora con Lazy Import
    if TTS_ENGINE == "kokoro":
        from app.services.tts_kokoro import generate_audio_kokoro
        generator_func = generate_audio_kokoro
        loop = asyncio.get_running_loop()
        wav_bytes = await loop.run_in_executor(executor, generator_func, text)
    elif TTS_ENGINE == "vibevoice":
        from app.services.tts_vibevoice import generate_audio_vibevoice
        loop = asyncio.get_running_loop()
        # Pass voice_name from config
        wav_bytes = await loop.run_in_executor(executor, generate_audio_vibevoice, text, VIBEVOICE_VOICE)
    else:
        from app.services.tts_f5 import generate_audio_f5
        generator_func = generate_audio_f5
        loop = asyncio.get_running_loop()
        wav_bytes = await loop.run_in_executor(executor, generator_func, text)

    # Guardar en Caché
    if wav_bytes:
        if len(TTS_CACHE) >= MAX_CACHE_SIZE:
            TTS_CACHE.pop(next(iter(TTS_CACHE))) # Eliminar el más antiguo (simple FIFO)
        TTS_CACHE[text] = wav_bytes

    return wav_bytes
