import asyncio
from app.core.config import TTS_ENGINE
from app.core.executor import executor
from app.services.tts_f5 import generate_audio_f5
from app.services.tts_kokoro import generate_audio_kokoro

# --- CACHÉ TTS (Fase 2) ---
TTS_CACHE = {}
MAX_CACHE_SIZE = 1000

async def run_tts(text):
    """
    Ejecuta el motor TTS configurado en un hilo separado con Caché.
    Motor actual: {TTS_ENGINE}
    """
    if not text.strip():
        return None
        
    # Check Caché
    if text in TTS_CACHE:
        return TTS_CACHE[text]
    
    # Seleccionar función generadora
    if TTS_ENGINE == "kokoro":
        generator_func = generate_audio_kokoro
    else:
        generator_func = generate_audio_f5
    
    loop = asyncio.get_running_loop()
    # Ejecutar en ThreadPoolExecutor para no bloquear el Event Loop
    wav_bytes = await loop.run_in_executor(executor, generator_func, text)
    
    # Guardar en Caché
    if wav_bytes:
        if len(TTS_CACHE) >= MAX_CACHE_SIZE:
            TTS_CACHE.pop(next(iter(TTS_CACHE))) # Eliminar el más antiguo (simple FIFO)
        TTS_CACHE[text] = wav_bytes
        
    return wav_bytes
