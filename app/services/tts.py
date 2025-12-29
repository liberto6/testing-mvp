import asyncio
import io
import numpy as np
import scipy.io.wavfile as wavfile
from f5_tts.api import F5TTS
from app.core.config import DEVICE, REF_AUDIO, REF_TEXT
from app.core.executor import executor
from app.core.logging import logger

print(f"Cargando F5-TTS en {DEVICE}...")
tts = F5TTS(device=DEVICE)

# --- CACHÉ TTS (Fase 2) ---
TTS_CACHE = {}
MAX_CACHE_SIZE = 1000

async def run_tts(text):
    """Ejecuta F5-TTS en un hilo separado con Caché."""
    if not text.strip():
        return None
        
    # Check Caché
    if text in TTS_CACHE:
        return TTS_CACHE[text]
    
    loop = asyncio.get_running_loop()
    wav_bytes = await loop.run_in_executor(executor, _execute_tts, text)
    
    # Guardar en Caché
    if wav_bytes:
        if len(TTS_CACHE) >= MAX_CACHE_SIZE:
            TTS_CACHE.pop(next(iter(TTS_CACHE))) # Eliminar el más antiguo (simple FIFO)
        TTS_CACHE[text] = wav_bytes
        
    return wav_bytes

def _execute_tts(text):
    try:
        # F5-TTS infer devuelve: audio (numpy), sr, spectr
        # Pasamos file_wave=None para que no escriba en disco
        audio, sr, _ = tts.infer(
            gen_text=text,
            ref_file=REF_AUDIO,
            ref_text=REF_TEXT,
            file_wave=None, 
            file_spec=None
        )
        
        if len(audio) == 0:
            return None

        # Convertir a Int16
        audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        
        # Escribir a memoria (BytesIO) en lugar de disco
        byte_io = io.BytesIO()
        wavfile.write(byte_io, sr, audio)
        return byte_io.getvalue()
    except Exception as e:
        print(f"❌ Error en TTS Worker: {e}")
        return None
