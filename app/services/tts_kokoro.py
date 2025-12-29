import io
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from kokoro import KPipeline
from app.core.config import KOKORO_VOICE, DEVICE
from app.core.logging import logger

pipeline = None

def init_kokoro():
    global pipeline
    if pipeline is None:
        print(f"Cargando Kokoro TTS en {DEVICE}...")
        try:
            pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M') 
        except Exception as e:
            logger.error(f"Error inicializando Kokoro: {e}")
            pipeline = None

def generate_audio_kokoro(text):
    """
    Genera audio usando Kokoro TTS.
    Retorna bytes WAV.
    """
    global pipeline
    if pipeline is None:
        init_kokoro()

    if not pipeline:
        logger.error("Kokoro pipeline no está inicializado.")
        return None

    try:
        # Kokoro genera un generador de segmentos.
        # Para streaming real, deberíamos ceder chunks, pero aquí seguiremos
        # el patrón actual de devolver la frase completa como un WAV.
        generator = pipeline(
            text, 
            voice=KOKORO_VOICE,
            speed=1.0, 
            split_pattern=r'\n+'
        )
        
        # Concatenar todos los segmentos de audio
        full_audio = []
        # El generador de Kokoro retorna: (graphemes, phonemes, audio_tensor)
        for i, (gs, ps, audio) in enumerate(generator):
            full_audio.append(audio)
            
        if not full_audio:
            return None
            
        # Unir y procesar
        audio_np = np.concatenate(full_audio)
        
        # Convertir a Int16 (Kokoro devuelve float32 en rango -1 a 1 usualmente)
        audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
        
        # Escribir a WAV en memoria
        byte_io = io.BytesIO()
        wavfile.write(byte_io, 24000, audio_int16) # Kokoro usa 24khz por defecto
        return byte_io.getvalue()

    except Exception as e:
        logger.error(f"❌ Error en Kokoro TTS: {e}")
        return None
