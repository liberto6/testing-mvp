import io
import numpy as np
import scipy.io.wavfile as wavfile
from app.core.config import DEVICE, REF_AUDIO, REF_TEXT
from app.core.logging import logger

tts_model = None

def init_f5():
    global tts_model
    if tts_model is None:
        print(f"Cargando F5-TTS en {DEVICE}...")
        # Importación Lazy para evitar error si no está instalado
        try:
            from f5_tts.api import F5TTS
            tts_model = F5TTS(device=DEVICE)
        except ImportError:
            logger.error("❌ No se pudo importar f5_tts. Asegúrate de que está instalado o en el path.")
            tts_model = None

def generate_audio_f5(text):
    global tts_model
    if tts_model is None:
        init_f5()
        
    if tts_model is None:
        return None

    try:
        # F5-TTS infer devuelve: audio (numpy), sr, spectr
        # Pasamos file_wave=None para que no escriba en disco
        audio, sr, _ = tts_model.infer(
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
        logger.error(f"❌ Error en F5-TTS Worker: {e}")
        return None
