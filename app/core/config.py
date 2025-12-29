import os
import torch
from dotenv import load_dotenv

load_dotenv()

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(BASE_DIR, "static")
F5_TTS_DIR = os.path.join(BASE_DIR, "F5-TTS")

# Configuración de Modelos
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# TTS Config
REF_AUDIO = os.path.join(F5_TTS_DIR, "src", "f5_tts", "infer", "examples", "basic", "basic_ref_en.wav")
REF_TEXT = "Some call me nature, others call me mother nature"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = """
You are Sarah, an expert English teacher. 
Rules:
1. Always respond in English.
2. If the user speaks Spanish, translate their message to English and then answer.
3. If the user makes a mistake in English, politely correct them before continuing the conversation.
4. Keep your answers concise and friendly.
"""
