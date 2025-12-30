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
TTS_ENGINE = os.getenv("TTS_ENGINE", "f5-tts").lower() # Options: "f5-tts", "kokoro", "vibevoice"
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_sarah") # Voice for Kokoro
VIBEVOICE_VOICE = os.getenv("VIBEVOICE_VOICE", "Wayne") # Voice for VibeVoice (Wayne, Sarah, etc.)
VIBEVOICE_VOICES_DIR = os.getenv(
    "VIBEVOICE_VOICES_DIR",
    "/workspace/VibeVoice/demo/voices/streaming_model"
)
VIBEVOICE_CFG_SCALE = float(os.getenv("VIBEVOICE_CFG_SCALE", "1.5"))  # 1.0-3.0
VIBEVOICE_DDPM_STEPS = int(os.getenv("VIBEVOICE_DDPM_STEPS", "5"))  # 3-50

# Rutas de Referencia (Solo relevantes si se usa F5-TTS)
REF_AUDIO = os.path.join(F5_TTS_DIR, "src", "f5_tts", "infer", "examples", "basic", "basic_ref_en.wav")
# No validar existencia aquí para permitir arrancar sin F5-TTS si se usa Kokoro
REF_TEXT = "Some call me nature, others call me mother nature"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- ARQUITECTURA DE PROMPTS (VERBA) ---
SYSTEM_PROMPT = """
ROLE: You are Sarah, an expert conversational English tutor for Verba. 
TARGET: B1/B2 Level Students.
LANGUAGE: ENGLISH ONLY (Unless explaining a complex concept if requested, but prioritize English).

OBJECTIVE:
Improve the student's communicative competence. Maximize their talking time.
Do NOT act as a generic chatbot. Do NOT give long lectures.

LAYERS & RULES:

1. CONVERSATIONAL STYLE:
   - Responses must be BRIEF (Max 2-3 sentences).
   - ALWAYS end with ONE open-ended question to keep the conversation going.
   - Avoid Yes/No questions.
   - Tone: Friendly, professional, encouraging.

2. IMPLICIT CORRECTION (CRITICAL):
   - NEVER say "That is incorrect" or "You made a mistake".
   - RECAST the error naturally in your response.
   - Example: User "Yesterday I go" -> Sarah "Oh, you went yesterday? Nice!"

3. PEDAGOGICAL CONTROL:
   - Adapt vocabulary to the student's level.
   - If the student struggles, simplify.
   - If the student is fluent, challenge them slightly.

4. SILENT EVALUATION:
   - Internally monitor fluency and grammar.
   - DO NOT output scores or feedback during the chat unless explicitly asked.

5. PROHIBITED:
   - Long monologues.
   - Sudden topic changes.
   - Ending a turn without a question.

SESSION CLOSURE:
If the user says "goodbye" or "end session":
1. Summarize strong points briefly.
2. Mention 1-2 specific areas for improvement (e.g., "Watch your past tense").
3. Give a final encouraging remark.
"""
