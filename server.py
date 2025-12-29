import os
import torch
import numpy as np
import uvicorn
import time
import asyncio
import io
import re
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv() # Cargar variables de entorno desde .env

from groq import AsyncGroq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from f5_tts.api import F5TTS
from fastapi.middleware.cors import CORSMiddleware
import scipy.io.wavfile as wavfile

# Inicializar cliente Groq Asíncrono
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️ ADVERTENCIA: No se encontró la variable de entorno GROQ_API_KEY.")
    
client = AsyncGroq(api_key=GROQ_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURACIÓN GLOBAL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_AUDIO = os.path.join(BASE_DIR, "F5-TTS", "src", "f5_tts", "infer", "examples", "basic", "basic_ref_en.wav")
REF_TEXT = "Some call me nature, others call me mother nature"

SYSTEM_PROMPT = """
You are Sarah, an expert English teacher. 
Rules:
1. Always respond in English.
2. If the user speaks Spanish, translate their message to English and then answer.
3. If the user makes a mistake in English, politely correct them before continuing the conversation.
4. Keep your answers concise and friendly.
"""

# --- INICIALIZACIÓN DE MODELOS ---
print("Configurando dispositivo...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo seleccionado: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. STT (Whisper) - CPU
# Usamos int8 en CPU para STT como estaba configurado
print(f"Cargando Whisper (STT) en CPU...")
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# 2. TTS (F5-TTS) - GPU/CPU
print(f"Cargando F5-TTS en {device}...")
tts = F5TTS(device=device)

# Executor para tareas bloqueantes (STT y TTS)
# Max workers limitado para no saturar CPU/GPU
executor = ThreadPoolExecutor(max_workers=3) 

# --- FUNCIONES DE AYUDA (NO BLOQUEANTES) ---

async def run_stt(audio_np):
    """Ejecuta Whisper en un hilo separado para no bloquear el loop de eventos."""
    loop = asyncio.get_running_loop()
    # Whisper transcribe es bloqueante
    result = await loop.run_in_executor(executor, _execute_whisper, audio_np)
    return result

def _execute_whisper(audio_np):
    segments, _ = stt_model.transcribe(audio_np, language="en")
    return " ".join([s.text for s in segments]).strip()

async def run_tts(text):
    """Ejecuta F5-TTS en un hilo separado."""
    if not text.strip():
        return None
    
    loop = asyncio.get_running_loop()
    wav_bytes = await loop.run_in_executor(executor, _execute_tts, text)
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

# --- MANEJO DE FLUJO DE LLM ---

async def stream_sentences(user_text):
    """
    Genera respuesta del LLM y cede oraciones completas lo más rápido posible.
    """
    try:
        completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            model="llama-3.3-70b-versatile",
            stream=True
        )

        buffer = ""
        # Regex simple para detectar finales de oración (. ! ?)
        # Cuidado con abreviaciones, pero para MVP funciona bien
        sentence_endings = re.compile(r'(?<=[.!?])\s+')

        async for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                buffer += content
                
                # Intentar dividir por oraciones
                parts = sentence_endings.split(buffer)
                
                # Si hay más de una parte, es que encontramos un delimitador
                if len(parts) > 1:
                    # Todo menos el último fragmento son oraciones completas
                    for sentence in parts[:-1]:
                        if sentence.strip():
                            yield sentence.strip()
                    
                    # El último fragmento es el comienzo de la siguiente oración
                    buffer = parts[-1]
        
        # Rendir lo que quede en el buffer al final
        if buffer.strip():
            yield buffer.strip()

    except Exception as e:
        print(f"⚠️ Error Groq Stream: {e}")
        yield "Sorry, I had an error."

# --- RUTAS ---

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/ort-wasm-simd.wasm")
async def get_wasm_simd():
    return FileResponse(os.path.join(BASE_DIR, "ort-wasm-simd.wasm"), media_type="application/wasm")

@app.get("/ort-wasm.wasm")
async def get_wasm_basic():
    return FileResponse(os.path.join(BASE_DIR, "ort-wasm.wasm"), media_type="application/wasm")

@app.get("/silero_vad.onnx")
async def get_model():
    return FileResponse(os.path.join(BASE_DIR, "silero_vad.onnx"), media_type="application/octet-stream")

@app.get("/vad.js")
async def get_vad():
    return FileResponse(os.path.join(BASE_DIR, "vad.js"), media_type="application/javascript")

@app.get("/ort.js")
async def get_ort():
    return FileResponse(os.path.join(BASE_DIR, "ort.js"), media_type="application/javascript")

@app.get("/vad.worklet.v2.js")
async def get_worklet_v2():
    return FileResponse(os.path.join(BASE_DIR, "vad.worklet.v2.js"), media_type="application/javascript")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🚀 Pipeline Streaming conectado.")
    
    try:
        while True:
            # 1. RECIBIR AUDIO (Esperar datos)
            data = await websocket.receive_bytes()
            
            if len(data) == 0: continue
            
            # 2. STT (Procesar entrada)
            # Convertir bytes a numpy
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Verificar si hay señal
            amplitude = np.max(np.abs(audio_np))
            if amplitude < 0.01:
                print("DEBUG: Silencio detectado, ignorando.")
                continue

            print("🎤 Procesando voz de usuario...")
            t0 = time.time()
            user_text = await run_stt(audio_np)
            print(f"📝 Transcripción ({time.time() - t0:.2f}s): {user_text}")
            
            if not user_text: continue

            # 3. PIPELINE LLM -> TTS (Streaming)
            # Iniciamos la generación de respuesta
            print("🤖 Generando respuesta (Streaming)...")
            
            async for sentence in stream_sentences(user_text):
                print(f"  🗣️ Frase detectada: {sentence}")
                
                # Generar audio para esta frase en paralelo
                # (Mientras esto ocurre, el loop sigue disponible si hubiéramos diseñado 
                #  una arquitectura full-duplex real, pero aquí iteramos sobre el stream)
                t_tts = time.time()
                audio_bytes = await run_tts(sentence)
                
                if audio_bytes:
                    await websocket.send_bytes(audio_bytes)
                    print(f"  ✅ Audio enviado ({len(audio_bytes)} bytes) - TTS: {time.time() - t_tts:.2f}s")
                else:
                    print("  ❌ Falló generación de audio para frase.")

    except WebSocketDisconnect:
        print("🔌 Cliente desconectado.")
    except Exception as e:
        print(f"🔥 Error Pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
