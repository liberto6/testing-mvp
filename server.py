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
import logging
from logging.config import dictConfig

# Configuración de Logging Estructurado
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%H:%M:%S",
        },
        "metrics": {
            "format": "\n📊 METRICS --------------------------------------------------\n%(message)s\n-----------------------------------------------------------\n",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "metrics_console": {
            "class": "logging.StreamHandler",
            "formatter": "metrics",
            "level": "INFO",
        },
    },
    "loggers": {
        "root": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False
        },
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO", 
            "propagate": False
        },
        "metrics": {
            "handlers": ["metrics_console"],
            "level": "INFO",
            "propagate": False
        }
    },
}
dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("root")
metrics_logger = logging.getLogger("metrics")

# ... (resto de imports y código)


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
    Genera respuesta del LLM y cede fragmentos de texto lo más rápido posible.
    Estrategia: Streaming agresivo (Chunking por puntuación y longitud).
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
        MIN_CHUNK_LENGTH = 20  # Mínimo caracteres para cortar en coma (evita cortes robóticos en "Yes,")

        async for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                buffer += content
                
                while True:
                    # Buscamos el primer delimitador (fuerte o débil)
                    # [.!?] -> Fuerte
                    # [,;:\—\n] -> Débil
                    match = re.search(r'([.?!])|([,;:\—\n])', buffer)
                    
                    if not match:
                        break
                    
                    delimiter = match.group(0) # El caracter que hizo match
                    end_idx = match.end()      # Índice justo después del delimitador
                    
                    candidate = buffer[:end_idx].strip()
                    
                    # Si quedó vacío (ej: "...")
                    if not candidate:
                        buffer = buffer[end_idx:]
                        continue

                    # Verificar condiciones
                    is_strong = delimiter in ".!?"
                    is_long_enough = len(candidate) >= MIN_CHUNK_LENGTH
                    
                    # Cortamos SI: Es fuerte O (Es débil Y es suficientemente largo)
                    if is_strong or is_long_enough:
                        yield candidate
                        buffer = buffer[end_idx:]
                    else:
                        # Si es débil y corto, NO cortamos todavía.
                        # Esperamos a que llegue más texto que pueda:
                        # 1. Convertirlo en algo más largo.
                        # 2. Encontrar un delimitador fuerte más adelante.
                        break
        
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
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Verificar si hay señal
            amplitude = np.max(np.abs(audio_np))
            if amplitude < 0.01:
                logger.debug("Silencio detectado, ignorando.")
                continue

            logger.info("🎤 Procesando voz de usuario...")
            t_start_pipeline = time.time()
            
            # STT
            t0 = time.time()
            user_text = await run_stt(audio_np)
            t_stt = time.time() - t0
            
            if not user_text:
                logger.warning("STT no detectó texto.")
                continue

            # 3. PIPELINE LLM -> TTS (Streaming)
            logger.info(f"📝 Usuario: '{user_text}' (STT: {t_stt:.2f}s)")
            
            # Métricas acumuladas para el reporte final de esta interacción
            interaction_metrics = {
                "stt_time": t_stt,
                "sentences": []
            }

            t_llm_start = time.time()
            first_audio_sent = False
            t_first_byte = 0
            
            async for sentence in stream_sentences(user_text):
                t_sent_gen = time.time()
                
                # Generar audio
                t_tts_start = time.time()
                audio_bytes = await run_tts(sentence)
                t_tts_dur = time.time() - t_tts_start
                
                if audio_bytes:
                    await websocket.send_bytes(audio_bytes)
                    
                    if not first_audio_sent:
                        t_first_byte = time.time() - t_start_pipeline
                        first_audio_sent = True
                    
                    # Registrar métricas de esta frase
                    sent_metric = {
                        "text": sentence[:30] + "..." if len(sentence) > 30 else sentence,
                        "chars": len(sentence),
                        "tts_time": t_tts_dur,
                        "audio_size": len(audio_bytes)
                    }
                    interaction_metrics["sentences"].append(sent_metric)
                    logger.info(f"  📤 Enviado: '{sentence[:20]}...' | TTS: {t_tts_dur:.2f}s")
                else:
                    logger.error(f"  ❌ Falló TTS para: '{sentence[:20]}...'")

            # Reporte Final Visual
            total_time = time.time() - t_start_pipeline
            
            report = f"""
🎯 INTERACTION REPORT
   Total Latency (End-to-End): {total_time:.2f}s
   Time to First Audio (TTFA): {t_first_byte:.2f}s {'⚡ FAST' if t_first_byte < 1.5 else '🐢 SLOW'}
   
   [STT] Whisper (CPU): {interaction_metrics['stt_time']:.2f}s
   
   [TTS Pipeline Breakdown]
   {'#':<3} | {'Text Segment':<30} | {'Chars':<5} | {'TTS Time':<8} | {'Speed (ms/char)':<15}
   {'-'*75}"""
            
            for i, m in enumerate(interaction_metrics["sentences"]):
                speed = (m['tts_time'] * 1000) / m['chars'] if m['chars'] > 0 else 0
                report += f"\n   {i+1:<3} | {m['text']:<30} | {m['chars']:<5} | {m['tts_time']:.2f}s   | {speed:.0f} ms/char"
            
            metrics_logger.info(report)

    except WebSocketDisconnect:
        logger.info("🔌 Cliente desconectado.")
    except Exception as e:
        logger.error(f"🔥 Error Crítico Pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
