import os
import torch
import numpy as np
import uvicorn
import time
from dotenv import load_dotenv

load_dotenv() # Cargar variables de entorno desde .env

from groq import Groq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from f5_tts.api import F5TTS
from fastapi.middleware.cors import CORSMiddleware
import scipy.io.wavfile as wavfile

# Inicializar cliente Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️ ADVERTENCIA: No se encontró la variable de entorno GROQ_API_KEY.")
    print("   Por favor, configura tu API Key en un archivo .env o en las variables del sistema.")
    # Fallback para pruebas locales (Opcional: eliminar antes de producción si es crítico)
    # GROQ_API_KEY = "tu_api_key_aqui" 

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 1. INICIALIZACIÓN ROBUSTA
print("Configurando dispositivo...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo seleccionado: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print(f"Cargando Whisper (STT) en {device}...")
# Usamos float16 si estamos en CUDA para mayor velocidad, int8 en CPU
compute_type = "float16" if device == "cuda" else "int8"
stt_model = WhisperModel("small", device=device, compute_type=compute_type)

print(f"Cargando F5-TTS en {device}...")
tts = F5TTS(device=device)

# Rutas dinámicas
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas de referencia (Ajusta estas rutas a tus archivos reales)
# Usamos el audio de ejemplo que viene dentro del repo clonado F5-TTS
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
@app.get("/")
async def get_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))
    
@app.get("/ort-wasm-simd.wasm")
async def get_wasm_simd():
    return FileResponse(os.path.join(BASE_DIR, "ort-wasm-simd.wasm"), media_type="application/wasm")

@app.get("/ort-wasm.wasm")
async def get_wasm_basic():
    return FileResponse(os.path.join(BASE_DIR, "ort-wasm.wasm"), media_type="application/wasm")
# Servir el modelo ONNX con el MIME type correcto
@app.get("/silero_vad.onnx")
async def get_model():
    return FileResponse(os.path.join(BASE_DIR, "silero_vad.onnx"), media_type="application/octet-stream")

# Servir los archivos JS y Worklets
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
    print("🚀 Pipeline conectado y listo.")
    
    try:
        while True:
            # A. RECIBIR AUDIO
            data = await websocket.receive_bytes()
            
            # DIAGNÓSTICO 1: ¿Llegan bytes?
            print(f"DEBUG: Bytes recibidos del VAD: {len(data)}")

            if len(data) == 0:
                print("DEBUG: Buffer vacío, saltando...")
                continue
            
            # Convertir a numpy
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # DIAGNÓSTICO 2: ¿Hay sonido o es silencio absoluto?
            amplitude = np.max(np.abs(audio_np))
            print(f"DEBUG: Amplitud máxima del audio: {amplitude:.4f}")

            if amplitude < 0.01:
                print("DEBUG: Audio demasiado bajo (posible silencio), Whisper podría ignorarlo.")

            # B. PASAR A WHISPER
            print("DEBUG: Procesando con Whisper...")
            t0 = time.time()
            segments, _ = stt_model.transcribe(audio_np, language="en")
            user_text = " ".join([s.text for s in segments]).strip()
            print(f"⏱️ Whisper tardó: {time.time() - t0:.2f}s")
            
            if not user_text:
                print("DEBUG: Whisper no detectó palabras en este audio.")
                continue
                
            print(f"🎤 USUARIO DIJO: {user_text}")

            # C. LLM: Groq
            ai_text = ""
            try:
                t1 = time.time()
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text}
                    ],
                    model="llama-3.3-70b-versatile",
                )
                ai_text = chat_completion.choices[0].message.content
                print(f"🤖 Groq tardó: {time.time() - t1:.2f}s | Respuesta: {ai_text}")
            except Exception as e:
                print(f"⚠️ Error Groq: {e}")
                import traceback
                traceback.print_exc()
                ai_text = "Can u repeat it pls"

            # D. TTS: F5-TTS (Generación de audio)
            output_file = os.path.join(BASE_DIR, "output.wav")
            print("🔊 Generando voz...")
            t2 = time.time()
            
            try:
                # 1. Realizar la inferencia (devuelve el audio y la frecuencia)
                # Eliminamos model_name y output_file de los argumentos
                audio, sr, spectr = tts.infer(
                    gen_text=ai_text,
                    ref_file=REF_AUDIO,
                    ref_text=REF_TEXT
                )
                
                print(f"DEBUG: TTS Audio shape: {audio.shape}, Sample rate: {sr}")
                
                if len(audio) == 0:
                    print("❌ Error: TTS generó audio vacío.")
                    continue

                # Convertir a Int16 para máxima compatibilidad con navegadores
                audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)

                # 2. Guardar el archivo manualmente usando scipy
                wavfile.write(output_file, sr, audio)
                print(f"DEBUG: Archivo WAV escrito en {output_file}")

                # 3. Enviar el archivo generado
                if os.path.exists(output_file):
                    with open(output_file, "rb") as f:
                        audio_bytes = f.read()
                        await websocket.send_bytes(audio_bytes)
                    print(f"✅ Audio enviado ({len(audio_bytes)} bytes)")
                else:
                    print("❌ Error: No se pudo crear el archivo WAV.")

            except Exception as e:
                print(f"❌ Error en TTS: {e}")

    except WebSocketDisconnect:
        print("🔌 Cliente desconectado.")
    except Exception as e:
        print(f"🔥 Error Pipeline: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)