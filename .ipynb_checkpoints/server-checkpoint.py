import os
import torch
import numpy as np
import requests
import uvicorn
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from f5_tts.api import F5TTS
from fastapi.middleware.cors import CORSMiddleware  # <--- ESTA ES LA LÍNEA QUE FALTA
import scipy.io.wavfile as wavfile # Asegúrate de tener este import arriba
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 1. INICIALIZACIÓN ROBUSTA
print("Cargando Whisper (STT) en CPU...")
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")

print("Cargando F5-TTS...")
# Aseguramos que F5-TTS use la GPU si está disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = F5TTS(device=device)

# Rutas de referencia (Ajusta estas rutas a tus archivos reales)
REF_AUDIO = "/workspace/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav"
REF_TEXT = "Some call me nature, others call me mother nature"

# Ruta base donde están tus archivos
BASE_DIR = "/workspace"
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
    return FileResponse(f"{BASE_DIR}/index.html")
    
@app.get("/ort-wasm-simd.wasm")
async def get_wasm_simd():
    return FileResponse("/workspace/ort-wasm-simd.wasm", media_type="application/wasm")

@app.get("/ort-wasm.wasm")
async def get_wasm_basic():
    return FileResponse("/workspace/ort-wasm.wasm", media_type="application/wasm")
# Servir el modelo ONNX con el MIME type correcto
@app.get("/silero_vad.onnx")
async def get_model():
    return FileResponse(f"{BASE_DIR}/silero_vad.onnx", media_type="application/octet-stream")

# Servir los archivos JS y Worklets
@app.get("/vad.js")
async def get_vad():
    return FileResponse(f"{BASE_DIR}/vad.js", media_type="application/javascript")

@app.get("/ort.js")
async def get_ort():
    return FileResponse(f"{BASE_DIR}/ort.js", media_type="application/javascript")

@app.get("/vad.worklet.v2.js")
async def get_worklet_v2():
    return FileResponse(f"{BASE_DIR}/vad.worklet.v2.js", media_type="application/javascript")


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
            segments, _ = stt_model.transcribe(audio_np, language="en")
            user_text = " ".join([s.text for s in segments]).strip()
            
            if not user_text:
                print("DEBUG: Whisper no detectó palabras en este audio.")
                continue
                
            print(f"🎤 USUARIO DIJO: {user_text}")

            # C. LLM: OLLAMA (Llama 3.1)
            ai_text = ""
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.1",
                        "prompt": f"Eres un profesor de ingles. Responde en ingles y en frases cortas, corrigiendo al alumno y enseñandole: {user_text}",
                        "stream": False
                    },
                    timeout=5
                )
                ai_text = response.json().get('response', '')
                print(f"🤖 Llama: {ai_text}")
            except Exception as e:
                print(f"⚠️ Error Ollama: {e}")
                ai_text = "Can u repeat it pls"

            # D. TTS: F5-TTS (Generación de audio)
            output_file = "/workspace/output.wav"
            print("🔊 Generando voz...")
            
            try:
                # 1. Realizar la inferencia (devuelve el audio y la frecuencia)
                # Eliminamos model_name y output_file de los argumentos
                audio, sr, spectr = tts.infer(
                    gen_text=ai_text,
                    ref_file=REF_AUDIO,
                    ref_text=REF_TEXT
                )

                # 2. Guardar el archivo manualmente usando scipy
                wavfile.write(output_file, sr, audio)

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