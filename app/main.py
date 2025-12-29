import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import STATIC_DIR
from app.core.logging import setup_logging
from app.api.endpoints import router as api_router

# Configurar Logging
setup_logging()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas API (WebSocket)
app.include_router(api_router)

# Rutas para archivos estáticos con tipos MIME específicos
# Aunque StaticFiles sirve archivos, para WASM a veces es delicado.
# Sin embargo, FastAPI maneja bien los tipos si mimetypes los conoce.
# Mantendremos las rutas manuales para asegurar compatibilidad con lo que había, 
# pero sirviendo desde STATIC_DIR.

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/ort-wasm-simd.wasm")
async def get_wasm_simd():
    return FileResponse(os.path.join(STATIC_DIR, "ort-wasm-simd.wasm"), media_type="application/wasm")

@app.get("/ort-wasm.wasm")
async def get_wasm_basic():
    return FileResponse(os.path.join(STATIC_DIR, "ort-wasm.wasm"), media_type="application/wasm")

@app.get("/silero_vad.onnx")
async def get_model():
    return FileResponse(os.path.join(STATIC_DIR, "silero_vad.onnx"), media_type="application/octet-stream")

@app.get("/vad.js")
async def get_vad():
    return FileResponse(os.path.join(STATIC_DIR, "vad.js"), media_type="application/javascript")

@app.get("/ort.js")
async def get_ort():
    return FileResponse(os.path.join(STATIC_DIR, "ort.js"), media_type="application/javascript")

@app.get("/vad.worklet.v2.js")
async def get_worklet_v2():
    return FileResponse(os.path.join(STATIC_DIR, "vad.worklet.v2.js"), media_type="application/javascript")

# Fallback para otros estáticos si fuera necesario (ej: iconos)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
