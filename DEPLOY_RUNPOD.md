# 🚀 Deploy en RunPod - Guía Rápida

## ✅ Cambios Aplicados

1. **Frontend (static/index.html)**
   - ✅ VAD mode forzado por defecto (`forceVADMode = true`)
   - ✅ Rutas absolutas para archivos WASM (`basePath`)
   - ✅ Logging detallado para debugging

2. **Backend (src/main.py)**
   - ✅ Parsing correcto de mensajes JSON del frontend
   - ✅ Logging mejorado para troubleshooting

3. **Archivos WASM**
   - ✅ 5 archivos `.mjs` descargados
   - ✅ 5 archivos `.wasm` ya existentes

4. **Configuración**
   - ✅ `configs/runpod_cpu_stt.yaml` - CPU STT, GPU TTS
   - ✅ `start_cpu.sh` - Script de inicio

## 📋 Pasos para Desplegar en RunPod

### 1. Sincronizar código en RunPod

Conéctate a tu pod y ejecuta:

```bash
cd /workspace/testing-mvp

# Opción A: Git pull
git pull origin feature/migracion_orquestador_pipecat

# Opción B: Si hay conflictos, hacer stash primero
git stash
git pull origin feature/migracion_orquestador_pipecat
```

### 2. Descargar archivos WASM y modelo VAD

**IMPORTANTE**: El archivo `silero_vad.onnx` puede estar corrupto en RunPod. Este script lo descarga de nuevo:

```bash
chmod +x download_wasm_files.sh
./download_wasm_files.sh
```

Deberías ver:
```
1️⃣ Verificando modelo silero_vad.onnx...
✅ silero_vad.onnx descargado

2️⃣ Descargando archivos WASM de ONNX Runtime...
✅ ort-wasm-simd-threaded.jsep.mjs descargado
✅ ort-wasm-simd-threaded.mjs descargado
✅ ort-wasm-simd.mjs descargado
✅ ort-wasm-threaded.mjs descargado
✅ ort-wasm.mjs descargado
```

### 3. Configurar API Key

```bash
export GROQ_API_KEY="tu-api-key"

# Para persistir:
echo 'export GROQ_API_KEY="tu-api-key"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Iniciar servidor

```bash
./start_cpu.sh
```

Deberías ver:
```
🚀 Starting Pipecat Voice Pipeline with CPU STT...

📝 STT (Speech-to-Text):
   Provider: whisper
   Model: base
   Device: cpu

INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 5. Probar en el navegador

Abre:
```
https://[TU-POD-ID]-8000.proxy.runpod.net/index.html
```

## 🧪 Verificación

### En la Consola del Navegador (F12 → Console)

Deberías ver:

```
🔌 Conectando WebSocket a: wss://...
✅ WebSocket ABIERTO
🎤 Solicitando permisos de micrófono...
✅ Permisos de micrófono concedidos
🔊 Usando VAD + Whisper (Modo Local)
⚙️ Configurando ONNX Runtime...
📁 WASM path configurado: https://.../static/
🎤 Obteniendo stream de audio...
✅ Stream de audio obtenido
🤖 Inicializando VAD...
✅ VAD inicializado correctamente
```

**IMPORTANTE**: Si ves algún error sobre archivos WASM, ejecuta de nuevo `./download_wasm_files.sh` en RunPod.

### En el Terminal del Servidor

Cuando hables, deberías ver:

```
📥 Received message type: websocket.receive
🎤 Processing audio...
📝 Transcribed: '...' (0.85s)
🤖 LLM: '...'
🔊 TTS: 35 chars in 0.42s
```

## 🐛 Troubleshooting

### Error: "Failed to resolve module specifier 'static/ort-wasm...'"

**Solución**: Ejecuta el script de descarga:
```bash
./download_wasm_files.sh
```

### Error: "Unable to load libcudnn..."

**Solución**: Asegúrate de usar el script correcto:
```bash
./start_cpu.sh  # ✅ Correcto
# NO: python run.py  ❌
```

### Error: "Can't create a session... protobuf parsing failed"

**Causa**: El archivo `silero_vad.onnx` está corrupto o incompleto.

**Solución**: Ejecuta el script de descarga que lo descarga de nuevo:
```bash
./download_wasm_files.sh
```

O manualmente:
```bash
cd static
rm silero_vad.onnx  # Eliminar el corrupto
curl -L -o silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
cd ..

# Verificar tamaño (debe ser ~290KB)
ls -lh static/silero_vad.onnx
```

### VAD no detecta voz

**Solución**: Ajusta sensibilidad en `static/index.html` línea ~283:

```javascript
positiveSpeechThreshold: 0.4,  // Más bajo = más sensible (default: 0.6)
```

## 📊 Diferencias con Web Speech API

| Aspecto | Web Speech API | VAD + Whisper |
|---------|----------------|---------------|
| **Funciona en RunPod** | ❌ Error network | ✅ Sí |
| **Velocidad** | Muy rápido (~0.3s) | Rápido (~1s) |
| **Privacidad** | Google Cloud | 100% local |
| **Requiere Internet** | Sí | No (solo LLM) |

## 📁 Archivos Importantes

```
testing-mvp/
├── start_cpu.sh                    # ⭐ Script de inicio (USA ESTE)
├── download_wasm_files.sh          # Script para descargar WASM
├── configs/runpod_cpu_stt.yaml    # Configuración CPU STT
├── static/
│   ├── index.html                  # Frontend con VAD
│   ├── test.html                   # Página de diagnóstico
│   ├── silero_vad.onnx             # Modelo VAD
│   ├── ort-wasm-*.mjs              # Módulos WASM (5 archivos)
│   └── ort-wasm-*.wasm             # Binarios WASM (5 archivos)
└── src/main.py                     # Backend con JSON parsing
```

## 🎯 Próximos Pasos

1. ✅ Probar que funciona en RunPod
2. Ajustar sensibilidad del VAD si es necesario
3. Considerar usar modelo Whisper `tiny` si quieres más velocidad

## 📞 Ayuda

- **Debugging detallado**: Ver [DEBUGGING_GUIDE.md](DEBUGGING_GUIDE.md)
- **Página de test**: Usa `/test.html` para diagnosticar
- **Logs del navegador**: F12 → Console
- **Logs del servidor**: En el terminal donde corriste `./start_cpu.sh`
