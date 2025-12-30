# 🔍 Debug: "No me escucha el modelo"

## Diagnóstico Paso a Paso

### 1. Verificar la Consola del Navegador

**Abre Developer Tools:**
- Chrome/Edge: `F12` o `Ctrl+Shift+I`
- Safari: `Cmd+Option+I`

**Busca en la pestaña "Console":**

#### ✅ Mensajes que DEBERÍAS ver:

```javascript
🚀 Usando Web Speech API (Modo Rápido)
// O
🐢 Web Speech no soportado. Usando VAD + Audio Raw.
```

#### ❌ Errores comunes:

**Error 1: HTTPS requerido**
```
NotAllowedError: The request is not allowed by the user agent
```
**Solución:** Web Speech API requiere HTTPS o localhost. En RunPOD, verifica que accedas via HTTPS.

**Error 2: Permisos de micrófono**
```
NotAllowedError: Permission denied
```
**Solución:** El navegador bloqueó el micrófono. Haz clic en el ícono del candado/cámara en la barra de direcciones.

**Error 3: No hay micrófono**
```
NotFoundError: Requested device not found
```
**Solución:** Tu dispositivo no tiene micrófono o no está conectado.

**Error 4: WebSocket no conecta**
```
WebSocket connection to 'ws://...' failed
```
**Solución:** El servidor no está corriendo o el puerto está bloqueado.

---

### 2. Verificar URL de Acceso

**En RunPOD, debes acceder via HTTPS:**

❌ **INCORRECTO:**
```
http://<pod-id>.runpod.io:8000
```

✅ **CORRECTO:**
```
https://<pod-id>.runpod.io:8000
```

**Por qué:** Web Speech API y getUserMedia (micrófono) solo funcionan en contextos seguros (HTTPS o localhost).

---

### 3. Verificar Estado del Servidor

**En el terminal del POD:**

```bash
# Ver logs del servidor
tail -f /var/log/...  # O donde estén tus logs

# O si corriste manualmente:
python server.py
# Deberías ver:
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Verificar que el WebSocket funciona:**

En la consola del navegador:
```javascript
// Debería mostrar:
WebSocket {url: 'wss://...', readyState: 1, ...}
```

Si `readyState === 1`, el WebSocket está conectado ✅

---

### 4. Test Manual del Micrófono

**Pega esto en la consola del navegador:**

```javascript
// Test 1: Verificar Web Speech API
if (window.SpeechRecognition || window.webkitSpeechRecognition) {
    console.log("✅ Web Speech API disponible");
} else {
    console.log("❌ Web Speech API NO disponible");
}

// Test 2: Verificar permisos de micrófono
navigator.mediaDevices.getUserMedia({ audio: true })
    .then(() => console.log("✅ Micrófono accesible"))
    .catch(err => console.error("❌ Error micrófono:", err));

// Test 3: Test de grabación simple
navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
        console.log("✅ Stream obtenido:", stream);
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();
        console.log("✅ Grabando... (habla algo)");

        setTimeout(() => {
            mediaRecorder.stop();
            console.log("✅ Grabación detenida. Si viste esto, el micrófono funciona.");
        }, 3000);
    })
    .catch(err => console.error("❌ Error:", err));
```

---

### 5. Verificar Flujo Completo

**Logs esperados en la consola del navegador:**

```
1. (Al cargar) Cargando inteligencia...
2. (Librerías cargadas) Listo para conectar.
3. (Conectado WebSocket) Conectado. Pulsa el botón.
4. (Click en "EMPEZAR CLASE")
   - Si Web Speech: "🚀 Usando Web Speech API (Modo Rápido)"
   - Si VAD: "🐢 Web Speech no soportado. Usando VAD + Audio Raw."
5. (Al hablar) "🗣️ Detectado: [tu texto]"
6. (Procesando) Indicator cambia a azul
7. (Sarah responde) Indicator cambia a verde, audio se reproduce
```

**Logs esperados en el servidor (terminal del POD):**

```
📨 Recibido TEXTO directo: 'Hello'
📝 Usuario: 'Hello' (STT: 0.00s)
  📤 Enviado: 'Hi! How can I help...' | TTS: 1.23s
```

---

### 6. Soluciones Comunes

#### Problema: "No pasa nada al hablar"

**Diagnóstico:**
1. Abre consola del navegador
2. Habla algo
3. ¿Ves "🗣️ Detectado: ..." ?
   - **SÍ:** El problema está en el backend (WebSocket, STT o TTS)
   - **NO:** El problema está en el frontend (micrófono o permisos)

**Solución si NO ves "🗣️ Detectado":**

```javascript
// En la consola, forzar test de reconocimiento:
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'en-US';
recognition.start();
recognition.onresult = (e) => console.log("DETECTADO:", e.results[0][0].transcript);
recognition.onerror = (e) => console.error("ERROR:", e);

// Ahora habla. Deberías ver el texto en la consola.
```

#### Problema: "Veo el texto detectado pero no responde"

**Significa:** Frontend funciona, backend NO.

**Verificar:**

```bash
# En el terminal del POD
python server.py

# Deberías ver logs cuando hablas
# Si NO ves logs, el WebSocket no está enviando datos
```

**Test de WebSocket desde consola:**

```javascript
// En la consola del navegador
ws.send(JSON.stringify({text: "test"}));
// Deberías ver logs en el servidor inmediatamente
```

#### Problema: "Sale 'Procesando...' pero nunca responde"

**Posibles causas:**

1. **STT/TTS falló:**
   - Ver logs del servidor para errores de CUDA, modelo no encontrado, etc.

2. **Groq API Key inválida:**
   ```bash
   # Verificar .env
   cat .env | grep GROQ_API_KEY
   ```

3. **TTS no configurado:**
   ```bash
   # Verificar .env
   cat .env | grep TTS_ENGINE
   # Debe ser: vibevoice, kokoro o f5-tts
   ```

---

### 7. Script de Debug Automático

**Crea este archivo para test completo:**

```bash
# test_audio_flow.sh
#!/bin/bash

echo "🔍 DIAGNÓSTICO COMPLETO"
echo ""

echo "1. Verificando GPU..."
nvidia-smi --query-gpu=name --format=csv,noheader
echo ""

echo "2. Verificando CUDA en PyTorch..."
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
echo ""

echo "3. Verificando VibeVoice instalado..."
python -c "from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference; print('✅ VibeVoice OK')"
echo ""

echo "4. Verificando .env..."
if [ -f .env ]; then
    echo "✅ .env existe"
    echo "TTS_ENGINE=$(grep TTS_ENGINE .env | cut -d '=' -f2)"
    echo "VIBEVOICE_VOICE=$(grep VIBEVOICE_VOICE .env | cut -d '=' -f2)"

    if grep -q "your_groq_api_key_here" .env; then
        echo "❌ GROQ_API_KEY no configurado!"
    else
        echo "✅ GROQ_API_KEY configurado"
    fi
else
    echo "❌ .env NO existe"
fi
echo ""

echo "5. Test rápido de síntesis..."
python -c "
from app.services.tts_vibevoice import generate_audio_vibevoice
import time
start = time.time()
wav = generate_audio_vibevoice('Test')
elapsed = time.time() - start
if wav:
    print(f'✅ TTS funciona ({elapsed:.2f}s, {len(wav)} bytes)')
else:
    print('❌ TTS falló')
"
echo ""

echo "6. Verificando servidor corriendo..."
if pgrep -f "server.py" > /dev/null; then
    echo "✅ Servidor corriendo"
else
    echo "❌ Servidor NO corriendo"
fi
echo ""

echo "RESUMEN:"
echo "Si todos los ✅ están OK, el problema está en el frontend (navegador/permisos)"
echo "Si hay ❌, arregla esos primero"
```

**Ejecutar:**
```bash
chmod +x test_audio_flow.sh
./test_audio_flow.sh
```

---

### 8. Modo Debug en el Frontend

**Añade esto al HTML temporalmente (después de la línea 254):**

```javascript
// DEBUG MODE - añadir después de la línea 254
console.log("DEBUG: Iniciando sistema...");

// Override de ws.onmessage para ver qué llega
const originalOnMessage = ws.onmessage;
ws.onmessage = (event) => {
    console.log("🔵 RECIBIDO del servidor:", event.data.byteLength || event.data.length, "bytes");
    originalOnMessage(event);
};

// Override de ws.send para ver qué se envía
const originalSend = ws.send.bind(ws);
ws.send = (data) => {
    if (typeof data === 'string') {
        console.log("🟢 ENVIANDO al servidor (texto):", data);
    } else {
        console.log("🟢 ENVIANDO al servidor (audio):", data.byteLength, "bytes");
    }
    originalSend(data);
};

console.log("DEBUG MODE ACTIVADO - Revisa logs arriba 🔍");
```

---

## Checklist de Solución Rápida

- [ ] ¿Accedes via HTTPS (no HTTP)?
- [ ] ¿La consola muestra "Conectado. Pulsa el botón."?
- [ ] ¿Al hacer clic sale permiso de micrófono y lo aceptaste?
- [ ] ¿La consola muestra "🚀 Usando Web Speech API" o "🐢 ... VAD"?
- [ ] ¿Al hablar, la consola muestra "🗣️ Detectado: ..."?
- [ ] ¿El servidor muestra logs cuando hablas?
- [ ] ¿El .env tiene GROQ_API_KEY configurado (no "your_...here")?
- [ ] ¿El .env tiene TTS_ENGINE=vibevoice?
- [ ] ¿El test_vibevoice.py pasó sin errores?

---

## Solución Rápida Más Común

**90% de las veces es uno de estos:**

1. **Accediendo via HTTP en lugar de HTTPS**
   - Solución: Usa `https://...` en la URL

2. **Permisos de micrófono bloqueados**
   - Solución: Click en el ícono del micrófono/cámara en la barra de direcciones, permitir

3. **GROQ_API_KEY no configurado**
   - Solución: `nano .env` y poner tu key real

4. **Servidor no corriendo**
   - Solución: `python server.py` en el POD

---

¿Cuál de estos problemas tienes? Dime qué ves en la consola del navegador y te ayudo a arreglarlo.
