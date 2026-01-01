# 🔧 Guía de Debugging para RunPod

## Problema Identificado
El sistema no procesaba la voz porque Web Speech API falla con error `network` en entornos con proxy (como RunPod).

## Solución Implementada
El sistema ahora usa **VAD + Whisper (Modo Local)** por defecto, que procesa todo localmente sin depender de servicios externos de Google.

## Pasos de Diagnóstico

### 1. Verificar que el servidor está corriendo
```bash
# En el pod de RunPod, deberías ver:
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### 2. Acceder a la página de pruebas
Abre en tu navegador (usando el proxy de RunPod):
```
https://[TU-POD-ID]-8000.proxy.runpod.net/test.html
```

### 3. Ejecutar tests en orden
1. **Test 1: WebSocket Connection**
   - Debe mostrar: `✅ WebSocket CONECTADO`
   - Si falla, revisa los logs del servidor

2. **Test 2: Microphone**
   - Debe pedir permisos de micrófono
   - Debe mostrar: `✅ Permisos de micrófono concedidos`
   - Si falla: El navegador bloqueó el micrófono (solo funciona con HTTPS)

3. **Test 3: Web Speech API**
   - Debe iniciar el reconocimiento
   - **HABLA EN INGLÉS** (configurado para 'en-US')
   - Debe mostrar el texto transcrito
   - Si falla: Chrome/Edge requerido (Firefox no soporta Web Speech API completamente)

4. **Test 4: Send Test Message**
   - Envía un mensaje de prueba al servidor
   - Revisa los logs del servidor para ver si lo recibe

### 4. Revisar logs del navegador
Abre las DevTools (F12) y ve a la pestaña **Console**. Deberías ver:

```
🔌 Conectando WebSocket a: wss://...
✅ WebSocket ABIERTO
🎤 Solicitando permisos de micrófono...
✅ Permisos de micrófono concedidos
🚀 Usando Web Speech API (Modo Rápido)
🎤 Web Speech API iniciada
```

Si ves errores, anótalos.

### 5. Revisar logs del servidor (RunPod)
En el terminal del pod, deberías ver:

```
INFO:     100.64.1.35:37710 - "WebSocket /ws" [accepted]
INFO:     connection open
📥 Received message type: websocket.receive
📨 Received text: 'hello world'
```

## Problemas Comunes

### ❌ Web Speech API falla con error "network"
**Causa**: Web Speech API de Google no funciona en entornos con proxy (RunPod, etc.)
**Solución**:
- ✅ **YA RESUELTO**: El sistema ahora usa VAD + Whisper local por defecto
- Si quieres usar Web Speech API en desarrollo local, cambia `forceVADMode = false` en index.html línea 167

### ❌ WebSocket no conecta
**Causa**: URL incorrecta o servidor no está escuchando
**Solución**:
- Verifica que estás usando la URL correcta: `https://[POD-ID]-8000.proxy.runpod.net`
- Verifica que el puerto 8000 está expuesto en RunPod
- Reinicia el servidor: `./start_cpu.sh`

### ❌ Reconocimiento se detiene inmediatamente
**Causa**: Web Speech API configurado con `continuous: false`
**Solución**: Es normal. El sistema reinicia automáticamente después de cada frase.

### ❌ No se envía el mensaje al backend
**Causa**: WebSocket no está en estado OPEN cuando se intenta enviar
**Solución**:
- Espera a que el indicador cambie a "Conectado. Pulsa el botón."
- Revisa la consola del navegador para ver el estado del WebSocket

### ❌ Backend no responde
**Causa**: Error en el pipeline (STT, LLM, o TTS)
**Solución**:
- Revisa los logs del servidor completos
- Verifica que tienes API keys configuradas (Groq, etc.)
- Verifica que los modelos están descargados

## Logs Esperados (Flujo Completo)

### Navegador (Console) - MODO VAD (NUEVO)
```
🔌 Conectando WebSocket a: wss://...
✅ WebSocket ABIERTO
🎤 Solicitando permisos de micrófono...
✅ Permisos de micrófono concedidos
🔊 Usando VAD + Whisper (Modo Local)
⚙️ Configurando ONNX Runtime...
🎤 Obteniendo stream de audio...
✅ Stream de audio obtenido
🤖 Inicializando VAD...
🚀 Iniciando VAD...
✅ VAD iniciado correctamente
🗣️ VAD: Detectando voz...
🎙️ VAD: Voz finalizada (48000 samples)
📊 WebSocket estado: 1 (1 = OPEN)
📦 Audio convertido a Int16: 48000 samples
📤 Enviando 96000 bytes de audio
✅ Audio enviado al servidor
```

### Servidor (Terminal) - MODO VAD
```
INFO:     100.64.1.35:37710 - "WebSocket /ws" [accepted]
INFO:     connection open
📥 Received message type: websocket.receive
🎤 Processing audio...
📝 Transcribed: 'hello how are you' (0.85s)
🤖 LLM: 'I'm doing great, thanks for asking...'
🔊 TTS: 35 chars in 0.42s
⚡ Time to first audio: 1.53s
✅ Total latency: 2.65s
```

## Comandos Útiles

### Ver logs en tiempo real
```bash
# En el pod
tail -f /workspace/testing-mvp/logs/*.log
```

### Reiniciar servidor
```bash
# Detener (Ctrl+C)
# Iniciar
./start_cpu.sh
```

### Verificar puerto
```bash
netstat -tlnp | grep 8000
```

## Información de Contacto
Si sigues teniendo problemas, guarda:
1. Los logs completos del navegador (Console)
2. Los logs completos del servidor (Terminal)
3. La URL que estás usando
4. El navegador y versión
