# 🔧 Guía de Debugging para RunPod

## Problema
El sistema no procesa la voz del usuario después de presionar el botón "EMPEZAR CLASE".

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

### ❌ Web Speech API no funciona
**Causa**: Navegador no compatible o no tienes HTTPS
**Solución**:
- Usa Chrome o Edge (no Firefox/Safari)
- Asegúrate de usar la URL del proxy de RunPod (con HTTPS)
- Concede permisos de micrófono cuando se soliciten

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

### Navegador (Console)
```
🔌 Conectando WebSocket a: wss://...
✅ WebSocket ABIERTO
🎤 Solicitando permisos de micrófono...
✅ Permisos de micrófono concedidos
🚀 Usando Web Speech API (Modo Rápido)
🎤 Web Speech API iniciada
🗣️ Detectado: hello how are you
📊 WebSocket estado: 1 (1 = OPEN)
📤 Enviando mensaje: {text: "hello how are you"}
✅ Mensaje enviado
```

### Servidor (Terminal)
```
INFO:     100.64.1.35:37710 - "WebSocket /ws" [accepted]
INFO:     connection open
📥 Received message type: websocket.receive
📨 Received text: 'hello how are you'
🤖 LLM: 'I'm doing great, thanks for asking...'
🔊 TTS: 35 chars in 0.42s
⚡ Time to first audio: 1.23s
✅ Total latency: 2.45s
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
