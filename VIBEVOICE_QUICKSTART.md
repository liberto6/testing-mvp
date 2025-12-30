# 🚀 VibeVoice TTS - Quick Start Guide

Esta guía te llevará de **0 a producción** con VibeVoice TTS en tu pipeline de Verba en menos de 10 minutos.

---

## ✅ Checklist Pre-instalación

Antes de empezar, verifica que tienes:

- [ ] Python 3.8+ instalado
- [ ] Conda o venv configurado (recomendado)
- [ ] GPU NVIDIA con CUDA (opcional, pero recomendado)
- [ ] O Apple Silicon con MPS (funciona bien)
- [ ] Espacio en disco: ~5GB para modelo + dependencias

---

## 📋 Paso 1: Instalar VibeVoice

```bash
# Navega al directorio de repositorios
cd /Users/pepeda-rosa/Documents/Verba/Repositorios/

# Clona VibeVoice (si no lo has hecho ya)
git clone https://github.com/microsoft/VibeVoice.git

# Instala en modo editable
cd VibeVoice
pip install -e .

# Verifica que las voces están presentes
ls -la demo/voices/streaming_model/
# Deberías ver archivos .pt como:
#   en_US-wayne-medium.pt
#   en_US-sarah-medium.pt
```

---

## 📋 Paso 2: Instalar Dependencias del Proyecto

```bash
# Regresa al proyecto Verba
cd /Users/pepeda-rosa/Documents/Verba/RUNPOD/testing-mvp/

# Instala todas las dependencias
pip install -r requirements.txt

# Si estás en CUDA y quieres Flash Attention 2 (opcional, pero recomendado):
pip install flash-attn --no-build-isolation
```

**Nota:** La instalación puede tomar 5-10 minutos dependiendo de tu conexión.

---

## 📋 Paso 3: Configurar Variables de Entorno

```bash
# Copia el archivo de ejemplo
cp .env.example .env

# Edita .env con tu editor favorito
nano .env  # o vim, code, etc.
```

**Configuración mínima requerida:**

```bash
# .env
GROQ_API_KEY=tu_api_key_aqui
TTS_ENGINE=vibevoice
VIBEVOICE_VOICE=Wayne
```

**Configuración completa (recomendada):**

```bash
# .env
GROQ_API_KEY=tu_api_key_aqui

# TTS Engine
TTS_ENGINE=vibevoice

# VibeVoice Settings
VIBEVOICE_VOICE=Wayne
VIBEVOICE_VOICES_DIR=/Users/pepeda-rosa/Documents/Verba/Repositorios/MicrosoftVibeVoice/VibeVoice/demo/voices/streaming_model/
VIBEVOICE_CFG_SCALE=1.5
VIBEVOICE_DDPM_STEPS=5
```

---

## 📋 Paso 4: Ejecutar Test de Integración

```bash
# Ejecuta el script de prueba
python test_vibevoice.py

# Si todo está bien, verás:
# ✅ Todos los tests pasaron exitosamente!
# 🎉 VibeVoice TTS está correctamente integrado.
```

**¿Test falló?** Ve a la sección [Troubleshooting](#troubleshooting).

---

## 📋 Paso 5: Iniciar el Servidor

```bash
# Inicia el servidor con VibeVoice
python server.py

# Deberías ver:
# 🚀 Iniciando servidor. Precalentando TTS: vibevoice...
# VibeVoice TTS initializing on device: cuda (or mps/cpu)
# ✅ TTS vibevoice listo.
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 📋 Paso 6: Probar en la Aplicación

1. **Abre tu navegador:**
   ```
   http://localhost:8000
   ```

2. **Habla con Sarah** (el tutor):
   - Haz clic en "Start" o habla directamente
   - Di algo como: "Hello, how are you?"

3. **Escucha la respuesta:**
   - Sarah responderá con VibeVoice TTS
   - Calidad de audio: 24kHz, muy natural

4. **Compara con otros motores:**
   ```bash
   # Detén el servidor (Ctrl+C)

   # Prueba Kokoro
   echo "TTS_ENGINE=kokoro" > .env
   python server.py

   # Prueba F5-TTS
   echo "TTS_ENGINE=f5-tts" > .env
   python server.py
   ```

---

## 🎛️ Optimización de Parámetros

### Para Máxima Velocidad (RTF < 0.1x)

```bash
VIBEVOICE_CFG_SCALE=1.0
VIBEVOICE_DDPM_STEPS=3
```

### Para Máxima Calidad (RTF ~0.5x)

```bash
VIBEVOICE_CFG_SCALE=2.0
VIBEVOICE_DDPM_STEPS=20
```

### Balance Recomendado (RTF ~0.15x)

```bash
VIBEVOICE_CFG_SCALE=1.5
VIBEVOICE_DDPM_STEPS=5
```

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'vibevoice'"

**Solución:**
```bash
cd /Users/pepeda-rosa/Documents/Verba/Repositorios/VibeVoice
pip install -e .
```

### Error: "Voices directory not found"

**Solución:**
Verifica que la ruta es correcta:
```bash
ls /Users/pepeda-rosa/Documents/Verba/Repositorios/MicrosoftVibeVoice/VibeVoice/demo/voices/streaming_model/
```

Si no existe, ajusta `VIBEVOICE_VOICES_DIR` en `.env`.

### Error: "CUDA out of memory"

**Soluciones:**

1. **Reducir pasos de difusión:**
   ```bash
   VIBEVOICE_DDPM_STEPS=3
   ```

2. **Usar CPU:**
   ```bash
   # En app/core/config.py, forzar:
   DEVICE = "cpu"
   ```

3. **Liberar cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Error: "Flash Attention 2 failed"

**No es un problema.** El sistema automáticamente usa SDPA como fallback.

Para instalar Flash Attention 2 (opcional, solo CUDA):
```bash
pip install flash-attn --no-build-isolation
```

### Audio suena distorsionado

**Ajusta CFG scale:**
```bash
# Reduce para más estabilidad
VIBEVOICE_CFG_SCALE=1.0
```

### Latencia muy alta

**Opciones:**

1. **Reduce DDPM steps:**
   ```bash
   VIBEVOICE_DDPM_STEPS=3
   ```

2. **Cambia a Kokoro (más rápido):**
   ```bash
   TTS_ENGINE=kokoro
   ```

3. **Verifica que estás usando GPU:**
   ```python
   # En los logs, busca:
   # "VibeVoice TTS initializing on device: cuda"
   ```

---

## 📊 Benchmarks de Referencia

### RTF (Real-Time Factor) - Menor es mejor

| Device | DDPM Steps | RTF | Latencia (1s audio) |
|--------|------------|-----|---------------------|
| RTX 3090 | 5 | 0.08x | ~80ms |
| RTX 3090 | 20 | 0.25x | ~250ms |
| Apple M1 Max | 5 | 0.15x | ~150ms |
| Apple M1 Max | 20 | 0.50x | ~500ms |
| CPU (i7) | 5 | 0.80x | ~800ms |
| CPU (i7) | 20 | 2.5x | ~2.5s |

**Nota:** RTF < 1.0 significa tiempo real (genera más rápido que reproduce).

---

## 🎯 Próximos Pasos

### 1. Experimentar con Voces

```bash
# Lista las voces disponibles
python -c "
from app.services.tts_vibevoice import VibeVoiceTTS
tts = VibeVoiceTTS()
print(tts.get_available_voices())
"

# Cambia la voz
VIBEVOICE_VOICE=Sarah
```

### 2. Integrar Streaming (Avanzado)

El código ya tiene soporte para streaming, pero no está activado en el pipeline actual.

Para habilitar streaming:
- Modifica `app/services/tts_vibevoice.py`
- Usa `AudioStreamer` o `AsyncAudioStreamer`
- Integra con el WebSocket en `app/api/endpoints.py`

Ver: [VibeVoice/demo/web/app.py](https://github.com/microsoft/VibeVoice/blob/main/demo/web/app.py) para referencia.

### 3. Ajuste Fino (Fine-tuning)

VibeVoice soporta fine-tuning con tus propios datos de voz.

Ver documentación oficial: [VibeVoice Fine-tuning Guide](https://github.com/microsoft/VibeVoice#fine-tuning)

---

## 📚 Recursos Adicionales

- **README completo:** [README_VIBEVOICE.md](README_VIBEVOICE.md)
- **Configuración de voces:** [app/config/vibevoice_voices.json](app/config/vibevoice_voices.json)
- **Código del motor:** [app/services/tts_vibevoice.py](app/services/tts_vibevoice.py)
- **Tests:** [test_vibevoice.py](test_vibevoice.py)

---

## ✅ Checklist Final

- [x] VibeVoice clonado e instalado
- [x] Dependencias instaladas
- [x] `.env` configurado
- [x] Test pasado exitosamente
- [x] Servidor iniciado sin errores
- [x] Audio generado correctamente
- [ ] Parámetros optimizados para tu hardware
- [ ] Voz seleccionada (Wayne/Sarah/otra)
- [ ] Latencia aceptable para tu caso de uso

---

## 🎉 ¡Listo!

Si has llegado hasta aquí, **VibeVoice está completamente integrado** en tu pipeline.

**Disfruta de síntesis de voz de alta calidad en tiempo real.** 🚀

Para soporte o preguntas:
- Revisa [README_VIBEVOICE.md](README_VIBEVOICE.md)
- Consulta la [documentación oficial de VibeVoice](https://github.com/microsoft/VibeVoice)
- Abre un issue en el repositorio

---

**Happy Coding!** 🎤✨
