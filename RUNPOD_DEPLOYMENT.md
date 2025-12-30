# 🚀 VibeVoice TTS - RunPOD Deployment Guide

Esta guía específica te ayudará a desplegar VibeVoice TTS en tu POD de RunPOD de manera rápida y eficiente.

---

## 📋 Contenido

- [Requisitos Previos](#requisitos-previos)
- [Método 1: Setup Automatizado (Recomendado)](#método-1-setup-automatizado-recomendado)
- [Método 2: Setup Manual](#método-2-setup-manual)
- [Configuración](#configuración)
- [Verificación](#verificación)
- [Troubleshooting RunPOD](#troubleshooting-runpod)
- [Optimizaciones](#optimizaciones)
- [FAQ](#faq)

---

## ✅ Requisitos Previos

### Hardware Mínimo

| Componente | Requerimiento |
|------------|---------------|
| **GPU** | NVIDIA con CUDA (RTX 3090, A100, etc.) |
| **VRAM** | Mínimo 4GB, recomendado 8GB+ |
| **Storage** | 10GB libres (modelo + dependencias) |
| **RAM** | 8GB+ |

### Software en RunPOD

- ✅ Imagen base con CUDA 12.x
- ✅ Python 3.8+
- ✅ Git
- ✅ Persistent storage en `/workspace` (recomendado)

### Preparación

1. **Crear un POD en RunPOD:**
   - Selecciona una GPU (RTX 3090 o superior recomendada)
   - Habilita **Persistent Storage** (recomendado)
   - Expón el puerto **8000**
   - Inicia el POD

2. **Obtener tu API Key de Groq:**
   - Ve a [https://console.groq.com](https://console.groq.com)
   - Crea una cuenta si no tienes
   - Genera una API key
   - Guárdala de forma segura

---

## 🎯 Método 1: Setup Automatizado (Recomendado)

Este método usa un script que hace toda la instalación automáticamente.

### Paso 1: Acceder a tu POD

```bash
# Conectarse al POD via Web Terminal o SSH
# En RunPOD UI: Click en "Connect" -> "Start Web Terminal"
```

### Paso 2: Navegar al Directorio del Proyecto

```bash
cd /workspace/testing-mvp
```

Si no existe, clona tu proyecto primero:
```bash
cd /workspace
git clone <tu-repo-url> testing-mvp
cd testing-mvp
```

### Paso 3: Ejecutar el Script de Setup

```bash
# Dar permisos de ejecución
chmod +x runpod_setup.sh

# Ejecutar el script
./runpod_setup.sh
```

El script hará lo siguiente automáticamente:
1. ✅ Validar GPU y CUDA
2. ✅ Instalar dependencias del sistema (ffmpeg, git)
3. ✅ Instalar PyTorch con CUDA
4. ✅ Clonar VibeVoice en `/workspace/VibeVoice`
5. ✅ Instalar VibeVoice en modo editable
6. ✅ Instalar requirements del proyecto
7. ✅ Instalar Flash Attention 2 (si GPU compatible)
8. ✅ Crear archivo `.env` con configuración base
9. ✅ Ejecutar test de validación
10. ✅ Mostrar instrucciones de next steps

**Tiempo estimado:** 5-10 minutos

### Paso 4: Configurar API Key

```bash
nano .env
```

Cambia:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

Por:
```bash
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx
```

Guarda con `Ctrl+O`, `Enter`, `Ctrl+X`

### Paso 5: Iniciar el Servidor

```bash
python server.py
```

### Paso 6: Acceder a la Aplicación

Abre tu navegador:
```
http://<tu-pod-id>.runpod.io:8000
```

Ejemplo:
```
http://abc123xyz.runpod.io:8000
```

---

## 🔧 Método 2: Setup Manual

Si prefieres controlar cada paso:

### 1. Validar GPU

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Deberías ver tu GPU y `True`.

### 2. Instalar Dependencias del Sistema

```bash
apt-get update
apt-get install -y ffmpeg git
```

### 3. Instalar PyTorch con CUDA

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verifica:
```bash
python -c "import torch; print(torch.cuda.is_available())"  # Debe ser True
```

### 4. Clonar VibeVoice

```bash
cd /workspace
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

Verifica voces:
```bash
ls -la demo/voices/streaming_model/
# Deberías ver archivos .pt como en_US-wayne-medium.pt
```

### 5. Instalar Dependencias del Proyecto

```bash
cd /workspace/testing-mvp
pip install -r requirements.txt
```

### 6. (Opcional) Instalar Flash Attention 2

```bash
# Solo si GPU tiene compute capability >= 8.0 (RTX 3090, A100, etc.)
pip install flash-attn --no-build-isolation
```

### 7. Configurar .env

```bash
cp .env.example .env
nano .env
```

Configuración mínima:
```bash
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx
TTS_ENGINE=vibevoice
VIBEVOICE_VOICE=Wayne
VIBEVOICE_VOICES_DIR=/workspace/VibeVoice/demo/voices/streaming_model
VIBEVOICE_CFG_SCALE=1.5
VIBEVOICE_DDPM_STEPS=5
```

### 8. Ejecutar Test

```bash
python test_vibevoice.py
```

Si todo sale bien:
```
✅ Todos los tests pasaron exitosamente!
```

### 9. Iniciar Servidor

```bash
python server.py
```

---

## ⚙️ Configuración

### Estructura de Persistent Storage

```
/workspace/
├── testing-mvp/              # Tu proyecto
│   ├── app/
│   │   ├── services/
│   │   │   └── tts_vibevoice.py
│   │   └── core/
│   │       └── config.py
│   ├── requirements.txt
│   ├── server.py
│   ├── .env                   # Tu configuración (NO subir a Git)
│   └── runpod_setup.sh
│
├── VibeVoice/                 # Repo de VibeVoice
│   ├── demo/
│   │   └── voices/
│   │       └── streaming_model/
│   │           ├── en_US-wayne-medium.pt     (~500MB)
│   │           └── en_US-sarah-medium.pt     (~500MB)
│   └── vibevoice/
│
└── .cache/                    # Cache de HuggingFace (automático)
    └── huggingface/
        └── hub/
            └── models--microsoft--VibeVoice-Realtime-0.5B/
```

### Variables de Entorno para RunPOD

**Archivo `.env` completo:**

```bash
# ============================================
# VERBA - RunPOD Configuration
# ============================================

# API Keys
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx

# TTS Engine Selection
TTS_ENGINE=vibevoice

# VibeVoice Configuration
VIBEVOICE_VOICE=Wayne
VIBEVOICE_VOICES_DIR=/workspace/VibeVoice/demo/voices/streaming_model
VIBEVOICE_CFG_SCALE=1.5
VIBEVOICE_DDPM_STEPS=5

# (Opcional) Si usas otros TTS
# KOKORO_VOICE=af_sarah
```

### Ajuste de Parámetros

#### Para Máxima Velocidad (RTF < 0.1x)
```bash
VIBEVOICE_CFG_SCALE=1.0
VIBEVOICE_DDPM_STEPS=3
```

#### Para Balance (Recomendado)
```bash
VIBEVOICE_CFG_SCALE=1.5
VIBEVOICE_DDPM_STEPS=5
```

#### Para Máxima Calidad
```bash
VIBEVOICE_CFG_SCALE=2.0
VIBEVOICE_DDPM_STEPS=20
```

---

## ✅ Verificación

### Checklist Post-Instalación

```bash
# 1. Verificar GPU
nvidia-smi
# Debe mostrar tu GPU

# 2. Verificar CUDA en PyTorch
python -c "import torch; print(torch.cuda.is_available())"
# Debe imprimir: True

# 3. Verificar VibeVoice instalado
python -c "from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference"
# No debe dar error

# 4. Verificar voces
ls /workspace/VibeVoice/demo/voices/streaming_model/*.pt
# Debe mostrar archivos .pt

# 5. Ejecutar test completo
python test_vibevoice.py

# 6. Probar síntesis directa
python test_vibevoice.py --text "Hello from RunPOD" --voice Wayne
```

### Tests de Performance

```bash
# Test de RTF (Real-Time Factor)
python -c "
from app.services.tts_vibevoice import VibeVoiceTTS
import time

tts = VibeVoiceTTS()
start = time.time()
wav = tts.synthesize('Testing performance on RunPOD GPU')
elapsed = time.time() - start

audio_duration = (len(wav) - 44) / 2 / 24000  # WAV header + samples
rtf = elapsed / audio_duration
print(f'RTF: {rtf:.2f}x (lower is better, <1.0 is real-time)')
"
```

**Benchmarks esperados:**

| GPU | RTF (DDPM=5) | Latencia (1s audio) |
|-----|--------------|---------------------|
| RTX 3090 | 0.08x | ~80ms |
| RTX 4090 | 0.06x | ~60ms |
| A100 | 0.05x | ~50ms |
| A6000 | 0.10x | ~100ms |

---

## 🔧 Troubleshooting RunPOD

### Error: "CUDA out of memory"

**Causa:** GPU VRAM insuficiente.

**Soluciones:**

1. **Reducir DDPM steps:**
   ```bash
   # En .env
   VIBEVOICE_DDPM_STEPS=3
   ```

2. **Liberar cache CUDA:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Reiniciar kernel:**
   ```bash
   # Reiniciar el servidor
   pkill -f server.py
   python server.py
   ```

4. **Upgrade GPU:**
   - Cambia a un POD con más VRAM (ej: de 3090 a A100)

### Error: "Failed to clone VibeVoice repository"

**Causa:** Problemas de red o Git.

**Solución:**

```bash
# Clonar manualmente
cd /workspace
git clone --depth 1 https://github.com/microsoft/VibeVoice.git

# Si sigue fallando, usar HTTPS con token
git clone https://<github-token>@github.com/microsoft/VibeVoice.git
```

### Error: "ModuleNotFoundError: No module named 'vibevoice'"

**Causa:** VibeVoice no instalado correctamente.

**Solución:**

```bash
cd /workspace/VibeVoice
pip install -e .

# Verificar
python -c "import vibevoice; print(vibevoice.__file__)"
```

### Error: "Voices directory not found"

**Causa:** Ruta incorrecta a voces.

**Solución:**

```bash
# Verificar que las voces existen
ls /workspace/VibeVoice/demo/voices/streaming_model/

# Actualizar .env con la ruta correcta
nano .env
# VIBEVOICE_VOICES_DIR=/workspace/VibeVoice/demo/voices/streaming_model
```

### Error: "Flash Attention 2 not available"

**Causa:** GPU no soportada o instalación fallida.

**Solución:**

Esto **NO es crítico**. El sistema automáticamente usa SDPA como fallback.

Para instalar Flash Attention 2 (opcional):
```bash
pip install flash-attn --no-build-isolation
```

Si falla, ignóralo. SDPA funciona perfectamente.

### Servidor no accesible desde navegador

**Causa:** Puerto no expuesto o firewall.

**Solución:**

1. **Verificar puerto en RunPOD:**
   - En RunPOD UI, asegúrate de que el puerto 8000 esté mapeado
   - Debería ser algo como: `8000 -> <puerto-externo>`

2. **Verificar servidor corriendo:**
   ```bash
   ps aux | grep server.py
   ```

3. **Verificar logs:**
   ```bash
   python server.py
   # Busca: "Uvicorn running on http://0.0.0.0:8000"
   ```

4. **Probar localmente primero:**
   ```bash
   curl http://localhost:8000
   # Debería retornar el HTML
   ```

### Audio no se genera

**Causa:** Múltiples posibles.

**Debug paso a paso:**

```bash
# 1. Test directo de VibeVoice
python test_vibevoice.py --text "Test" --voice Wayne

# 2. Verificar logs del servidor
tail -f logs/verba.log  # Si tienes logging a archivo

# 3. Test desde Python
python -c "
from app.services.tts_vibevoice import generate_audio_vibevoice
wav = generate_audio_vibevoice('Hello')
print('Success' if wav else 'Failed')
"
```

---

## ⚡ Optimizaciones

### 1. Pre-warmup del Modelo

Añade al inicio del servidor para pre-calentar:

```python
# En app/main.py, en el lifespan startup
from app.services.tts_vibevoice import init_vibevoice
init_vibevoice()

# Pre-warmup (opcional)
from app.services.tts_vibevoice import generate_audio_vibevoice
generate_audio_vibevoice("warmup")  # Síntesis dummy
```

### 2. Cache de HuggingFace en Persistent Volume

```bash
# Asegurar que cache esté en persistent storage
export HF_HOME=/workspace/.cache/huggingface
```

Añadir al `.env`:
```bash
HF_HOME=/workspace/.cache/huggingface
```

### 3. Usar Fast Boot Template

Crear un template de RunPOD con todo pre-instalado:

1. Ejecuta `runpod_setup.sh` una vez
2. En RunPOD UI: "Save Template"
3. Próximos PODs arrancan instantáneamente

### 4. Monitoring de GPU

```bash
# Watch GPU usage en tiempo real
watch -n 1 nvidia-smi
```

Para logging automático:
```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1 > gpu_log.csv &
```

---

## 📊 Comparación de GPUs en RunPOD

| GPU | VRAM | RTF (DDPM=5) | Costo/hora | Recomendación |
|-----|------|--------------|------------|---------------|
| **RTX 3090** | 24GB | 0.08x | ~$0.30 | ✅ Mejor balance |
| RTX 4090 | 24GB | 0.06x | ~$0.50 | Muy rápido pero caro |
| **A100** | 40GB | 0.05x | ~$1.00 | ✅ Para producción |
| A6000 | 48GB | 0.10x | ~$0.60 | Bueno para batch |
| RTX 3080 | 10GB | 0.15x | ~$0.20 | Budget (puede OOM) |

**Recomendación:** RTX 3090 para desarrollo, A100 para producción.

---

## 📝 FAQ

### ¿Cuánto cuesta ejecutar VibeVoice en RunPOD?

- **RTX 3090:** ~$0.30/hora
- **Horas de desarrollo:** 4-8 horas = $1.20 - $2.40
- **Producción (24/7):** $216/mes (considera Serverless)

### ¿Puedo usar CPU en lugar de GPU?

Sí, pero **NO es recomendado**. RTF sería ~5-10x, latencia inaceptable para real-time.

### ¿Qué pasa si reinicio el POD?

- **Con Persistent Storage:** Todo se mantiene
- **Sin Persistent Storage:** Debes reinstalar todo

Solución: Usa Persistent Storage siempre.

### ¿Puedo cambiar de voz en runtime?

Sí:
```python
# Cambiar en .env
VIBEVOICE_VOICE=Sarah

# O en código
from app.services.tts_vibevoice import generate_audio_vibevoice
generate_audio_vibevoice("Hello", voice_name="Sarah")
```

### ¿Cómo añado mis propias voces?

1. Graba muestras de voz (WAV, 24kHz)
2. Usa script de VibeVoice para generar embedding
3. Guarda archivo `.pt` en `/workspace/VibeVoice/demo/voices/streaming_model/`
4. Usa el nombre del archivo (sin `.pt`) como voice_name

Ver: [VibeVoice Fine-tuning Guide](https://github.com/microsoft/VibeVoice#fine-tuning)

---

## 🎯 Próximos Pasos

1. **Optimiza parámetros** según tu GPU
2. **Experimenta con voces** diferentes
3. **Monitorea performance** con GPU logs
4. **Crea un template** para fast boot
5. **Considera Serverless** para producción

---

## 📚 Recursos Adicionales

- **RunPOD Docs:** https://docs.runpod.io
- **VibeVoice GitHub:** https://github.com/microsoft/VibeVoice
- **README Local:** [README_VIBEVOICE.md](README_VIBEVOICE.md)
- **Quick Start:** [VIBEVOICE_QUICKSTART.md](VIBEVOICE_QUICKSTART.md)

---

## ✅ Checklist Final

- [ ] GPU validada con `nvidia-smi`
- [ ] CUDA disponible en PyTorch
- [ ] VibeVoice clonado e instalado
- [ ] Voces accesibles (archivos .pt)
- [ ] Requirements instalados
- [ ] `.env` configurado con Groq API key
- [ ] Test pasado (`test_vibevoice.py`)
- [ ] Servidor corriendo sin errores
- [ ] Aplicación accesible desde navegador
- [ ] Audio generándose correctamente

---

**¡Listo para producción!** 🚀

Si tienes problemas, revisa la sección [Troubleshooting](#troubleshooting-runpod) o consulta los recursos adicionales.
