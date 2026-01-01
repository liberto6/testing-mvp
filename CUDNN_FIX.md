# Soluciones para Error de cuDNN

## ❌ Error
```
Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so}
Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor
Aborted (core dumped)
```

Este error ocurre porque `faster-whisper` necesita cuDNN y puede no estar compatible con tu versión.

---

## 🚀 Solución 1: Usar CPU para STT (Más Rápido de Implementar)

### Opción A: Usar configuración CPU

```bash
# Usar config con STT en CPU
export CONFIG_PATH=configs/runpod_cpu_stt.yaml
python run.py
```

### Opción B: Modificar config manualmente

Edita tu `.env`:
```bash
# Forzar CPU para STT
export WHISPER_MODEL=base
export FORCE_CPU=true
```

**Ventajas:**
- ✅ Funciona inmediatamente
- ✅ No requiere arreglar cuDNN
- ✅ TTS puede seguir usando GPU

**Desventajas:**
- ⚠️ STT será más lento (~200-300ms en lugar de ~90ms)
- ⚠️ Solo puede usar modelos pequeños (tiny/base/small)

---

## 🔧 Solución 2: Arreglar cuDNN (Recomendado para Producción)

### Paso 1: Instalar cuDNN

```bash
# Ejecutar script de fix
./fix_cudnn.sh

# O manualmente:
pip install nvidia-cudnn-cu12
```

### Paso 2: Configurar LD_LIBRARY_PATH

```bash
# Encontrar cuDNN
find /usr -name "libcudnn*.so*" 2>/dev/null

# Agregar al PATH (reemplaza con tu ruta)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Hacer permanente
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Paso 3: Verificar

```bash
# Test rápido
python -c "from faster_whisper import WhisperModel; print('✅ cuDNN OK')"

# Si funciona, iniciar servidor
python run.py
```

---

## 🔄 Solución 3: Usar OpenAI Whisper (Alternativa)

OpenAI Whisper es más compatible que faster-whisper pero un poco más lento.

### Instalar

```bash
pip install openai-whisper
```

### Modificar código

Edita `src/pipeline.py` y cambia:

```python
# ANTES:
from src.processors.stt_whisper_gpu import WhisperGPUProcessor, WhisperGPUConfig

# DESPUÉS:
from src.processors.stt_whisper_openai import OpenAIWhisperProcessor, OpenAIWhisperConfig
```

Y en el init:

```python
# ANTES:
self.stt_processor = WhisperGPUProcessor(
    config=WhisperGPUConfig(...)
)

# DESPUÉS:
self.stt_processor = OpenAIWhisperProcessor(
    config=OpenAIWhisperConfig(
        model_size=self.config.stt.model,
        device=self.config.stt.device,
        language=self.config.stt.language
    )
)
```

---

## 🐳 Solución 4: Usar Docker con cuDNN Incluido

Si estás en RunPod, usa el Dockerfile que incluye cuDNN:

```bash
# Build con cuDNN incluido
docker build -f deployment/runpod/Dockerfile -t pipecat-voice .

# Run
docker run --gpus all -p 8000:8000 \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  pipecat-voice
```

---

## 🧪 Diagnóstico

### Verificar CUDA

```bash
nvidia-smi
nvcc --version
```

### Verificar cuDNN

```bash
# Buscar librerías
ldconfig -p | grep cudnn

# Buscar archivos
find /usr -name "libcudnn*.so*" 2>/dev/null

# Python check
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Verificar faster-whisper

```bash
python -c "from faster_whisper import WhisperModel; model = WhisperModel('tiny', device='cuda'); print('OK')"
```

---

## ✅ Recomendación por Escenario

### Testing/Desarrollo Rápido
→ **Solución 1**: Usar CPU para STT
```bash
export CONFIG_PATH=configs/runpod_cpu_stt.yaml
python run.py
```

### Producción con RunPod
→ **Solución 2**: Arreglar cuDNN
```bash
./fix_cudnn.sh
python run.py
```

### Máxima Compatibilidad
→ **Solución 3**: Usar OpenAI Whisper
```bash
pip install openai-whisper
# Modificar src/pipeline.py según instrucciones arriba
python run.py
```

---

## 📊 Comparación de Performance

| Solución | Latencia STT | Calidad | Complejidad |
|----------|--------------|---------|-------------|
| CPU STT (base) | ~200ms | Media | ⭐ Fácil |
| faster-whisper GPU | ~90ms | Alta | ⭐⭐⭐ Complejo |
| openai-whisper GPU | ~120ms | Alta | ⭐⭐ Medio |

---

## 🎯 Mi Recomendación

Para empezar **ahora mismo**:

```bash
# 1. Usar config CPU (funciona siempre)
export CONFIG_PATH=configs/runpod_cpu_stt.yaml
python run.py

# 2. En paralelo, arreglar cuDNN
./fix_cudnn.sh

# 3. Cuando cuDNN funcione, volver a config GPU
python run.py  # Sin CONFIG_PATH, usará auto-detección
```

---

## 💡 Tips Adicionales

1. **Kokoro TTS puede seguir en GPU** incluso si STT está en CPU
2. **El performance general seguirá siendo bueno** (LLM es el componente más lento)
3. **CPU STT con modelo "base" es aceptable** para desarrollo
4. **Para producción, vale la pena arreglar cuDNN** para usar GPU

---

## 📞 Siguiente Paso

Ejecuta esto AHORA para iniciar con CPU STT:

```bash
cd /workspace/testing-mvp
export CONFIG_PATH=configs/runpod_cpu_stt.yaml
python run.py
```

Deberías ver el servidor iniciando sin el error de cuDNN! 🚀
