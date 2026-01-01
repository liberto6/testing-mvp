# 🚀 START HERE - Inicio Rápido

## ⚡ Inicio Ultra-Rápido (Recomendado)

Si tienes problemas con cuDNN (error "Unable to load libcudnn"):

```bash
# Opción 1: Script automático
./start_cpu.sh

# Opción 2: Manual
export CONFIG_PATH=configs/runpod_cpu_stt.yaml
python run.py
```

Esto usará:
- ✅ STT en CPU (evita cuDNN)
- ✅ TTS en GPU (Kokoro)
- ✅ LLM en Groq (ultra-rápido)

---

## 🎮 Si cuDNN Funciona

```bash
# Opción 1: Script automático
./start_gpu.sh

# Opción 2: Manual
export CONFIG_PATH=configs/runpod_optimized.yaml
python run.py
```

---

## 🔍 Verificar que Funciona

```bash
# En otra terminal
curl http://localhost:8000/health

# Deberías ver:
# {"status": "healthy", "gpu": {...}}
```

---

## 📋 Opciones de Configuración

### 1. CPU STT (Sin cuDNN, Siempre Funciona)
```bash
export CONFIG_PATH=configs/runpod_cpu_stt.yaml
python run.py
```

### 2. GPU Completo (Requiere cuDNN)
```bash
export CONFIG_PATH=configs/runpod_optimized.yaml
python run.py
```

### 3. Auto-Detección (Default)
```bash
# No exportar CONFIG_PATH
python run.py
```

---

## ❌ Si Hay Errores

### Error: "Unable to load libcudnn"
**Solución:** Usa `./start_cpu.sh`

### Error: "ModuleNotFoundError: No module named 'src'"
**Solución:** Usa `python run.py` NO `python src/main.py`

### Error: "GROQ_API_KEY not set"
**Solución:**
```bash
nano .env
# Agregar: GROQ_API_KEY=tu_clave
```

---

## 📊 Comparación de Configs

| Config | STT Device | STT Latency | Requiere cuDNN |
|--------|------------|-------------|----------------|
| `runpod_cpu_stt.yaml` | CPU | ~200ms | ❌ No |
| `runpod_optimized.yaml` | GPU | ~90ms | ✅ Sí |
| Auto-detect | GPU/CPU | Variable | Depende |

---

## 🎯 Mi Recomendación

**Para empezar AHORA:**
```bash
./start_cpu.sh
```

**Luego, arreglar cuDNN:**
```bash
./fix_cudnn.sh
```

**Después, usar GPU completo:**
```bash
./start_gpu.sh
```

---

## 📞 Más Ayuda

- **Errores de cuDNN:** [CUDNN_FIX.md](CUDNN_FIX.md)
- **Soluciones rápidas:** [QUICKFIX.md](QUICKFIX.md)
- **Documentación completa:** [README_PIPECAT.md](README_PIPECAT.md)

---

## ✅ Comando para Copiar y Pegar

```bash
cd /workspace/testing-mvp
./start_cpu.sh
```

¡Listo! 🎉
