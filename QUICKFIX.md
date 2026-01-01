# Quick Fixes - Errores Comunes

## ❌ Error: `ValueError: unsupported device cuda:0`

**Causa:** faster-whisper solo acepta `"cuda"` no `"cuda:0"`

**Solución:** Ya está corregido en la última versión. Si aún lo ves:

```bash
# Detén el servidor y reinicia
# Asegúrate de usar la versión actualizada
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

## ❌ Error: `ModuleNotFoundError: No module named 'src'`

**Causa:** Python no encuentra el módulo src

**Solución:** NO uses `python src/main.py`. Usa:

```bash
# Opción 1
python run.py

# Opción 2
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

## ❌ Error: `ModuleNotFoundError: No module named 'pipecat'`

**Causa:** Dependencias no instaladas

**Solución:**

```bash
pip install -r requirements-gpu.txt

# Si necesitas PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## ❌ Error: GROQ_API_KEY not set

**Causa:** Variable de entorno no configurada

**Solución:**

```bash
# Opción 1: Editar .env
nano .env
# Agregar: GROQ_API_KEY=tu_clave_aqui

# Opción 2: Exportar
export GROQ_API_KEY=tu_clave_aqui
```

Obtén tu clave en: https://console.groq.com

---

## ❌ Error: CUDA Out of Memory (OOM)

**Causa:** Modelo muy grande para la VRAM disponible

**Solución:**

```bash
# Opción 1: Usar modelo más pequeño
# Editar configs/gpu_optimized.yaml o crear custom:
# stt:
#   model: "medium"  # en lugar de large-v3

# Opción 2: Usar configuración para tu GPU
# Para T4 (16GB):
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
# La auto-detección debería elegir el modelo correcto
```

---

## ❌ Error: `ImportError: cannot import name 'LLMService'`

**Causa:** Deprecation warning en pipecat (no crítico)

**Solución:** Es solo un warning, el servidor debería funcionar. Para eliminarlo:

```bash
# Actualizar pipecat
pip install --upgrade pipecat-ai
```

---

## ⚠️ Warning: TTS provider shows 'vibevoice' instead of 'kokoro'

**Causa:** Kokoro puede no estar instalado correctamente

**Solución:**

```bash
# Reinstalar kokoro
pip uninstall kokoro
pip install kokoro>=0.3.4

# O usar Edge TTS como fallback (gratis, funciona bien)
# Editar .env:
# TTS_ENGINE=edge
```

---

## 🔍 Diagnóstico Rápido

Ejecuta el script de verificación:

```bash
python check_setup.py
```

Este script verificará:
- ✅ Versión de Python
- ✅ CUDA disponible
- ✅ Dependencias instaladas
- ✅ Variables de entorno
- ✅ GPU utilities
- ✅ Configuración

---

## 🚀 Inicio Limpio

Si nada funciona, reset completo:

```bash
# 1. Limpiar dependencias
pip uninstall -y pipecat-ai faster-whisper kokoro groq

# 2. Reinstalar desde cero
pip install -r requirements-gpu.txt

# 3. Reinstalar PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Verificar setup
python check_setup.py

# 5. Iniciar servidor
python run.py
```

---

## 📊 Verificar que está funcionando

```bash
# Terminal 1: Iniciar servidor
python run.py

# Terminal 2: Verificar
curl http://localhost:8000/health

# Deberías ver:
# {
#   "status": "healthy",
#   "gpu": {
#     "available": true,
#     "device": "cuda"
#   }
# }
```

---

## 🐛 Logs Detallados

Si necesitas más información de debug:

```bash
# Iniciar con logs detallados
python -m uvicorn src.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level debug
```

---

## 💡 Tips

1. **Siempre** verifica primero con `python check_setup.py`
2. **Usa** `python run.py` o `python -m uvicorn src.main:app`
3. **No uses** `python src/main.py` (causará import errors)
4. **Verifica** GROQ_API_KEY esté configurada
5. **Si usas RunPod**, asegúrate de que la GPU esté asignada

---

## 📞 Más Ayuda

- **Testing completo:** [TESTING.md](TESTING.md)
- **Documentación:** [README_PIPECAT.md](README_PIPECAT.md)
- **Migración:** [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

---

## ✅ Checklist Rápido

Antes de iniciar el servidor:

- [ ] `python check_setup.py` pasa todas las verificaciones
- [ ] GROQ_API_KEY está configurada en .env
- [ ] Dependencies instaladas (`pip list | grep pipecat`)
- [ ] GPU detectada (`nvidia-smi`)
- [ ] Puerto 8000 libre (`lsof -i :8000` debería estar vacío)

---

¡Listo para iniciar! 🚀

```bash
python run.py
```
