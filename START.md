# 🚀 Cómo Iniciar Pipecat Voice Pipeline

## Opciones para Iniciar el Servidor

### Opción 1: Script run.py (RECOMENDADO)
```bash
python run.py
```

### Opción 2: Usando uvicorn como módulo
```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Opción 3: Quick Start Script
```bash
./quick_start.sh
```

### Opción 4: Docker
```bash
docker-compose up -d
```

---

## ❌ NO USAR (Causará error de imports)
```bash
# ❌ NO FUNCIONA
python src/main.py

# ❌ NO FUNCIONA
python3 src/main.py
```

**¿Por qué?** Python no encuentra el módulo `src` cuando se ejecuta así.

---

## ✅ Verificar que el servidor está corriendo

```bash
# Health check
curl http://localhost:8000/health

# Ver configuración
curl http://localhost:8000/config

# Info del servidor
curl http://localhost:8000/
```

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'src'"

**Solución:** Usa uno de los métodos recomendados arriba.

### Error: "ModuleNotFoundError: No module named 'pipecat'"

**Solución:**
```bash
pip install -r requirements-gpu.txt
```

### Error: "GROQ_API_KEY not set"

**Solución:**
```bash
# Editar .env
nano .env

# O exportar variable
export GROQ_API_KEY=your_key_here
```

---

## 📍 Tu Ubicación

Estás en: `/workspace/testing-mvp`

Comandos correctos desde esta ubicación:

```bash
# Iniciar servidor
python run.py

# O con uvicorn
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Verificar salud
curl http://localhost:8000/health
```

---

## 🎯 Próximos Pasos

Una vez que el servidor esté corriendo:

1. **Verificar health:** `curl http://localhost:8000/health`
2. **Ver configuración:** `curl http://localhost:8000/config`
3. **Conectar frontend:** Abrir `http://localhost:8000/index.html`
4. **Probar WebSocket:** Usar test client en `TESTING.md`

---

## 📚 Más Información

- **Documentación completa:** [README_PIPECAT.md](README_PIPECAT.md)
- **Guía de migración:** [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Testing:** [TESTING.md](TESTING.md)
- **Resumen técnico:** [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
