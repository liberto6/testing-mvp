# VibeVoice TTS Integration Guide

Esta guía documenta la integración de **Microsoft VibeVoice Realtime TTS** en el pipeline de Verba.

## 📋 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Características](#características)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso](#uso)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Optimizaciones](#optimizaciones)

---

## 🎯 Descripción General

**VibeVoice** es un modelo de síntesis de voz en tiempo real desarrollado por Microsoft Research. Ofrece:

- **Streaming de audio** para baja latencia
- **Alta calidad** de síntesis (24kHz, comparable a sistemas comerciales)
- **Múltiples voces** pre-embedidas (Wayne, Sarah, etc.)
- **Soporte multi-dispositivo** (CUDA, MPS/Apple Silicon, CPU)

### Especificaciones Técnicas

| Propiedad | Valor |
|-----------|-------|
| Modelo | microsoft/VibeVoice-Realtime-0.5B |
| Sample Rate | 24 kHz |
| Canales | Mono (1) |
| Formato de salida | 16-bit PCM WAV |
| Formato interno | Float32 [-1, 1] |
| Latencia típica | ~0.5-2s (RTF 0.1-0.5x en GPU) |

---

## ✨ Características

### 1. **Detección Automática de Dispositivo**
```python
# Auto-detecta CUDA > MPS > CPU
tts = VibeVoiceTTS()  # device='auto'
```

### 2. **Gestión de Voces**
- Voces pre-embedidas en archivos `.pt`
- Cache automático de voces cargadas
- Escaneo dinámico del directorio de voces

### 3. **Streaming (Futuro)**
- Soporte para `AudioStreamer` y `AsyncAudioStreamer`
- Chunks de audio progresivos
- Integración con WebSocket para streaming real-time

### 4. **Manejo Robusto de Errores**
- Fallback SDPA si Flash Attention 2 falla
- Detección de CUDA OOM
- Logging detallado de errores

### 5. **Optimizaciones de Rendimiento**
- Lazy loading del modelo
- Flash Attention 2 en CUDA (mejor calidad)
- DDPM con 5 pasos (balance velocidad/calidad)
- Cache de voces en memoria

---

## 📦 Instalación

### Paso 1: Instalar Dependencias

```bash
# En el directorio del proyecto
pip install -r requirements.txt
```

Las dependencias clave son:
- `transformers==4.51.3` (versión exacta requerida)
- `torch` (con soporte CUDA si disponible)
- `accelerate==1.6.0`
- `diffusers`
- `librosa`
- `scipy`
- `soundfile`

### Paso 2: Instalar VibeVoice

```bash
# Clonar el repositorio de VibeVoice
cd /Users/pepeda-rosa/Documents/Verba/Repositorios/
git clone https://github.com/microsoft/VibeVoice.git

# Instalar en modo editable
cd VibeVoice
pip install -e .
```

### Paso 3: Verificar Voces

Las voces deben estar en:
```
/Users/pepeda-rosa/Documents/Verba/Repositorios/MicrosoftVibeVoice/VibeVoice/demo/voices/streaming_model/
```

Archivos esperados:
- `en_US-wayne-medium.pt`
- `en_US-sarah-medium.pt`
- Otros archivos `.pt`

### Paso 4: Ejecutar Test

```bash
python test_vibevoice.py --text "Hello world" --voice Wayne
```

Si todo está correcto, verás:
```
✅ Todos los tests pasaron exitosamente!
🎉 VibeVoice TTS está correctamente integrado.
```

---

## ⚙️ Configuración

### Variables de Entorno

Crea o edita tu archivo `.env`:

```bash
# Motor TTS activo
TTS_ENGINE=vibevoice

# Configuración de VibeVoice
VIBEVOICE_VOICE=Wayne                    # Voz por defecto (Wayne, Sarah, etc.)
VIBEVOICE_VOICES_DIR=/ruta/a/voices/     # Directorio con archivos .pt
VIBEVOICE_CFG_SCALE=1.5                  # Classifier-Free Guidance (1.0-3.0)
VIBEVOICE_DDPM_STEPS=5                   # Pasos de difusión (3-50)
```

### Configuración de Voces

Edita `app/config/vibevoice_voices.json`:

```json
{
  "default_voice": "Wayne",
  "available_voices": {
    "Wayne": {
      "file": "en_US-wayne-medium.pt",
      "description": "Male voice, medium pitch",
      "gender": "male"
    },
    "Sarah": {
      "file": "en_US-sarah-medium.pt",
      "description": "Female voice, medium pitch",
      "gender": "female"
    }
  },
  "synthesis_parameters": {
    "cfg_scale": {
      "default": 1.5,
      "min": 1.0,
      "max": 3.0
    },
    "ddpm_steps": {
      "default": 5,
      "min": 3,
      "max": 50
    }
  }
}
```

---

## 🚀 Uso

### Opción 1: A través del Pipeline (Recomendado)

```python
# En tu código existente
from app.services.tts import run_tts

# Configura TTS_ENGINE=vibevoice en .env
wav_bytes = await run_tts("Hello, this is a test.")
```

### Opción 2: Uso Directo

```python
from app.services.tts_vibevoice import VibeVoiceTTS

# Inicializar
tts = VibeVoiceTTS(
    model_path="microsoft/VibeVoice-Realtime-0.5B",
    device="cuda",  # o "mps", "cpu", None (auto)
    cfg_scale=1.5,
    ddpm_steps=5
)

# Sintetizar
wav_bytes = tts.synthesize(
    text="Hello, how are you?",
    voice_name="Wayne",
    return_format="wav_bytes"
)

# Guardar
with open("output.wav", "wb") as f:
    f.write(wav_bytes)
```

### Opción 3: API Pública

```python
from app.services.tts_vibevoice import generate_audio_vibevoice, init_vibevoice

# Inicializar (solo una vez)
init_vibevoice()

# Generar audio
wav_bytes = generate_audio_vibevoice(
    text="Testing the API",
    voice_name="Sarah"
)
```

---

## 📚 API Reference

### Clase `VibeVoiceTTS`

```python
class VibeVoiceTTS:
    def __init__(
        self,
        model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
        voices_dir: str = "/path/to/voices/",
        device: Optional[str] = None,  # Auto-detect if None
        cfg_scale: float = 1.5,
        ddpm_steps: int = 5,
    )

    def synthesize(
        self,
        text: str,
        voice_name: str = "Wayne",
        return_format: str = "wav_bytes",  # 'wav_bytes', 'numpy', 'torch'
    ) -> Optional[bytes]

    def get_available_voices(self) -> List[str]
```

### Funciones Públicas

```python
def init_vibevoice() -> None:
    """Inicializa el pipeline global (llamado en app startup)"""

def generate_audio_vibevoice(
    text: str,
    voice_name: str = "Wayne"
) -> Optional[bytes]:
    """Genera audio WAV (función compatible con el router TTS)"""
```

---

## 🔧 Troubleshooting

### Error: "No module named 'vibevoice'"

**Solución:**
```bash
cd /path/to/VibeVoice
pip install -e .
```

### Error: "Flash Attention 2 not available"

**Solución:** Esto es normal en CPU/MPS. El sistema automáticamente hace fallback a SDPA.

Para habilitar Flash Attention 2 en CUDA:
```bash
pip install flash-attn --no-build-isolation
```

### Error: "CUDA out of memory"

**Soluciones:**
1. Reduce `ddpm_steps` a 3
2. Usa `device="cpu"`
3. Limpia cache: `torch.cuda.empty_cache()`

### Audio Quality Issues

**Ajustar `cfg_scale`:**
- `1.0`: Más estable, menos expresivo
- `1.5`: Balance (default)
- `2.0-3.0`: Más expresivo, potencialmente inestable

**Ajustar `ddpm_steps`:**
- `3`: Muy rápido, calidad reducida
- `5`: Balance (default)
- `10-20`: Mejor calidad, más lento
- `50`: Máxima calidad (muy lento)

### Voice Not Found

Verifica que el archivo `.pt` existe:
```bash
ls -la /path/to/voices/streaming_model/
```

Nombres comunes:
- `en_US-wayne-medium.pt`
- `en_US-sarah-medium.pt`

---

## ⚡ Optimizaciones

### 1. **Rendimiento en GPU**

```python
# Habilitar Flash Attention 2 (CUDA only)
pip install flash-attn

# Usar bfloat16 (CUDA)
# Configurado automáticamente en CUDA
```

### 2. **Reducir Latencia**

```python
# Usar menos pasos de difusión
VIBEVOICE_DDPM_STEPS=3

# Reducir CFG scale
VIBEVOICE_CFG_SCALE=1.0
```

### 3. **Caché de Voces**

Las voces se cachean automáticamente en memoria:
```python
# Primera carga: ~2-5s
tts.synthesize("Hello", voice_name="Wayne")

# Subsecuentes: instantáneo
tts.synthesize("World", voice_name="Wayne")
```

### 4. **Batch Processing (Futuro)**

Actualmente solo soporta `batch_size=1`. Para múltiples textos:
```python
for text in texts:
    wav = tts.synthesize(text)
    process(wav)
```

---

## 🎛️ Parámetros Avanzados

### CFG Scale (Classifier-Free Guidance)

Controla la adherencia al condicionamiento (voz):

| Valor | Efecto |
|-------|--------|
| 1.0 | Mínimo guidance, más natural pero menos controlado |
| 1.5 | Balance (recomendado) |
| 2.0 | Más expresivo |
| 3.0 | Máximo guidance, puede sonar artificial |

### DDPM Steps

Pasos del sampler de difusión:

| Valor | RTF (GPU) | Calidad |
|-------|-----------|---------|
| 3 | 0.05x | Baja |
| 5 | 0.1x | Media (recomendado) |
| 10 | 0.2x | Alta |
| 20 | 0.4x | Muy alta |
| 50 | 1.0x | Máxima |

---

## 📊 Comparación con Otros Motores TTS

| Motor | Latencia | Calidad | Streaming | Recursos |
|-------|----------|---------|-----------|----------|
| **VibeVoice** | Baja-Media | Alta | ✅ | GPU > MPS > CPU |
| Kokoro | Muy Baja | Media-Alta | ✅ | CPU-friendly |
| F5-TTS | Media | Muy Alta | ❌ | GPU recomendado |

**Recomendación:**
- **Producción con GPU:** VibeVoice
- **Desarrollo local (Mac):** VibeVoice con MPS o Kokoro
- **CPU solo:** Kokoro
- **Máxima calidad (offline):** F5-TTS

---

## 🔗 Referencias

- [VibeVoice GitHub](https://github.com/microsoft/VibeVoice)
- [VibeVoice Paper](https://arxiv.org/abs/2501.xxxxx)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)

---

## 📝 Ejemplo Completo

```python
# ejemplo_vibevoice.py
import asyncio
from app.services.tts_vibevoice import VibeVoiceTTS

async def main():
    # Inicializar TTS
    tts = VibeVoiceTTS(
        device="cuda",  # o "mps", "cpu"
        cfg_scale=1.5,
        ddpm_steps=5
    )

    # Textos a sintetizar
    texts = [
        "Hello, welcome to Verba.",
        "This is a test of the VibeVoice system.",
        "The audio quality is excellent."
    ]

    # Generar audio para cada texto
    for i, text in enumerate(texts):
        print(f"Generando audio {i+1}/{len(texts)}...")
        wav_bytes = tts.synthesize(
            text=text,
            voice_name="Wayne",
            return_format="wav_bytes"
        )

        # Guardar
        output_path = f"output_{i+1}.wav"
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        print(f"  ✅ Guardado: {output_path}")

    # Listar voces disponibles
    voices = tts.get_available_voices()
    print(f"\nVoces disponibles: {', '.join(voices)}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🎉 Próximos Pasos

1. **Activar VibeVoice:**
   ```bash
   echo "TTS_ENGINE=vibevoice" >> .env
   python server.py
   ```

2. **Probar en tu aplicación:**
   - Abre `http://localhost:8000`
   - Habla con Sarah (el tutor)
   - Escucha la respuesta con VibeVoice

3. **Experimentar con voces:**
   - Cambia `VIBEVOICE_VOICE=Sarah` en `.env`
   - Reinicia el servidor
   - Compara la calidad

4. **Optimizar parámetros:**
   - Ajusta `cfg_scale` y `ddpm_steps`
   - Mide RTF y latencia
   - Encuentra el balance óptimo

---

**¡Listo!** VibeVoice está integrado en tu pipeline. 🚀
