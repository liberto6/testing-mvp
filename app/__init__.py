import sys
import os

# Asegurar que F5-TTS esté en el path ANTES de importar cualquier submódulo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Sube desde app/ a root
F5_TTS_SRC = os.path.join(BASE_DIR, "F5-TTS", "src")

if os.path.exists(F5_TTS_SRC) and F5_TTS_SRC not in sys.path:
    print(f"🔧 Inyectando F5-TTS en sys.path: {F5_TTS_SRC}")
    sys.path.insert(0, F5_TTS_SRC) # Insertar al principio para prioridad
