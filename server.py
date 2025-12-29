import sys
import os
import uvicorn

# Inyección explícita antes de importar app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
F5_TTS_SRC = os.path.join(BASE_DIR, "F5-TTS", "src")
if os.path.exists(F5_TTS_SRC) and F5_TTS_SRC not in sys.path:
    sys.path.insert(0, F5_TTS_SRC)

from app.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
