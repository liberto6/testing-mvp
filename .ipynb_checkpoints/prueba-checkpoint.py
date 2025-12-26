import requests

# Probar conexión con Ollama (LLM)
try:
    res = requests.post("http://localhost:11434/api/generate", 
                         json={"model": "llama3.1", "prompt": "Hola, ¿estás funcionando?", "stream": False})
    print("Ollama dice:", res.json()['response'])
except Exception as e:
    print("Error conectando con Ollama:", e)

# Probar si F5-TTS detecta la GPU
import torch
print("¿GPU disponible para F5-TTS?:", torch.cuda.is_available())