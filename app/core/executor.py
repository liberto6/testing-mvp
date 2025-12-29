from concurrent.futures import ThreadPoolExecutor

# Executor para tareas bloqueantes (STT y TTS)
# Max workers limitado para no saturar CPU/GPU
executor = ThreadPoolExecutor(max_workers=3)
