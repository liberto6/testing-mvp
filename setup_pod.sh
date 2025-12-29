#!/bin/bash

echo "🚀 Iniciando configuración del entorno en el Pod..."

# 1. Instalar dependencias del sistema (ffmpeg es crucial para audio)
echo "📦 Instalando dependencias del sistema..."
apt-get update && apt-get install -y ffmpeg

# 2. Instalar PyTorch con soporte CUDA (Asegura uso de GPU)
echo "🔥 Instalando PyTorch con CUDA..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Instalar dependencias del proyecto Verba
echo "📚 Instalando dependencias de Verba..."
pip install -r requirements.txt

# 4. Instalar F5-TTS en modo editable
echo "🗣️ Instalando F5-TTS..."
cd F5-TTS
pip install -e .
cd ..

echo "✅ Instalación completada! Ahora puedes ejecutar el servidor con:"
echo "python server.py"
