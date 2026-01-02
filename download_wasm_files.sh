#!/bin/bash
# Script para descargar archivos WASM necesarios para ONNX Runtime y modelo VAD

echo "📦 Descargando archivos necesarios para VAD..."

cd static

# 1. Descargar modelo VAD si no existe o está corrupto
echo ""
echo "1️⃣ Verificando modelo silero_vad.onnx..."
if [ -f "silero_vad.onnx" ]; then
    SIZE=$(stat -f%z "silero_vad.onnx" 2>/dev/null || stat -c%s "silero_vad.onnx" 2>/dev/null)
    if [ "$SIZE" -lt 200000 ]; then
        echo "⚠️  Archivo corrupto (solo $SIZE bytes), descargando de nuevo..."
        rm silero_vad.onnx
    else
        echo "✓ silero_vad.onnx existe y parece válido ($SIZE bytes)"
    fi
fi

if [ ! -f "silero_vad.onnx" ]; then
    echo "⬇️  Descargando silero_vad.onnx..."
    curl -sL "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx" -o "silero_vad.onnx"
    if [ $? -eq 0 ]; then
        echo "✅ silero_vad.onnx descargado"
    else
        echo "❌ Error descargando silero_vad.onnx"
    fi
fi

# 2. Descargar archivos WASM
echo ""
echo "2️⃣ Descargando archivos WASM de ONNX Runtime..."

# Version de ONNX Runtime que usa vad-web
VERSION="1.19.0"

# Descargar archivos WASM necesarios
FILES=(
    "ort-wasm-simd-threaded.jsep.mjs"
    "ort-wasm-simd-threaded.jsep.wasm"
    "ort-wasm-simd-threaded.mjs"
    "ort-wasm-simd-threaded.wasm"
    "ort-wasm-simd.mjs"
    "ort-wasm-simd.wasm"
    "ort-wasm-threaded.mjs"
    "ort-wasm-threaded.wasm"
    "ort-wasm.mjs"
    "ort-wasm.wasm"
)

BASE_URL="https://cdn.jsdelivr.net/npm/onnxruntime-web@${VERSION}/dist/"

for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⬇️  Descargando $file..."
        curl -sL "${BASE_URL}${file}" -o "$file"
        if [ $? -eq 0 ]; then
            echo "✅ $file descargado"
        else
            echo "❌ Error descargando $file"
        fi
    else
        echo "✓ $file ya existe"
    fi
done

echo ""
echo "✅ Descarga completa"
echo ""
echo "Archivos en static/:"
ls -lh *.wasm *.mjs 2>/dev/null || echo "No se encontraron archivos WASM"
