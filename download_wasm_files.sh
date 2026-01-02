#!/bin/bash
# Script para descargar archivos WASM necesarios para ONNX Runtime

echo "📦 Descargando archivos WASM de ONNX Runtime..."

cd static

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
