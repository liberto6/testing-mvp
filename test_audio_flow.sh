#!/bin/bash

################################################################################
# Test de Flujo Completo de Audio
# Diagnostica problemas de "no me escucha"
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   🔍 DIAGNÓSTICO COMPLETO DEL SISTEMA DE AUDIO${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

ERRORS=0

################################################################################
# Test 1: GPU y CUDA
################################################################################

echo -e "${BLUE}[1/10] Verificando GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo -e "  ${GREEN}✅ GPU detectada: $GPU_NAME${NC}"
else
    echo -e "  ${YELLOW}⚠️  nvidia-smi no disponible (puede que estés en CPU)${NC}"
fi
echo ""

################################################################################
# Test 2: CUDA en PyTorch
################################################################################

echo -e "${BLUE}[2/10] Verificando CUDA en PyTorch...${NC}"
python3 -c "
import torch
if torch.cuda.is_available():
    print('  \033[0;32m✅ CUDA disponible\033[0m')
    print(f'  Device: {torch.cuda.get_device_name(0)}')
else:
    print('  \033[1;33m⚠️  CUDA NO disponible (usando CPU)\033[0m')
" 2>&1
echo ""

################################################################################
# Test 3: Faster-Whisper (STT)
################################################################################

echo -e "${BLUE}[3/10] Verificando Faster-Whisper (STT)...${NC}"
python3 -c "
try:
    from faster_whisper import WhisperModel
    print('  \033[0;32m✅ Faster-Whisper instalado\033[0m')
except ImportError as e:
    print('  \033[0;31m❌ Faster-Whisper NO instalado\033[0m')
    print(f'  Error: {e}')
    exit(1)
" 2>&1
if [ $? -ne 0 ]; then
    ((ERRORS++))
fi
echo ""

################################################################################
# Test 4: VibeVoice TTS
################################################################################

echo -e "${BLUE}[4/10] Verificando VibeVoice TTS...${NC}"
python3 -c "
try:
    from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
    print('  \033[0;32m✅ VibeVoice instalado\033[0m')
except ImportError as e:
    print('  \033[0;31m❌ VibeVoice NO instalado\033[0m')
    print(f'  Error: {e}')
    exit(1)
" 2>&1
if [ $? -ne 0 ]; then
    ((ERRORS++))
fi
echo ""

################################################################################
# Test 5: Archivo .env
################################################################################

echo -e "${BLUE}[5/10] Verificando archivo .env...${NC}"
if [ -f .env ]; then
    echo -e "  ${GREEN}✅ .env existe${NC}"

    # Verificar variables clave
    TTS_ENGINE=$(grep "^TTS_ENGINE=" .env 2>/dev/null | cut -d '=' -f2)
    VIBEVOICE_VOICE=$(grep "^VIBEVOICE_VOICE=" .env 2>/dev/null | cut -d '=' -f2)
    GROQ_KEY=$(grep "^GROQ_API_KEY=" .env 2>/dev/null | cut -d '=' -f2)

    if [ -n "$TTS_ENGINE" ]; then
        echo -e "  ${GREEN}✅ TTS_ENGINE=$TTS_ENGINE${NC}"
    else
        echo -e "  ${RED}❌ TTS_ENGINE no configurado${NC}"
        ((ERRORS++))
    fi

    if [ -n "$VIBEVOICE_VOICE" ]; then
        echo -e "  ${GREEN}✅ VIBEVOICE_VOICE=$VIBEVOICE_VOICE${NC}"
    else
        echo -e "  ${YELLOW}⚠️  VIBEVOICE_VOICE no configurado (usará default)${NC}"
    fi

    if [ -n "$GROQ_KEY" ]; then
        if [[ "$GROQ_KEY" == *"your_"* ]] || [[ "$GROQ_KEY" == *"_here"* ]]; then
            echo -e "  ${RED}❌ GROQ_API_KEY NO configurado (tiene valor placeholder)${NC}"
            ((ERRORS++))
        else
            echo -e "  ${GREEN}✅ GROQ_API_KEY configurado${NC}"
        fi
    else
        echo -e "  ${RED}❌ GROQ_API_KEY no encontrado en .env${NC}"
        ((ERRORS++))
    fi
else
    echo -e "  ${RED}❌ .env NO existe${NC}"
    ((ERRORS++))
fi
echo ""

################################################################################
# Test 6: Voces de VibeVoice
################################################################################

echo -e "${BLUE}[6/10] Verificando voces de VibeVoice...${NC}"
VOICES_DIR="/workspace/VibeVoice/demo/voices/streaming_model"

if [ -d "$VOICES_DIR" ]; then
    VOICE_COUNT=$(ls -1 "$VOICES_DIR"/*.pt 2>/dev/null | wc -l)
    if [ $VOICE_COUNT -gt 0 ]; then
        echo -e "  ${GREEN}✅ $VOICE_COUNT voces encontradas en $VOICES_DIR${NC}"
        ls -1 "$VOICES_DIR"/*.pt 2>/dev/null | head -n 3 | sed 's/^/    /'
        if [ $VOICE_COUNT -gt 3 ]; then
            echo "    ..."
        fi
    else
        echo -e "  ${RED}❌ No se encontraron archivos .pt en $VOICES_DIR${NC}"
        ((ERRORS++))
    fi
else
    echo -e "  ${RED}❌ Directorio de voces no encontrado: $VOICES_DIR${NC}"
    ((ERRORS++))
fi
echo ""

################################################################################
# Test 7: Test de STT (Whisper)
################################################################################

echo -e "${BLUE}[7/10] Test rápido de STT (Whisper)...${NC}"
python3 -c "
import numpy as np
from app.services.stt import run_stt
import asyncio

# Crear audio dummy (silencio con un poco de ruido)
audio_np = np.random.randn(16000).astype(np.float32) * 0.1

try:
    result = asyncio.run(run_stt(audio_np))
    print('  \033[0;32m✅ STT funciona (puede que no detecte texto en audio dummy)\033[0m')
    if result:
        print(f'  Resultado: \"{result}\"')
    else:
        print('  (Sin resultado esperado con audio dummy)')
except Exception as e:
    print('  \033[0;31m❌ STT falló\033[0m')
    print(f'  Error: {e}')
    exit(1)
" 2>&1
if [ $? -ne 0 ]; then
    ((ERRORS++))
fi
echo ""

################################################################################
# Test 8: Test de TTS (VibeVoice)
################################################################################

echo -e "${BLUE}[8/10] Test rápido de TTS (VibeVoice)...${NC}"
python3 -c "
from app.services.tts_vibevoice import generate_audio_vibevoice
import time

try:
    start = time.time()
    wav = generate_audio_vibevoice('Test')
    elapsed = time.time() - start

    if wav and len(wav) > 0:
        print(f'  \033[0;32m✅ TTS funciona ({elapsed:.2f}s, {len(wav):,} bytes)\033[0m')
    else:
        print('  \033[0;31m❌ TTS retornó None o vacío\033[0m')
        exit(1)
except Exception as e:
    print('  \033[0;31m❌ TTS falló\033[0m')
    print(f'  Error: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1
if [ $? -ne 0 ]; then
    ((ERRORS++))
fi
echo ""

################################################################################
# Test 9: Test de LLM (Groq)
################################################################################

echo -e "${BLUE}[9/10] Test rápido de LLM (Groq)...${NC}"
python3 -c "
from app.services.llm import stream_sentences
import asyncio

async def test_llm():
    chat_history = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Say hi briefly'}
    ]

    sentences = []
    try:
        async for sentence in stream_sentences(chat_history):
            sentences.append(sentence)
            if len(sentences) >= 1:  # Solo primera frase para test rápido
                break

        if sentences:
            print(f'  \033[0;32m✅ LLM funciona\033[0m')
            print(f'  Respuesta: \"{sentences[0]}\"')
            return True
        else:
            print('  \033[0;31m❌ LLM no retornó nada\033[0m')
            return False
    except Exception as e:
        print('  \033[0;31m❌ LLM falló\033[0m')
        print(f'  Error: {e}')
        return False

result = asyncio.run(test_llm())
exit(0 if result else 1)
" 2>&1
if [ $? -ne 0 ]; then
    ((ERRORS++))
fi
echo ""

################################################################################
# Test 10: Verificar servidor
################################################################################

echo -e "${BLUE}[10/10] Verificando servidor...${NC}"
if pgrep -f "server.py" > /dev/null; then
    echo -e "  ${GREEN}✅ Servidor corriendo (PID: $(pgrep -f server.py))${NC}"

    # Intentar hacer request HTTP
    if command -v curl &> /dev/null; then
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000 2>/dev/null)
        if [ "$HTTP_CODE" = "200" ]; then
            echo -e "  ${GREEN}✅ Servidor responde en puerto 8000${NC}"
        else
            echo -e "  ${YELLOW}⚠️  Servidor corriendo pero no responde (HTTP $HTTP_CODE)${NC}"
        fi
    fi
else
    echo -e "  ${YELLOW}⚠️  Servidor NO está corriendo${NC}"
    echo -e "  ${BLUE}     Para iniciarlo: python server.py${NC}"
fi
echo ""

################################################################################
# Resumen
################################################################################

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   RESUMEN DEL DIAGNÓSTICO${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ TODOS LOS TESTS PASARON${NC}"
    echo ""
    echo -e "${GREEN}El backend está funcionando correctamente.${NC}"
    echo ""
    echo -e "${YELLOW}Si aún así no te escucha, el problema está en el FRONTEND:${NC}"
    echo ""
    echo "1. Abre la consola del navegador (F12)"
    echo "2. Busca errores en la pestaña Console"
    echo "3. Verifica que accedes via HTTPS (no HTTP)"
    echo "4. Verifica permisos de micrófono"
    echo ""
    echo -e "${BLUE}Ver guía completa: DEBUG_AUDIO.md${NC}"
else
    echo -e "${RED}❌ ENCONTRADOS $ERRORS ERRORES${NC}"
    echo ""
    echo "Arregla los errores marcados con ❌ arriba."
    echo ""
    echo "Errores comunes:"
    echo "  • GROQ_API_KEY no configurado → Edita .env"
    echo "  • Voces no encontradas → Verifica /workspace/VibeVoice"
    echo "  • TTS falló → Ejecuta: python test_vibevoice.py"
    echo ""
fi

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

exit $ERRORS
