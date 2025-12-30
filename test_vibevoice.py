#!/usr/bin/env python3
"""
Script de prueba para verificar la integración de VibeVoice TTS.

Este script prueba:
1. Importación correcta de módulos
2. Inicialización del modelo
3. Síntesis de audio básica
4. Verificación de formato de salida

Uso:
    python test_vibevoice.py [--text "Texto a sintetizar"] [--voice Wayne] [--output test_output.wav]
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.tts_vibevoice import VibeVoiceTTS, init_vibevoice, generate_audio_vibevoice
from app.core.logging import setup_logging, logger


def test_import():
    """Test 1: Verificar imports"""
    print("=" * 60)
    print("TEST 1: Verificando imports de VibeVoice...")
    print("=" * 60)

    try:
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor
        )
        print("✅ Imports exitosos")
        return True
    except ImportError as e:
        print(f"❌ Error en imports: {e}")
        print("\nAsegúrate de haber instalado las dependencias:")
        print("  pip install -r requirements.txt")
        print("\nY que el repositorio de VibeVoice esté disponible:")
        print("  git clone https://github.com/microsoft/VibeVoice")
        print("  pip install -e VibeVoice")
        return False


def test_initialization():
    """Test 2: Inicialización del modelo"""
    print("\n" + "=" * 60)
    print("TEST 2: Inicializando VibeVoice TTS...")
    print("=" * 60)

    try:
        tts = VibeVoiceTTS()
        print(f"Device detectado: {tts.device}")
        print(f"CFG Scale: {tts.cfg_scale}")
        print(f"DDPM Steps: {tts.ddpm_steps}")
        print("✅ Instancia creada (lazy init pendiente)")
        return tts
    except Exception as e:
        print(f"❌ Error en inicialización: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_synthesis(tts, text="Hello, this is a test of the VibeVoice text to speech system.", voice="Wayne"):
    """Test 3: Síntesis de audio"""
    print("\n" + "=" * 60)
    print("TEST 3: Sintetizando audio...")
    print("=" * 60)
    print(f"Texto: {text}")
    print(f"Voz: {voice}")

    try:
        start_time = time.time()
        wav_bytes = tts.synthesize(text, voice_name=voice)
        elapsed = time.time() - start_time

        if wav_bytes is None:
            print("❌ La síntesis retornó None")
            return None

        print(f"✅ Audio generado: {len(wav_bytes)} bytes")
        print(f"⏱️  Tiempo: {elapsed:.2f} segundos")

        # Calculate RTF (Real-Time Factor)
        # Assuming 24kHz sample rate and 16-bit samples
        num_samples = (len(wav_bytes) - 44) // 2  # WAV header is 44 bytes, 2 bytes per sample
        audio_duration = num_samples / 24000
        rtf = elapsed / audio_duration if audio_duration > 0 else float('inf')
        print(f"📊 Duración audio: {audio_duration:.2f}s")
        print(f"📊 RTF (Real-Time Factor): {rtf:.2f}x")

        return wav_bytes

    except Exception as e:
        print(f"❌ Error en síntesis: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_save_audio(wav_bytes, output_path="test_vibevoice_output.wav"):
    """Test 4: Guardar audio a archivo"""
    print("\n" + "=" * 60)
    print("TEST 4: Guardando audio...")
    print("=" * 60)

    try:
        with open(output_path, 'wb') as f:
            f.write(wav_bytes)

        file_size = os.path.getsize(output_path)
        print(f"✅ Audio guardado: {output_path}")
        print(f"📁 Tamaño: {file_size} bytes")
        print(f"\nPuedes reproducirlo con:")
        print(f"  ffplay {output_path}")
        print(f"  o abrirlo en cualquier reproductor de audio")

        return True

    except Exception as e:
        print(f"❌ Error guardando audio: {e}")
        return False


def test_api_function(text="Testing the API function.", voice="Wayne"):
    """Test 5: Probar la función pública de API"""
    print("\n" + "=" * 60)
    print("TEST 5: Probando función API generate_audio_vibevoice()...")
    print("=" * 60)

    try:
        # Initialize global pipeline
        init_vibevoice()

        # Generate audio
        start_time = time.time()
        wav_bytes = generate_audio_vibevoice(text, voice_name=voice)
        elapsed = time.time() - start_time

        if wav_bytes is None:
            print("❌ La función API retornó None")
            return None

        print(f"✅ Audio generado: {len(wav_bytes)} bytes")
        print(f"⏱️  Tiempo: {elapsed:.2f} segundos")

        return wav_bytes

    except Exception as e:
        print(f"❌ Error en función API: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Test VibeVoice TTS Integration")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the VibeVoice text to speech system.",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="Wayne",
        help="Voice to use (Wayne, Sarah, etc.)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_vibevoice_output.wav",
        help="Output WAV file path"
    )
    parser.add_argument(
        "--skip-api-test",
        action="store_true",
        help="Skip API function test"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    print("\n" + "🎤" * 30)
    print("VIBEVOICE TTS INTEGRATION TEST")
    print("🎤" * 30)

    # Run tests
    success = True

    # Test 1: Imports
    if not test_import():
        print("\n❌ Test de imports falló. Abortando.")
        return 1

    # Test 2: Initialization
    tts = test_initialization()
    if tts is None:
        print("\n❌ Test de inicialización falló. Abortando.")
        return 1

    # Test 3: Synthesis
    wav_bytes = test_synthesis(tts, text=args.text, voice=args.voice)
    if wav_bytes is None:
        print("\n❌ Test de síntesis falló.")
        success = False
    else:
        # Test 4: Save audio
        if not test_save_audio(wav_bytes, output_path=args.output):
            print("\n❌ Test de guardado falló.")
            success = False

    # Test 5: API function (optional)
    if not args.skip_api_test:
        api_wav_bytes = test_api_function(text=args.text, voice=args.voice)
        if api_wav_bytes is None:
            print("\n❌ Test de función API falló.")
            success = False
        else:
            # Save API output
            api_output = args.output.replace('.wav', '_api.wav')
            test_save_audio(api_wav_bytes, output_path=api_output)

    # Summary
    print("\n" + "=" * 60)
    print("RESUMEN DE TESTS")
    print("=" * 60)

    if success:
        print("✅ Todos los tests pasaron exitosamente!")
        print("\n🎉 VibeVoice TTS está correctamente integrado.")
        print("\nPróximos pasos:")
        print("1. Configura TTS_ENGINE=vibevoice en tu .env")
        print("2. Reinicia el servidor: python server.py")
        print("3. El sistema usará VibeVoice automáticamente")
        return 0
    else:
        print("❌ Algunos tests fallaron. Revisa los errores arriba.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
