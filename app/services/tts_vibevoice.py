"""
VibeVoice TTS Integration Module

Este módulo integra el sistema TTS de Microsoft VibeVoice en el pipeline de Verba.
VibeVoice es un modelo de síntesis de voz en tiempo real que soporta streaming
y genera audio de alta calidad a 24kHz.

Características:
- Detección automática de dispositivo (CUDA/MPS/CPU)
- Gestión de voces pre-embedidas (.pt files)
- Streaming de audio para baja latencia
- Conversión automática de formato (24kHz mono float32 -> 16-bit PCM WAV)
- Manejo robusto de errores (CUDA OOM, fallos de carga)
- Optimizaciones de rendimiento (ddpm_inference_steps=5, flash_attention si disponible)

Modelo: microsoft/VibeVoice-Realtime-0.5B
Audio Output: 24kHz, Mono, Float32 -> Int16 PCM WAV
"""

import io
import os
import traceback
import copy
from typing import Optional, Dict, Any
import numpy as np
import scipy.io.wavfile as wavfile
import torch

from app.core.logging import logger

# Lazy imports for VibeVoice (se cargan solo cuando se necesita)
# from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
# from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

# Global instances
_vibevoice_pipeline = None
_voice_cache: Dict[str, Any] = {}


class VibeVoiceTTS:
    """
    Wrapper reutilizable para VibeVoice TTS.

    Encapsula toda la lógica de inicialización, gestión de voces,
    síntesis de audio y conversión de formato.
    """

    def __init__(
        self,
        model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
        voices_dir: str = "/Users/pepeda-rosa/Documents/Verba/Repositorios/MicrosoftVibeVoice/VibeVoice/demo/voices/streaming_model/",
        device: Optional[str] = None,
        cfg_scale: float = 1.5,
        ddpm_steps: int = 5,
    ):
        """
        Inicializa el sistema VibeVoice TTS.

        Args:
            model_path: Ruta al modelo de HuggingFace
            voices_dir: Directorio con archivos .pt de voces
            device: Dispositivo ('cuda', 'mps', 'cpu' o None para auto-detección)
            cfg_scale: Classifier-Free Guidance scale (1.0-3.0, default 1.5)
            ddpm_steps: Número de pasos de difusión (5 recomendado para streaming)
        """
        self.model_path = model_path
        self.voices_dir = voices_dir
        self.cfg_scale = cfg_scale
        self.ddpm_steps = ddpm_steps

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            # Normalizar 'mpx' a 'mps'
            if device.lower() == "mpx":
                logger.warning("Device 'mpx' detected, treating as 'mps'")
                device = "mps"
            self.device = device

        # Validar MPS disponibilidad
        if self.device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available. Falling back to CPU.")
            self.device = "cpu"

        logger.info(f"VibeVoice TTS initializing on device: {self.device}")

        # Lazy loading attributes
        self.processor = None
        self.model = None
        self.voice_mapper = None
        self._initialized = False

    def _lazy_init(self):
        """Inicializa el modelo y procesador solo cuando se necesita (lazy loading)."""
        if self._initialized:
            return

        try:
            logger.info(f"Loading VibeVoice model from {self.model_path}")

            # Import VibeVoice modules
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                VibeVoiceStreamingForConditionalGenerationInference
            )
            from vibevoice.processor.vibevoice_streaming_processor import (
                VibeVoiceStreamingProcessor
            )

            # Load processor
            self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

            # Determine dtype and attention implementation based on device
            if self.device == "mps":
                load_dtype = torch.float32  # MPS requires float32
                attn_impl = "sdpa"  # flash_attention_2 not supported on MPS
            elif self.device == "cuda":
                load_dtype = torch.bfloat16
                attn_impl = "flash_attention_2"
            else:  # cpu
                load_dtype = torch.float32
                attn_impl = "sdpa"

            logger.info(f"Using torch_dtype: {load_dtype}, attn_implementation: {attn_impl}")

            # Load model with device-specific logic
            try:
                if self.device == "mps":
                    self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=load_dtype,
                        attn_implementation=attn_impl,
                        device_map=None,  # Load then move
                    )
                    self.model.to("mps")
                elif self.device == "cuda":
                    self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=load_dtype,
                        device_map="cuda",
                        attn_implementation=attn_impl,
                    )
                else:  # cpu
                    self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=load_dtype,
                        device_map="cpu",
                        attn_implementation=attn_impl,
                    )
            except Exception as e:
                # Fallback to SDPA if flash_attention_2 fails
                if attn_impl == 'flash_attention_2':
                    logger.error(f"Flash Attention 2 failed: {e}")
                    logger.warning("Falling back to SDPA. Note: Flash Attention 2 is recommended for best quality.")
                    self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=load_dtype,
                        device_map=(self.device if self.device in ("cuda", "cpu") else None),
                        attn_implementation='sdpa'
                    )
                    if self.device == "mps":
                        self.model.to("mps")
                else:
                    raise

            # Set model to eval mode
            self.model.eval()

            # Configure DDPM inference steps
            self.model.set_ddpm_inference_steps(num_steps=self.ddpm_steps)

            # Log attention implementation
            if hasattr(self.model.model, 'language_model'):
                logger.info(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")

            # Initialize voice mapper
            self.voice_mapper = VoiceMapper(self.voices_dir)

            self._initialized = True
            logger.info("✅ VibeVoice TTS initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize VibeVoice TTS: {e}")
            logger.error(traceback.format_exc())
            raise

    def _load_voice(self, voice_name: str) -> Dict[str, Any]:
        """
        Carga y cachea un archivo de voz .pt

        Args:
            voice_name: Nombre de la voz (ej: 'Wayne', 'Sarah')

        Returns:
            Dict con los embeddings de voz pre-computados
        """
        # Check cache first
        if voice_name in _voice_cache:
            logger.debug(f"Voice '{voice_name}' loaded from cache")
            return _voice_cache[voice_name]

        # Get voice path
        voice_path = self.voice_mapper.get_voice_path(voice_name)

        try:
            # Load voice file
            target_device = self.device if self.device != "cpu" else "cpu"
            all_prefilled_outputs = torch.load(
                voice_path,
                map_location=target_device,
                weights_only=False
            )

            # Cache it
            _voice_cache[voice_name] = all_prefilled_outputs
            logger.info(f"Voice '{voice_name}' loaded from {voice_path}")

            return all_prefilled_outputs

        except Exception as e:
            logger.error(f"Failed to load voice '{voice_name}' from {voice_path}: {e}")
            raise

    def synthesize(
        self,
        text: str,
        voice_name: str = "Wayne",
        return_format: str = "wav_bytes",
    ) -> Optional[bytes]:
        """
        Sintetiza texto a audio usando VibeVoice.

        Args:
            text: Texto a sintetizar
            voice_name: Nombre de la voz a usar
            return_format: Formato de salida ('wav_bytes', 'numpy', 'torch')

        Returns:
            Audio en el formato especificado (bytes WAV por defecto)
        """
        # Lazy init
        if not self._initialized:
            self._lazy_init()

        if not text.strip():
            logger.warning("Empty text provided to synthesize()")
            return None

        try:
            # Prepare text
            full_script = text.replace("'", "'").replace('"', '"').replace('"', '"')

            # Load voice
            all_prefilled_outputs = self._load_voice(voice_name)

            # Process input
            inputs = self.processor.process_input_with_cached_prompt(
                text=full_script,
                cached_prompt=all_prefilled_outputs,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move tensors to device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(self.device)

            # Generate audio
            logger.debug(f"Generating audio with cfg_scale={self.cfg_scale}")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=self.cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
            )

            # Extract audio
            if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
                logger.error("No audio output generated")
                return None

            audio_tensor = outputs.speech_outputs[0]  # First batch item

            # Convert to desired format
            if return_format == "torch":
                return audio_tensor
            elif return_format == "numpy":
                return audio_tensor.cpu().numpy()
            elif return_format == "wav_bytes":
                return self._tensor_to_wav_bytes(audio_tensor)
            else:
                raise ValueError(f"Unknown return_format: {return_format}")

        except Exception as e:
            logger.error(f"❌ Error in VibeVoice synthesis: {e}")
            logger.error(traceback.format_exc())
            return None

    def _tensor_to_wav_bytes(self, audio_tensor: torch.Tensor) -> bytes:
        """
        Convierte un tensor de audio a bytes WAV.

        VibeVoice genera audio float32 en rango [-1, 1] a 24kHz mono.
        Lo convertimos a 16-bit PCM WAV para compatibilidad con el pipeline.

        Args:
            audio_tensor: Tensor de audio (1D o 2D)

        Returns:
            Bytes WAV en formato PCM 16-bit, 24kHz, mono
        """
        # Convert to numpy
        audio_np = audio_tensor.cpu().numpy()

        # Ensure 1D
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        # Clip and convert to int16
        # VibeVoice outputs float32 in range [-1, 1]
        audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

        # Write to WAV in memory
        byte_io = io.BytesIO()
        wavfile.write(byte_io, 24000, audio_int16)  # 24kHz sample rate

        return byte_io.getvalue()

    def get_available_voices(self):
        """Retorna lista de voces disponibles."""
        if not self._initialized:
            self._lazy_init()
        return list(self.voice_mapper.available_voices.keys())


class VoiceMapper:
    """
    Maps speaker names to voice file paths.
    Scans the voices directory for .pt files.
    """

    def __init__(self, voices_dir: str):
        self.voices_dir = voices_dir
        self.voice_presets = {}
        self.available_voices = {}
        self._scan_voices()

    def _scan_voices(self):
        """Escanea el directorio de voces y carga los archivos .pt disponibles."""
        if not os.path.exists(self.voices_dir):
            logger.warning(f"Voices directory not found: {self.voices_dir}")
            return

        # Get all .pt files
        pt_files = [
            f for f in os.listdir(self.voices_dir)
            if f.lower().endswith('.pt') and os.path.isfile(os.path.join(self.voices_dir, f))
        ]

        # Create dictionary with filename (without extension) as key
        for pt_file in pt_files:
            name = os.path.splitext(pt_file)[0]
            full_path = os.path.join(self.voices_dir, pt_file)
            self.voice_presets[name] = full_path

        # Sort alphabetically
        self.voice_presets = dict(sorted(self.voice_presets.items()))

        # Filter existing files
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }

        # Create short name mappings (for names like 'en_US-wayne' -> 'wayne')
        new_dict = {}
        for name, path in self.voice_presets.items():
            if '_' in name:
                short_name = name.split('_')[0]
                new_dict[short_name] = path
            if '-' in name:
                short_name = name.split('-')[-1]
                new_dict[short_name] = path

        self.voice_presets.update(new_dict)

        logger.info(f"Found {len(self.available_voices)} voice files in {self.voices_dir}")
        logger.info(f"Available voices: {', '.join(self.available_voices.keys())}")

    def get_voice_path(self, speaker_name: str) -> str:
        """
        Get voice file path for a given speaker name.
        Tries exact match first, then partial matching.
        """
        # Exact match
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]

        # Partial match (case insensitive)
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path

        # Default to first voice if no match
        if self.voice_presets:
            default_voice = list(self.voice_presets.values())[0]
            logger.warning(f"No voice preset found for '{speaker_name}', using default: {default_voice}")
            return default_voice
        else:
            raise ValueError(f"No voices available in {self.voices_dir}")


# ========== PUBLIC API ==========

def init_vibevoice():
    """
    Inicializa el pipeline global de VibeVoice.
    Esta función se llama en el startup de la aplicación.
    """
    global _vibevoice_pipeline

    if _vibevoice_pipeline is None:
        try:
            logger.info("Initializing VibeVoice TTS pipeline...")
            _vibevoice_pipeline = VibeVoiceTTS()
            # Lazy init se hace en el primer synthesize()
            logger.info("✅ VibeVoice TTS ready (lazy init)")
        except Exception as e:
            logger.error(f"❌ Failed to create VibeVoice pipeline: {e}")
            _vibevoice_pipeline = None


def generate_audio_vibevoice(text: str, voice_name: str = "Wayne") -> Optional[bytes]:
    """
    Genera audio usando VibeVoice TTS.

    Esta es la función pública que se usa desde el router TTS.

    Args:
        text: Texto a sintetizar
        voice_name: Nombre de la voz (default: 'Wayne')

    Returns:
        Bytes WAV (24kHz, mono, 16-bit PCM) o None si falla
    """
    global _vibevoice_pipeline

    if _vibevoice_pipeline is None:
        init_vibevoice()

    if _vibevoice_pipeline is None:
        logger.error("VibeVoice pipeline not initialized")
        return None

    try:
        return _vibevoice_pipeline.synthesize(text, voice_name=voice_name)
    except Exception as e:
        logger.error(f"Error generating audio with VibeVoice: {e}")
        return None
