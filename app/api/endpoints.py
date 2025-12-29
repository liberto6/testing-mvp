import time
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.logging import logger, metrics_logger
from app.services.stt import run_stt
from app.services.llm import stream_sentences
from app.services.tts import run_tts

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🚀 Pipeline Streaming conectado.")
    
    try:
        while True:
            # 1. RECIBIR DATOS (Texto o Audio) - Fase 3: Hybrid Input
            message = await websocket.receive()
            
            user_text = ""
            t_start_pipeline = time.time()
            t_stt = 0
            
            # Caso A: Texto (Web Speech API) - Ultrarrápido
            if "text" in message:
                user_text = message["text"]
                logger.info(f"📨 Recibido TEXTO directo: '{user_text}'")
                if not user_text.strip(): continue

            # Caso B: Audio (Fallback / Legacy) - Lento (STT CPU)
            elif "bytes" in message:
                data = message["bytes"]
                if len(data) == 0: continue
                
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Verificar si hay señal
                amplitude = np.max(np.abs(audio_np))
                if amplitude < 0.01:
                    logger.debug("Silencio detectado, ignorando.")
                    continue

                logger.info("🎤 Procesando AUDIO de usuario (Whisper CPU)...")
                
                # STT
                t0 = time.time()
                user_text = await run_stt(audio_np)
                t_stt = time.time() - t0
                
                if not user_text:
                    logger.warning("STT no detectó texto.")
                    continue
            
            # Si no hay texto válido por ninguna vía
            if not user_text: continue

            # 3. PIPELINE LLM -> TTS (Streaming)
            logger.info(f"📝 Usuario: '{user_text}' (STT: {t_stt:.2f}s)")
            
            # Métricas acumuladas para el reporte final de esta interacción
            interaction_metrics = {
                "stt_time": t_stt,
                "sentences": []
            }

            t_llm_start = time.time()
            first_audio_sent = False
            t_first_byte = 0
            
            async for sentence in stream_sentences(user_text):
                t_sent_gen = time.time()
                
                # Generar audio
                t_tts_start = time.time()
                audio_bytes = await run_tts(sentence)
                t_tts_dur = time.time() - t_tts_start
                
                if audio_bytes:
                    await websocket.send_bytes(audio_bytes)
                    
                    if not first_audio_sent:
                        t_first_byte = time.time() - t_start_pipeline
                        first_audio_sent = True
                    
                    # Registrar métricas de esta frase
                    sent_metric = {
                        "text": sentence[:30] + "..." if len(sentence) > 30 else sentence,
                        "chars": len(sentence),
                        "tts_time": t_tts_dur,
                        "audio_size": len(audio_bytes)
                    }
                    interaction_metrics["sentences"].append(sent_metric)
                    logger.info(f"  📤 Enviado: '{sentence[:20]}...' | TTS: {t_tts_dur:.2f}s")
                else:
                    logger.error(f"  ❌ Falló TTS para: '{sentence[:20]}...'")

            # Reporte Final Visual
            total_time = time.time() - t_start_pipeline
            
            report = f"""
🎯 INTERACTION REPORT
   Total Latency (End-to-End): {total_time:.2f}s
   Time to First Audio (TTFA): {t_first_byte:.2f}s {'⚡ FAST' if t_first_byte < 1.5 else '🐢 SLOW'}
   
   [STT] Whisper (CPU): {interaction_metrics['stt_time']:.2f}s
   
   [TTS Pipeline Breakdown]
   {'#':<3} | {'Text Segment':<30} | {'Chars':<5} | {'TTS Time':<8} | {'Speed (ms/char)':<15}
   {'-'*75}"""
            
            for i, m in enumerate(interaction_metrics["sentences"]):
                speed = (m['tts_time'] * 1000) / m['chars'] if m['chars'] > 0 else 0
                report += f"\n   {i+1:<3} | {m['text']:<30} | {m['chars']:<5} | {m['tts_time']:.2f}s   | {speed:.0f} ms/char"
            
            metrics_logger.info(report)

    except WebSocketDisconnect:
        logger.info("🔌 Cliente desconectado.")
    except Exception as e:
        logger.error(f"🔥 Error Crítico Pipeline: {e}", exc_info=True)
