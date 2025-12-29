import time
import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.logging import logger, metrics_logger
from app.core.config import SYSTEM_PROMPT
from app.services.stt import run_stt
from app.services.llm import stream_sentences
from app.services.tts import run_tts

router = APIRouter()

async def process_pipeline(websocket: WebSocket, message: dict, chat_history: list):
    """Procesa una interacción completa: STT -> LLM -> TTS"""
    full_ai_response = ""
    try:
        user_text = ""
        t_start_pipeline = time.time()
        t_stt = 0
        
        # Caso A: Texto (Web Speech API) - Ultrarrápido
        if "text" in message:
            user_text = message["text"]
            logger.info(f"📨 Recibido TEXTO directo: '{user_text}'")
            if not user_text.strip(): return

        # Caso B: Audio (Fallback / Legacy) - Lento (STT CPU)
        elif "bytes" in message:
            data = message["bytes"]
            if len(data) == 0: return
            
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Verificar si hay señal
            amplitude = np.max(np.abs(audio_np))
            if amplitude < 0.01:
                logger.debug("Silencio detectado, ignorando.")
                return

            logger.info("🎤 Procesando AUDIO de usuario (Whisper CPU)...")
            
            # STT
            t0 = time.time()
            user_text = await run_stt(audio_np)
            t_stt = time.time() - t0
            
            if not user_text:
                logger.warning("STT no detectó texto.")
                return
        
        # Si no hay texto válido por ninguna vía
        if not user_text: return

        # --- ACTUALIZAR HISTORIAL (Usuario) ---
        chat_history.append({"role": "user", "content": user_text})

        # 3. PIPELINE LLM -> TTS (Streaming)
        logger.info(f"📝 Usuario: '{user_text}' (STT: {t_stt:.2f}s)")
        
        await websocket.send_json({"type": "response_start"})
        
        # Métricas acumuladas para el reporte final de esta interacción
        interaction_metrics = {
            "stt_time": t_stt,
            "sentences": []
        }

        t_llm_start = time.time()
        first_audio_sent = False
        t_first_byte = 0
        
        async for sentence in stream_sentences(chat_history):
            t_sent_gen = time.time()
            full_ai_response += sentence + " "
            
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

        # --- ACTUALIZAR HISTORIAL (Asistente - Completado) ---
        if full_ai_response.strip():
            chat_history.append({"role": "assistant", "content": full_ai_response.strip()})

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
        
    except asyncio.CancelledError:
        logger.warning("🛑 Pipeline cancelado por interrupción del usuario.")
        # --- ACTUALIZAR HISTORIAL (Asistente - Interrumpido) ---
        if full_ai_response.strip():
            chat_history.append({"role": "assistant", "content": full_ai_response.strip() + " ...[Interrupted]"})
        raise  # Re-lanzar para que asyncio maneje la cancelación correctamente

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🚀 Pipeline Streaming conectado.")
    
    # Inicializar historial con System Prompt
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    current_task = None
    
    try:
        while True:
            # 1. RECIBIR DATOS (Texto o Audio)
            # await websocket.receive() espera hasta que llegue un mensaje
            message = await websocket.receive()
            
            # Si ya hay una tarea procesando, la cancelamos (Barge-in / Interrupción)
            if current_task and not current_task.done():
                logger.info("✋ Interrupción detectada! Cancelando respuesta anterior...")
                current_task.cancel()
                try:
                    await current_task
                except asyncio.CancelledError:
                    pass # Esperar a que termine de cancelar
            
            # Crear nueva tarea para procesar el nuevo input
            current_task = asyncio.create_task(process_pipeline(websocket, message, chat_history))

    except WebSocketDisconnect:
        logger.info("🔌 Cliente desconectado.")
        if current_task: current_task.cancel()
    except Exception as e:
        logger.error(f"🔥 Error Crítico Pipeline: {e}", exc_info=True)
