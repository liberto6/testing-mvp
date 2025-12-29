import re
from groq import AsyncGroq
from app.core.config import GROQ_API_KEY, SYSTEM_PROMPT
from app.core.logging import logger

if not GROQ_API_KEY:
    print("⚠️ ADVERTENCIA: No se encontró la variable de entorno GROQ_API_KEY.")
    
client = AsyncGroq(api_key=GROQ_API_KEY)

async def stream_sentences(user_text):
    """
    Genera respuesta del LLM y cede fragmentos de texto lo más rápido posible.
    Estrategia: Streaming agresivo (Chunking por puntuación y longitud).
    """
    try:
        completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            model="llama-3.1-8b-instant", # Fase 2: Modelo más rápido (Actualizado)
            stream=True
        )

        buffer = ""
        MIN_CHUNK_LENGTH = 20  # Mínimo caracteres para cortar en coma (evita cortes robóticos en "Yes,")

        async for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                buffer += content
                
                while True:
                    # Buscamos el primer delimitador (fuerte o débil)
                    # [.!?] -> Fuerte
                    # [,;:\—\n] -> Débil
                    match = re.search(r'([.?!])|([,;:\—\n])', buffer)
                    
                    if not match:
                        break
                    
                    delimiter = match.group(0) # El caracter que hizo match
                    end_idx = match.end()      # Índice justo después del delimitador
                    
                    candidate = buffer[:end_idx].strip()
                    
                    # Si quedó vacío (ej: "...")
                    if not candidate:
                        buffer = buffer[end_idx:]
                        continue

                    # Verificar condiciones
                    is_strong = delimiter in ".!?"
                    is_long_enough = len(candidate) >= MIN_CHUNK_LENGTH
                    
                    # Cortamos SI: Es fuerte O (Es débil Y es suficientemente largo)
                    if is_strong or is_long_enough:
                        yield candidate
                        buffer = buffer[end_idx:]
                    else:
                        # Si es débil y corto, NO cortamos todavía.
                        # Esperamos a que llegue más texto que pueda:
                        # 1. Convertirlo en algo más largo.
                        # 2. Encontrar un delimitador fuerte más adelante.
                        break
        
        # Rendir lo que quede en el buffer al final
        if buffer.strip():
            yield buffer.strip()

    except Exception as e:
        print(f"⚠️ Error Groq Stream: {e}")
        yield "Sorry, I had an error."
