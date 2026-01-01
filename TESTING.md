# Testing Guide - Pipecat Voice Pipeline

## 🧪 Testing Overview

This guide covers how to test the Pipecat voice pipeline implementation.

---

## Quick Test Commands

### 1. Unit Tests (Individual Components)

```bash
# Test GPU detection
python -c "from src.utils.gpu_utils import gpu_manager; print(gpu_manager.get_gpu_config())"

# Test configuration
python -c "from src.utils.config import ConfigManager; cm = ConfigManager(auto_optimize=True); cm.print_config()"

# Test Whisper processor
python src/processors/stt_whisper_gpu.py

# Test Groq LLM processor
python src/processors/llm_groq.py

# Test Kokoro TTS processor
python src/processors/tts_kokoro.py

# Test Edge TTS processor
python src/processors/tts_edge.py

# Test full pipeline
python src/pipeline.py
```

### 2. Integration Tests

```bash
# Start server
python src/main.py

# In another terminal, test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/config
curl http://localhost:8000/

# Test with WebSocket client (see below)
```

### 3. Performance Tests

```bash
# Run comprehensive GPU benchmark
python tests/benchmark/gpu_benchmark.py

# View results
cat benchmark_results.json
```

---

## Detailed Testing

### GPU Detection Test

```bash
python3 << 'EOF'
from src.utils.gpu_utils import gpu_manager

print("\n🎮 GPU Detection Test\n")

if gpu_manager.device.type == "cuda":
    print("✅ CUDA detected")
    print(f"   GPU: {gpu_manager.capabilities.name}")
    print(f"   VRAM: {gpu_manager.capabilities.total_memory_gb:.1f}GB")
    print(f"   Compute: {gpu_manager.capabilities.compute_capability}")
    print(f"   FP16: {gpu_manager.capabilities.supports_fp16}")
    print(f"   Flash Attention: {gpu_manager.capabilities.supports_flash_attention}")

    # Memory stats
    stats = gpu_manager.get_memory_stats()
    print(f"\n📊 Memory:")
    print(f"   Free: {stats['free_gb']:.1f}GB")
    print(f"   Allocated: {stats['allocated_gb']:.1f}GB")

    # NVIDIA-SMI
    nvidia = gpu_manager.get_nvidia_smi_stats()
    if nvidia:
        print(f"\n🔥 GPU Stats:")
        print(f"   Utilization: {nvidia['gpu_utilization_percent']}%")
        print(f"   Temperature: {nvidia['temperature_celsius']}°C")
        print(f"   Power: {nvidia['power_draw_watts']}W")

    # Optimal config
    config = gpu_manager.get_optimal_whisper_model()
    print(f"\n💡 Recommended Whisper model: {config}")

else:
    print("⚠️ No CUDA GPU detected, will use CPU")

print("\n✅ GPU detection test complete\n")
EOF
```

### STT Processor Test

```bash
python3 << 'EOF'
import asyncio
import numpy as np
from src.processors.stt_whisper_gpu import WhisperGPUProcessor, WhisperGPUConfig
from pipecat.frames.frames import AudioRawFrame

async def test_stt():
    print("\n🎤 STT Processor Test\n")

    # Create config
    config = WhisperGPUConfig(
        model_size="tiny",  # Use tiny for fast testing
        device="cuda",
        compute_type="float16"
    )

    # Create processor
    processor = WhisperGPUProcessor(config=config)

    # Generate test audio (sine wave representing speech)
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Convert to int16 bytes
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()

    # Create frame
    audio_frame = AudioRawFrame(
        audio=audio_bytes,
        sample_rate=sample_rate,
        num_channels=1
    )

    print("Processing audio...")

    # Process
    async for frame in processor.process_frame(audio_frame):
        if hasattr(frame, 'text'):
            print(f"✅ Transcription: '{frame.text}'")

    # Metrics
    metrics = processor.get_metrics()
    print(f"\n📊 Metrics:")
    print(f"   Transcriptions: {metrics['transcription_count']}")
    print(f"   Average RTF: {metrics['average_rtf']:.2f}x")

    # Cleanup
    await processor.cleanup()

    print("\n✅ STT test complete\n")

asyncio.run(test_stt())
EOF
```

### LLM Processor Test

```bash
# Make sure GROQ_API_KEY is set
export GROQ_API_KEY=your_key_here

python3 << 'EOF'
import asyncio
import os
from src.processors.llm_groq import GroqLLMProcessor, GroqConfig
from pipecat.frames.frames import TextFrame

async def test_llm():
    print("\n🤖 LLM Processor Test\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not set")
        return

    # Create config
    config = GroqConfig(
        api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=50
    )

    # Create processor
    processor = GroqLLMProcessor(
        config=config,
        system_prompt="You are a helpful assistant. Be brief."
    )

    # Test query
    user_text = "What is the capital of France?"
    print(f"User: {user_text}\n")
    print("Assistant chunks:")

    text_frame = TextFrame(text=user_text)

    # Process
    async for frame in processor.process_frame(text_frame):
        if isinstance(frame, TextFrame):
            print(f"  > '{frame.text}'")

    # Metrics
    metrics = processor.get_metrics()
    print(f"\n📊 Metrics:")
    print(f"   Requests: {metrics['request_count']}")
    print(f"   Avg latency: {metrics['average_latency']:.2f}s")

    print("\n✅ LLM test complete\n")

asyncio.run(test_llm())
EOF
```

### TTS Processor Test

```bash
python3 << 'EOF'
import asyncio
from src.processors.tts_kokoro import KokoroTTSProcessor, KokoroConfig
from src.processors.tts_edge import EdgeTTSProcessor, EdgeTTSConfig
from pipecat.frames.frames import TextFrame

async def test_tts():
    print("\n🔊 TTS Processor Test\n")

    # Test Kokoro (GPU)
    print("Testing Kokoro TTS...")
    try:
        config = KokoroConfig(
            voice="af_sarah",
            device="cuda"
        )
        processor = KokoroTTSProcessor(config=config)

        text_frame = TextFrame(text="Hello, this is a test of Kokoro text to speech.")

        async for frame in processor.process_frame(text_frame):
            if hasattr(frame, 'audio'):
                print(f"✅ Kokoro: Generated {len(frame.audio)} bytes")

        metrics = processor.get_metrics()
        print(f"   Speed: {metrics['chars_per_second']:.0f} chars/s")

        await processor.cleanup()

    except Exception as e:
        print(f"⚠️ Kokoro failed: {e}")

    # Test Edge TTS (CPU fallback)
    print("\nTesting Edge TTS...")
    try:
        config = EdgeTTSConfig(voice="en-US-JennyNeural")
        processor = EdgeTTSProcessor(config=config)

        text_frame = TextFrame(text="Hello, this is a test of Edge text to speech.")

        async for frame in processor.process_frame(text_frame):
            if hasattr(frame, 'audio'):
                print(f"✅ Edge: Generated {len(frame.audio)} bytes")

        metrics = processor.get_metrics()
        print(f"   Speed: {metrics['chars_per_second']:.0f} chars/s")

        await processor.cleanup()

    except Exception as e:
        print(f"❌ Edge TTS failed: {e}")

    print("\n✅ TTS test complete\n")

asyncio.run(test_tts())
EOF
```

### Full Pipeline Test

```bash
python3 << 'EOF'
import asyncio
from src.pipeline import VoicePipeline
from src.utils.config import ConfigManager

async def test_pipeline():
    print("\n🔄 Full Pipeline Test\n")

    # Create config
    config_manager = ConfigManager(auto_optimize=True)
    print("Configuration:")
    config_manager.print_config()

    # Create pipeline
    pipeline = VoicePipeline(config_manager=config_manager)

    print("\n✅ Pipeline initialized successfully")

    # Get metrics
    metrics = pipeline.get_metrics()
    print(f"\n📊 Pipeline ready. GPU stats:")
    if 'gpu' in metrics:
        print(f"   VRAM allocated: {metrics['gpu'].get('allocated_gb', 0):.2f}GB")

    # Cleanup
    await pipeline.cleanup()

    print("\n✅ Pipeline test complete\n")

asyncio.run(test_pipeline())
EOF
```

---

## WebSocket Client Test

### Python WebSocket Client

```python
# test_websocket_client.py

import asyncio
import websockets
import json
import numpy as np

async def test_websocket():
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        print("✅ Connected to WebSocket")

        # Test 1: Send text
        print("\n📨 Sending text message...")
        await websocket.send(json.dumps({"text": "Hello, how are you?"}))

        # Receive response
        print("🎧 Receiving response...")

        while True:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10)

                if isinstance(response, bytes):
                    print(f"   🔊 Received audio: {len(response)} bytes")
                else:
                    data = json.loads(response)
                    print(f"   📨 Received message: {data}")

            except asyncio.TimeoutError:
                print("   ⏱️ Response complete")
                break

        print("\n✅ WebSocket test complete")

# Run test
asyncio.run(test_websocket())
```

Run with:
```bash
python test_websocket_client.py
```

### JavaScript WebSocket Client

```html
<!-- test_websocket.html -->
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Test</title>
</head>
<body>
    <h1>Pipecat WebSocket Test</h1>
    <button onclick="testWebSocket()">Test Connection</button>
    <div id="output"></div>

    <script>
        function log(message) {
            document.getElementById('output').innerHTML += message + '<br>';
        }

        function testWebSocket() {
            log('Connecting to WebSocket...');

            const ws = new WebSocket('ws://localhost:8000/ws');

            ws.onopen = () => {
                log('✅ Connected!');

                // Send text message
                ws.send(JSON.stringify({text: "Hello, how are you?"}));
                log('📨 Sent: Hello, how are you?');
            };

            ws.onmessage = (event) => {
                if (event.data instanceof Blob) {
                    log(`🔊 Received audio: ${event.data.size} bytes`);
                } else {
                    const data = JSON.parse(event.data);
                    log(`📨 Received: ${JSON.stringify(data)}`);
                }
            };

            ws.onerror = (error) => {
                log(`❌ Error: ${error}`);
            };

            ws.onclose = () => {
                log('🔌 Disconnected');
            };
        }
    </script>
</body>
</html>
```

---

## Benchmark Testing

### Run Full Benchmark Suite

```bash
python tests/benchmark/gpu_benchmark.py
```

This will:
1. Test all Whisper models (tiny → large-v3)
2. Test different batch sizes
3. Compare TTS providers
4. Generate `benchmark_results.json`

### View Benchmark Results

```bash
# Pretty print JSON
python -m json.tool benchmark_results.json

# Or parse with jq
cat benchmark_results.json | jq '.'
```

### Expected Results (RTX 4090)

```json
{
  "test_name": "Whisper large-v3",
  "model": "large-v3",
  "device": "cuda",
  "compute_type": "float16",
  "duration_seconds": 0.92,
  "throughput": 10.87,
  "rtf": 0.92,
  "vram_used_gb": 8.5,
  "peak_vram_gb": 10.1
}
```

---

## Load Testing

### Simple Load Test

```bash
# Install siege
# macOS: brew install siege
# Linux: apt-get install siege

# Test health endpoint
siege -c 10 -r 100 http://localhost:8000/health

# Results will show:
# - Transactions
# - Availability
# - Response time
# - Throughput
```

### WebSocket Load Test

```python
# load_test_ws.py

import asyncio
import websockets
import json
import time

async def client_session(client_id, duration=30):
    """Simulate one client for specified duration"""
    uri = "ws://localhost:8000/ws"

    start_time = time.time()
    message_count = 0

    try:
        async with websockets.connect(uri) as ws:
            while time.time() - start_time < duration:
                # Send text
                await ws.send(json.dumps({"text": "Hello"}))
                message_count += 1

                # Receive responses
                while True:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(response, bytes):
                            break
                    except asyncio.TimeoutError:
                        break

                await asyncio.sleep(5)  # Wait between messages

    except Exception as e:
        print(f"Client {client_id} error: {e}")

    return message_count

async def load_test(num_clients=5, duration=30):
    """Run load test with multiple concurrent clients"""

    print(f"\n🔥 Load Test: {num_clients} clients for {duration}s\n")

    # Run clients concurrently
    tasks = [
        client_session(i, duration)
        for i in range(num_clients)
    ]

    results = await asyncio.gather(*tasks)

    total_messages = sum(results)
    print(f"\n✅ Completed:")
    print(f"   Total messages: {total_messages}")
    print(f"   Messages/sec: {total_messages / duration:.2f}")
    print(f"   Avg per client: {total_messages / num_clients:.1f}")

# Run
asyncio.run(load_test(num_clients=5, duration=30))
```

---

## Continuous Testing

### Pre-commit Tests

```bash
# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash

echo "Running pre-commit tests..."

# Test GPU detection
python -c "from src.utils.gpu_utils import gpu_manager" || exit 1

# Test configuration
python -c "from src.utils.config import ConfigManager" || exit 1

echo "✅ Pre-commit tests passed"
EOF

chmod +x .git/hooks/pre-commit
```

### CI/CD Tests (GitHub Actions)

```yaml
# .github/workflows/test.yml

name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements-gpu.txt

    - name: Test imports
      run: |
        python -c "from src.utils.gpu_utils import gpu_manager"
        python -c "from src.utils.config import ConfigManager"
        python -c "from src.pipeline import VoicePipeline"

    - name: Test configuration
      run: |
        python -c "from src.utils.config import ConfigManager; cm = ConfigManager(auto_optimize=True)"

    - name: Health check
      run: |
        python src/main.py &
        sleep 10
        curl http://localhost:8000/health
```

---

## Troubleshooting Tests

### Test Fails: "Module not found"

```bash
# Ensure you're in the correct directory
cd /Users/pepeda-rosa/Documents/Verba/RUNPOD/testing-mvp

# Install dependencies
pip install -r requirements-gpu.txt
```

### Test Fails: "CUDA not available"

```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Test Fails: "GROQ_API_KEY not set"

```bash
# Set in environment
export GROQ_API_KEY=your_key_here

# Or add to .env
echo "GROQ_API_KEY=your_key_here" >> .env
```

---

## ✅ Test Checklist

Before deployment, verify:

- [ ] GPU detection working
- [ ] Configuration loads correctly
- [ ] STT processor initializes
- [ ] LLM processor connects to Groq
- [ ] TTS processor generates audio
- [ ] WebSocket accepts connections
- [ ] Health endpoint responds
- [ ] Benchmark runs successfully
- [ ] Frontend connects and works
- [ ] Docker build succeeds

---

**Happy Testing!** 🧪
