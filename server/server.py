import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from vosk import Model, KaldiRecognizer, SetLogLevel
import uvicorn

SetLogLevel(-1)  # tắt log spam của Vosk

app = FastAPI(title="Vosk STT Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model ─────────────────────────────────────────────────────────────────────
# Đường dẫn tương đối: thư mục model-small nằm cạnh server.py
# MODEL_PATH = Path(__file__).parent / "model-small"
MODEL_PATH = Path(__file__).parent / "model"
SAMPLE_RATE = 16000

model: Model = None

@app.on_event("startup")
async def startup_event():
    """Load model sau khi server đã bind port — không block startup."""
    global model
    loop = asyncio.get_event_loop()
    print(f"\n[INFO] Đang nạp model từ: {MODEL_PATH}")
    try:
        model = await loop.run_in_executor(None, lambda: Model(str(MODEL_PATH)))
        print("✅ [SUCCESS] Model sẵn sàng!\n")
    except Exception as e:
        print(f"❌ [ERROR] Không nạp được model: {e}")
        print(f"[CHECK] Đảm bảo thư mục '{MODEL_PATH}' tồn tại và có các thư mục con: am, graph, conf\n")


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"🔌 [CONNECTED] {websocket.client}")

    if model is None:
        await websocket.send_json({"type": "error", "text": "Model chưa sẵn sàng, thử lại sau vài giây."})
        await websocket.close()
        return

    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(False)  # tắt word-level timestamps cho nhanh hơn

    # Buffer tích lũy — gom đủ 0.5s (16000 samples * 2 bytes = 32000 bytes) rồi mới feed Vosk
    # Tránh gọi AcceptWaveform với chunk quá nhỏ gây rời rạc
    BUFFER_MIN_BYTES = 8000   # ~0.25s ở 16kHz 16-bit mono — đủ nhỏ để partial vẫn nhanh
    audio_buffer = bytearray()

    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)

            # Chỉ feed vào Vosk khi đủ lượng audio tối thiểu
            if len(audio_buffer) < BUFFER_MIN_BYTES:
                continue

            chunk = bytes(audio_buffer)
            audio_buffer.clear()

            is_final = await loop.run_in_executor(
                None, recognizer.AcceptWaveform, chunk
            )

            if is_final:
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    await websocket.send_json({"type": "final", "text": text})
            else:
                partial = json.loads(recognizer.PartialResult())
                text = partial.get("partial", "").strip()
                if text:
                    await websocket.send_json({"type": "partial", "text": text})

    except WebSocketDisconnect:
        print(f"🔌 [DISCONNECTED] {websocket.client}")
    except Exception as e:
        print(f"💥 [ERROR] {e}")


if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8001, reload=False)
