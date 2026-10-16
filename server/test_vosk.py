# """
# Test Vosk với audio normalization
# """
# import os, json, wave, numpy as np
# import sounddevice as sd
# import vosk

# SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH  = os.path.join(SCRIPT_DIR, "model")
# SAMPLE_RATE = 16000
# DURATION    = 5

# print("Loading model...")
# vosk.SetLogLevel(-1)
# model = vosk.Model(MODEL_PATH)
# print("Model loaded. Ghi âm 5 giây... HÃY NÓI NGAY!")

# audio = sd.rec(int(DURATION * SAMPLE_RATE),
#                samplerate=SAMPLE_RATE, channels=1, dtype="int16")
# sd.wait()

# flat = audio.flatten()
# rms_before = np.sqrt(np.mean(flat.astype(np.float32)**2))
# print(f"RMS gốc: {rms_before:.0f}")

# # Normalize lên target RMS 3000 (Vosk cần ~1000-5000)
# target_rms = 3000
# if rms_before > 0:
#     gain = target_rms / rms_before
#     amplified = np.clip(flat.astype(np.float32) * gain, -32767, 32767).astype(np.int16)
# else:
#     amplified = flat

# rms_after = np.sqrt(np.mean(amplified.astype(np.float32)**2))
# print(f"RMS sau normalize (gain={gain:.1f}x): {rms_after:.0f}")

# # Test không normalize
# print("\n--- KHÔNG normalize ---")
# rec1 = vosk.KaldiRecognizer(model, SAMPLE_RATE)
# data = flat.tobytes()
# for i in range(0, len(data), 4000):
#     rec1.AcceptWaveform(data[i:i+4000])
# r1 = json.loads(rec1.FinalResult())
# print(f"Kết quả: '{r1.get('text', '')}'")

# # Test có normalize
# print("\n--- CÓ normalize ---")
# rec2 = vosk.KaldiRecognizer(model, SAMPLE_RATE)
# data2 = amplified.tobytes()
# for i in range(0, len(data2), 4000):
#     rec2.AcceptWaveform(data2[i:i+4000])
# r2 = json.loads(rec2.FinalResult())
# print(f"Kết quả: '{r2.get('text', '')}'")

# """
# Test Vosk độc lập — ghi âm 5 giây rồi decode
# Chạy: python test_vosk.py
# """
# import os, json, wave, struct, numpy as np
# import sounddevice as sd
# import vosk

# SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH  = os.path.join(SCRIPT_DIR, "model")
# SAMPLE_RATE = 16000
# DURATION    = 5  # giây

# print("Loading model...")
# vosk.SetLogLevel(0)  # bật log Vosk để xem model load có lỗi không
# model = vosk.Model(MODEL_PATH)
# print("Model loaded.")

# print(f"\nGhi âm {DURATION} giây... HÃY NÓI NGAY!")
# audio = sd.rec(int(DURATION * SAMPLE_RATE),
#                samplerate=SAMPLE_RATE, channels=1, dtype="int16")
# sd.wait()
# print("Xong ghi âm.")

# # In thông tin audio
# flat = audio.flatten()
# print(f"Audio stats: min={flat.min()} max={flat.max()} "
#       f"rms={np.sqrt(np.mean(flat.astype(np.float32)**2)):.0f}")

# # Lưu wav để kiểm tra thủ công
# wav_path = os.path.join(SCRIPT_DIR, "test_record.wav")
# with wave.open(wav_path, "wb") as wf:
#     wf.setnchannels(1)
#     wf.setsampwidth(2)
#     wf.setframerate(SAMPLE_RATE)
#     wf.writeframes(flat.tobytes())
# print(f"Saved: {wav_path}")

# # Decode
# rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
# rec.SetWords(True)

# chunk_size = 4000
# data = flat.tobytes()
# for i in range(0, len(data), chunk_size):
#     rec.AcceptWaveform(data[i:i+chunk_size])

# result = json.loads(rec.FinalResult())
# print(f"\nKết quả: '{result.get('text', '')}'")
# if not result.get("text"):
#     print("⚠ Không có text — kiểm tra file test_record.wav bằng trình phát nhạc")
#     print("  Nếu file có giọng nói rõ → model bị lỗi")
#     print("  Nếu file rỗng/nhiễu → mic bị lỗi")