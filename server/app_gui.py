"""
Russian STT — Vosk real-time, từng từ hiện ngay
Cấu trúc thư mục:
  STT/
  └── server/
      ├── app.py
      └── model/   ← vosk-model-ru-0.42
"""

import os
import threading
import queue
import json
import logging
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox, filedialog
import sounddevice as sd
import vosk
import datetime
import time
import numpy as np

# ─── Debug logger ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(threadName)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("debug.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(),   # in ra terminal luôn
    ]
)
log = logging.getLogger("STT")

# ─── Cấu hình ────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(_SCRIPT_DIR, "model")  # vosk-model-ru-0.42 (model lớn)
SAMPLE_RATE = 16000
BLOCK_MS    = 80
BLOCK_SIZE  = int(SAMPLE_RATE * BLOCK_MS / 1000)

# ─── Màu sắc ─────────────────────────────────────────────────────────────────

C = {
    "bg":        "#0f1117",
    "surface":   "#1a1d27",
    "panel":     "#21253a",
    "border":    "#2d3148",
    "text":      "#e8eaf0",
    "muted":     "#5a6070",
    "accent":    "#6c8fff",
    "green":     "#3ecf72",
    "red":       "#ef4444",
    "amber":     "#f59e0b",
    "partial":   "#94a3b8",
    "scrollbar": "#2d3148",
    "scroll_hv": "#4a5080",
}

# ─── Vosk Worker ─────────────────────────────────────────────────────────────

class VoskWorker(threading.Thread):
    """
    Nhận model đã load sẵn — KHÔNG load lại model mỗi lần record.
    Flush khi:
      1. Vosk tự AcceptWaveform = True (câu hoàn chỉnh)
      2. Im lặng >= 400ms  (người dừng nói)
      3. Nói liên tục >= 4s (force flush để không tích lũy quá lâu)
    """
    # Tất cả tính theo số BLOCK (1 block = BLOCK_MS ms)
    SILENCE_BLOCKS  = max(1, int(800  / BLOCK_MS))   # 400ms im lặng → flush
    MAX_SPEECH_BLOCKS = max(1, int(8000 / BLOCK_MS)) # 4s nói liên tục → force flush
    MAX_BACKLOG     = 10   # drop nếu queue > 10 block

    def __init__(self, model: vosk.Model, audio_q: queue.Queue,
                 result_q: queue.Queue):
        super().__init__(daemon=True)
        self.audio_q          = audio_q
        self.result_q         = result_q
        self.running          = True
        self._model           = model   # giữ ref để tạo lại recognizer sau flush
        self.rec              = vosk.KaldiRecognizer(model, SAMPLE_RATE)
        self._silence_count   = 0
        self._speech_count    = 0   # tổng block có tiếng kể từ lần flush cuối
        self._speech_rms_sum  = 0.0
        self._speech_rms_n    = 0

    def _flush(self, reason: str):
        log.debug(f"FLUSH [{reason}] speech_blocks={self._speech_count} silence_blocks={self._silence_count}")
        res  = json.loads(self.rec.FinalResult())
        text = res.get("text", "").strip()
        log.debug(f"FLUSH result: '{text}'")
        # Reset để nhận câu tiếp theo sạch
        self.rec = vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
        self._silence_count  = 0
        self._speech_count   = 0
        self._speech_rms_sum = 0.0
        self._speech_rms_n   = 0
        if text:
            self.result_q.put({"type": "final", "text": text})
        else:
            log.debug("FLUSH: empty result, nothing pushed")

    def run(self):
        log.info("VoskWorker started")
        noise_samples     = []
        block_count       = 0
        silence_threshold = 300
        last_partial      = ""

        while self.running:
            # Drop backlog
            backlog = self.audio_q.qsize()
            if backlog > self.MAX_BACKLOG:
                dropped = 0
                while self.audio_q.qsize() > self.MAX_BACKLOG:
                    try: self.audio_q.get_nowait(); dropped += 1
                    except: break
                log.warning(f"DROP {dropped} blocks (backlog={backlog})")

            try:
                data = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            block_count += 1
            pcm = np.frombuffer(data, dtype=np.int16)
            rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))

            # Calibrate noise floor — chỉ cần 10 block (~0.8s)
            if len(noise_samples) < 10:
                noise_samples.append(rms)
                silence_threshold = 150   # default thấp hơn để không bỏ từ đầu
                if len(noise_samples) == 10:
                    floor = float(np.percentile(noise_samples, 30))
                    silence_threshold = max(80, floor * 2)
                    log.info(f"Noise calibrated: floor={floor:.0f} threshold={silence_threshold:.0f}")

            if block_count % 10 == 0:
                log.debug(f"block={block_count} rms={rms:.0f} thresh={silence_threshold:.0f} queue={self.audio_q.qsize()}")

            if self.rec.AcceptWaveform(data):
                # Câu hoàn chỉnh → final, xóa partial
                res  = json.loads(self.rec.Result())
                text = res.get("text", "").strip()
                log.info(f"FINAL: '{text}'")
                last_partial = ""
                self._silence_count = 0
                self._speech_count  = 0
                if text:
                    self.result_q.put({"type": "final", "text": text})
            else:
                # Partial — in ngay từng từ mới xuất hiện
                res     = json.loads(self.rec.PartialResult())
                partial = res.get("partial", "").strip()
                if partial != last_partial:
                    log.debug(f"PARTIAL: '{partial}'")
                    last_partial = partial
                    self.result_q.put({"type": "partial", "text": partial})

        log.info("VoskWorker stopped")

    def stop(self):
        log.info("VoskWorker.stop() called")
        try:
            res  = json.loads(self.rec.FinalResult())
            text = res.get("text", "").strip()
            if text:
                log.info(f"stop() final flush: '{text}'")
                self.result_q.put({"type": "final", "text": text})
        except Exception as e:
            log.error(f"stop() error: {e}")
        self.running = False


# ─── Custom Scrollbar ─────────────────────────────────────────────────────────

class SlimScrollbar(tk.Canvas):
    W = 7

    def __init__(self, parent, command=None, **kw):
        super().__init__(parent, width=self.W, bg=C["surface"],
                         highlightthickness=0, **kw)
        self._cmd    = command
        self._lo     = 0.0
        self._hi     = 1.0
        self._drag_y = None
        self.bind("<Configure>",       lambda e: self._draw())
        self.bind("<ButtonPress-1>",   self._press)
        self.bind("<B1-Motion>",       self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Enter>",           self._hover_on)
        self.bind("<Leave>",           self._hover_off)

    def set(self, lo, hi):
        self._lo, self._hi = float(lo), float(hi)
        self._draw()

    def _draw(self, hover=False):
        self.delete("all")
        h = self.winfo_height()
        w = self.winfo_width()
        if h < 2 or (self._hi - self._lo) >= 1.0:
            return
        ty  = max(2,   int(self._lo * h))
        by  = min(h-2, max(ty + 20, int(self._hi * h)))
        col = C["scroll_hv"] if hover else C["scrollbar"]
        pts = [2+3,ty, w-2-3,ty, w-2,ty, w-2,ty+3,
               w-2,by-3, w-2,by, w-2-3,by, 2+3,by,
               2,by, 2,by-3, 2,ty+3, 2,ty, 2+3,ty]
        self.create_polygon(pts, smooth=True, fill=col)

    def _press(self, e):   self._drag_y = e.y
    def _release(self, e): self._drag_y = None
    def _hover_on(self, e):  self._draw(hover=True)
    def _hover_off(self, e): self._draw(hover=False)

    def _drag(self, e):
        if self._drag_y is None or not self._cmd:
            return
        h    = self.winfo_height()
        diff = (e.y - self._drag_y) / max(h, 1)
        self._drag_y = e.y
        span = self._hi - self._lo
        new  = max(0.0, min(1.0 - span, self._lo + diff))
        self._cmd("moveto", new)


# ─── App ─────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Russian STT")
        self.geometry("860x660")
        self.minsize(640, 480)
        self.configure(bg=C["bg"])

        # Model load một lần, dùng lại mãi
        self._vosk_model    = None

        self.worker         = None
        self.stream         = None
        # audio_q và result_q tạo một lần, dùng chung
        self.audio_q        = queue.Queue()
        self.result_q       = queue.Queue()
        self.is_recording   = False
        self.is_paused      = False
        self.session_log    = []
        self.word_count     = 0
        self._session_start = 0.0
        self._partial_start = None

        self._setup_fonts()
        self._setup_styles()
        self._build_ui()
        self._load_model_async()
        self.after(30, self._poll_results)

    # ── Fonts & styles ────────────────────────────────────────────────────────

    def _setup_fonts(self):
        self.fn_title = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self.fn_sub   = tkfont.Font(family="Segoe UI", size=9)
        self.fn_body  = tkfont.Font(family="Segoe UI", size=15)
        self.fn_small = tkfont.Font(family="Segoe UI", size=9)
        self.fn_btn   = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.fn_stat  = tkfont.Font(family="Segoe UI", size=13, weight="bold")

    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Vol.Horizontal.TProgressbar",
                    troughcolor=C["panel"], background=C["green"],
                    bordercolor=C["panel"], lightcolor=C["green"],
                    darkcolor=C["green"], thickness=4)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=C["surface"], padx=20, pady=14)
        hdr.pack(fill="x")

        left = tk.Frame(hdr, bg=C["surface"])
        left.pack(side="left")
        tk.Label(left, text="Russian STT", font=self.fn_title,
                 bg=C["surface"], fg=C["text"]).pack(anchor="w")
        tk.Label(left, text="Vosk  ·  real-time  ·  offline  ·  vosk-model-ru-0.42",
                 font=self.fn_sub, bg=C["surface"], fg=C["muted"]).pack(anchor="w")

        right = tk.Frame(hdr, bg=C["surface"])
        right.pack(side="right")
        tk.Label(right, text="MIC", font=self.fn_small,
                 bg=C["surface"], fg=C["muted"]).pack(anchor="e")
        self.vol_var = tk.DoubleVar(value=0)
        ttk.Progressbar(right, variable=self.vol_var, maximum=100,
                        length=120, style="Vol.Horizontal.TProgressbar"
                        ).pack(anchor="e", pady=(3, 0))

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # Status bar
        sbar = tk.Frame(self, bg=C["panel"], padx=16, pady=7)
        sbar.pack(fill="x")
        self._dot = tk.Label(sbar, text="●", font=self.fn_small,
                             bg=C["panel"], fg=C["muted"])
        self._dot.pack(side="left", padx=(0, 7))
        self.status_var = tk.StringVar(value="Đang tải model Vosk…")
        tk.Label(sbar, textvariable=self.status_var, font=self.fn_small,
                 bg=C["panel"], fg=C["muted"], anchor="w").pack(side="left")

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # Toolbar — pack TRƯỚC txt_wrap
        self._build_toolbar()

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # Text area
        txt_wrap = tk.Frame(self, bg=C["bg"], padx=16, pady=12)
        txt_wrap.pack(fill="both", expand=True)

        lbl_row = tk.Frame(txt_wrap, bg=C["bg"])
        lbl_row.pack(fill="x", pady=(0, 6))
        tk.Label(lbl_row, text="Văn bản nhận dạng",
                 font=self.fn_small, bg=C["bg"], fg=C["muted"]).pack(side="left")
        self._word_lbl = tk.Label(lbl_row, text="0 từ",
                                  font=self.fn_small, bg=C["bg"], fg=C["muted"])
        self._word_lbl.pack(side="right")

        box = tk.Frame(txt_wrap, bg=C["surface"],
                       highlightbackground=C["border"], highlightthickness=1)
        box.pack(fill="both", expand=True)

        self.text_area = tk.Text(
            box,
            font=self.fn_body,
            bg=C["surface"], fg=C["text"],
            insertbackground=C["text"],
            selectbackground=C["accent"],
            relief="flat", bd=0,
            wrap=tk.WORD,
            padx=18, pady=14,
            spacing1=2, spacing3=8,
            cursor="arrow",
            state="disabled",
        )
        self.text_area.pack(side="left", fill="both", expand=True)

        sb = SlimScrollbar(box, command=self.text_area.yview)
        sb.pack(side="right", fill="y", padx=(0, 3), pady=6)
        self.text_area.configure(yscrollcommand=sb.set)

        self.text_area.tag_config("final",   foreground=C["text"],  font=self.fn_body)
        self.text_area.tag_config("partial", foreground=C["partial"], font=self.fn_body)
        self.text_area.tag_config("pause",   foreground=C["amber"],   font=self.fn_small)

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=C["surface"], padx=16, pady=10)
        bar.pack(fill="x")

        self.rec_btn = self._btn(bar, "⏺  Ghi âm", C["red"],
                                 self.toggle_recording, state="disabled")
        self.rec_btn.pack(side="left", padx=(0, 6))

        self.pause_btn = self._btn(bar, "⏸  Tạm dừng", C["amber"],
                                   self.toggle_pause, state="disabled")
        self.pause_btn.pack(side="left", padx=(0, 18))

        tk.Frame(bar, bg=C["border"], width=1, height=22).pack(side="left", padx=(0, 18))

        self._btn(bar, "💾  Lưu", C["panel"], self.save_session).pack(side="left", padx=(0, 6))
        self._btn(bar, "🗑  Xóa", C["panel"], self.clear_text ).pack(side="left")

        stat = tk.Frame(bar, bg=C["surface"])
        stat.pack(side="right")
        self._time_var = tk.StringVar(value="00:00")
        tk.Label(stat, textvariable=self._time_var, font=self.fn_stat,
                 bg=C["surface"], fg=C["accent"]).pack()
        tk.Label(stat, text="phiên", font=self.fn_small,
                 bg=C["surface"], fg=C["muted"]).pack()

    def _btn(self, parent, text, bg, cmd, state="normal"):
        b = tk.Button(parent, text=text, font=self.fn_btn,
                      bg=bg, fg=C["text"], activebackground=bg,
                      activeforeground=C["text"], relief="flat",
                      padx=14, pady=6, cursor="hand2",
                      command=cmd, state=state, bd=0, highlightthickness=0)
        def _hl(c): return "#{:02x}{:02x}{:02x}".format(
            *[min(255, int(c.lstrip("#")[i:i+2], 16) + 22) for i in (0, 2, 4)])
        b.bind("<Enter>", lambda e: b.config(bg=_hl(bg)))
        b.bind("<Leave>", lambda e: b.config(bg=bg))
        return b

    # ── Model load — chỉ chạy MỘT LẦN ───────────────────────────────────────

    def _load_model_async(self):
        self.status_var.set("Đang tải model Vosk… (chỉ lần đầu)")
        self._dot.config(fg=C["amber"])

        def _load():
            try:
                vosk.SetLogLevel(-1)
                model = vosk.Model(MODEL_PATH)
                self._vosk_model = model
                self.after(0, self._on_ready)
            except Exception as e:
                err = str(e)
                self.after(0, lambda: self._on_error(err))

        threading.Thread(target=_load, daemon=True).start()

    def _on_ready(self):
        self.status_var.set("Sẵn sàng  ·  Vosk  ·  real-time")
        self._dot.config(fg=C["green"])
        self.rec_btn.config(state="normal")

    def _on_error(self, err):
        self.status_var.set(f"Lỗi tải model: {err[:60]}")
        self._dot.config(fg=C["red"])
        messagebox.showerror("Không tải được model",
                             f"Kiểm tra thư mục:\n{MODEL_PATH}\n\n{err}")

    # ── Recording ─────────────────────────────────────────────────────────────

    def toggle_recording(self):
        if self.is_recording: self.stop_recording()
        else:                  self.start_recording()

    def start_recording(self):
        self.is_recording   = True
        self.is_paused      = False
        self._session_start = time.time()

        # Xả queue cũ còn sót
        while not self.audio_q.empty():
            self.audio_q.get_nowait()
        while not self.result_q.empty():
            self.result_q.get_nowait()

        # Worker mới — chỉ tạo KaldiRecognizer, KHÔNG load model lại
        self.worker = VoskWorker(self._vosk_model, self.audio_q, self.result_q)
        self.worker.start()

        self.rec_btn.config(text="⏹  Dừng", bg=C["green"])
        self.pause_btn.config(state="normal")
        self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
        self._dot.config(fg=C["red"])
        self._tick()

        _cb_count = [0]
        def _cb(indata, frames, t, status):
            if status:
                log.warning(f"sounddevice status: {status}")
            if self.is_paused:
                return
            # Boost cố định 6x — đủ để Vosk nghe, không clip với mic thường
            samples = indata[:, 0].astype(np.float32)
            samples = np.clip(samples * 6.0, -1.0, 1.0)
            pcm     = (samples * 32767).astype("int16").tobytes()
            self.audio_q.put_nowait(pcm)
            _cb_count[0] += 1
            if _cb_count[0] % 25 == 0:  # log mỗi 2s
                log.debug(f"audio_cb: block={_cb_count[0]} queue={self.audio_q.qsize()}")
            rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
            self.after(0, lambda: self.vol_var.set(min(100, rms * 2500)))

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype="float32", blocksize=BLOCK_SIZE,
            callback=_cb,
        )
        self.stream.start()

    def stop_recording(self):
        self.is_recording = False
        self.is_paused    = False
        if self.stream:
            self.stream.stop(); self.stream.close(); self.stream = None
        if self.worker:
            self.worker.stop(); self.worker = None
        self.vol_var.set(0)
        self.rec_btn.config(text="⏺  Ghi âm", bg=C["red"])
        self.pause_btn.config(state="disabled", text="⏸  Tạm dừng", bg=C["amber"])
        self.status_var.set("Đã dừng  ·  Nhấn để ghi âm tiếp")
        self._dot.config(fg=C["muted"])
        self._clear_partial()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="▶  Tiếp tục", bg=C["accent"])
            self.status_var.set("Tạm dừng")
            self._dot.config(fg=C["amber"])
            self.vol_var.set(0)
            self._clear_partial()
            self._insert_pause_marker()
        else:
            self.pause_btn.config(text="⏸  Tạm dừng", bg=C["amber"])
            self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
            self._dot.config(fg=C["red"])

    # ── Timer ─────────────────────────────────────────────────────────────────

    def _tick(self):
        if not self.is_recording:
            return
        e = int(time.time() - self._session_start)
        self._time_var.set(f"{e//60:02d}:{e%60:02d}")
        self.after(1000, self._tick)

    # ── Results ───────────────────────────────────────────────────────────────

    def _poll_results(self):
        try:
            while True:
                msg = self.result_q.get_nowait()
                if msg["type"] == "partial":
                    self._show_partial(msg["text"])
                else:
                    self._commit_final(msg["text"])
        except queue.Empty:
            pass
        self.after(30, self._poll_results)

    def _show_partial(self, text: str):
        """Xóa partial cũ, hiện partial mới màu xám."""
        self.text_area.config(state="normal")
        if self._partial_start:
            self.text_area.delete(self._partial_start, "end")
        if text:
            self._partial_start = self.text_area.index("end-1c")
            self.text_area.insert("end", " " + text, "partial")
            self.text_area.see("end")
        else:
            self._partial_start = None
        self.text_area.config(state="disabled")

    def _commit_final(self, text: str):
        """Xóa partial, ghi final màu trắng + xuống dòng."""
        log.info(f"UI commit: '{text}'")
        self.text_area.config(state="normal")
        if self._partial_start:
            self.text_area.delete(self._partial_start, "end")
            self._partial_start = None
        self.text_area.insert("end", text + " ", "final")
        self.text_area.see("end")
        self.text_area.config(state="disabled")
        self.session_log.append(text)
        self.word_count += len(text.split())
        self._word_lbl.config(text=f"{self.word_count} từ")

    def _clear_partial(self):
        if self._partial_start:
            self.text_area.config(state="normal")
            self.text_area.delete(self._partial_start, "end")
            self.text_area.config(state="disabled")
            self._partial_start = None

    def _insert_pause_marker(self):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.text_area.config(state="normal")
        self.text_area.insert("end", f"\n— tạm dừng {ts} —\n", "pause")
        self.text_area.see("end")
        self.text_area.config(state="disabled")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def clear_text(self):
        self.text_area.config(state="normal")
        self.text_area.delete("1.0", "end")
        self.text_area.config(state="disabled")
        self.session_log.clear()
        self.word_count     = 0
        self._partial_start = None
        self._word_lbl.config(text="0 từ")

    def save_session(self):
        if not self.session_log:
            messagebox.showinfo("Thông báo", "Chưa có nội dung để lưu.")
            return
        default = f"session_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=default, title="Lưu phiên ghi âm",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Russian STT — {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
            f.write(f"# Từ: {self.word_count}\n\n")
            f.write(" ".join(self.session_log))
        messagebox.showinfo("Đã lưu", f"Lưu thành công:\n{path}")

    def on_closing(self):
        if self.is_recording:
            self.stop_recording()
        self.destroy()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

# _________________________________ OK 4 __________________________________________ 
# """
# Russian STT — Vosk real-time, từng từ hiện ngay
# Cấu trúc thư mục:
#   STT/
#   └── server/
#       ├── app.py
#       └── model/   ← vosk-model-ru-0.42
# """

# import os
# import threading
# import queue
# import json
# import logging
# import tkinter as tk
# import tkinter.font as tkfont
# from tkinter import ttk, messagebox, filedialog
# import sounddevice as sd
# import vosk
# import datetime
# import time
# import numpy as np

# # ─── Debug logger ─────────────────────────────────────────────────────────────
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s.%(msecs)03d [%(threadName)s] %(levelname)s — %(message)s",
#     datefmt="%H:%M:%S",
#     handlers=[
#         logging.FileHandler("debug.log", encoding="utf-8", mode="w"),
#         logging.StreamHandler(),   # in ra terminal luôn
#     ]
# )
# log = logging.getLogger("STT")

# # ─── Cấu hình ────────────────────────────────────────────────────────────────

# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH  = os.path.join(_SCRIPT_DIR, "model")  # vosk-model-ru-0.42 (model lớn)
# SAMPLE_RATE = 16000
# BLOCK_MS    = 80
# BLOCK_SIZE  = int(SAMPLE_RATE * BLOCK_MS / 1000)

# # ─── Màu sắc ─────────────────────────────────────────────────────────────────

# C = {
#     "bg":        "#0f1117",
#     "surface":   "#1a1d27",
#     "panel":     "#21253a",
#     "border":    "#2d3148",
#     "text":      "#e8eaf0",
#     "muted":     "#5a6070",
#     "accent":    "#6c8fff",
#     "green":     "#3ecf72",
#     "red":       "#ef4444",
#     "amber":     "#f59e0b",
#     "partial":   "#94a3b8",
#     "scrollbar": "#2d3148",
#     "scroll_hv": "#4a5080",
# }

# # ─── Vosk Worker ─────────────────────────────────────────────────────────────

# class VoskWorker(threading.Thread):
#     """
#     Nhận model đã load sẵn — KHÔNG load lại model mỗi lần record.
#     Flush khi:
#       1. Vosk tự AcceptWaveform = True (câu hoàn chỉnh)
#       2. Im lặng >= 400ms  (người dừng nói)
#       3. Nói liên tục >= 4s (force flush để không tích lũy quá lâu)
#     """
#     # Tất cả tính theo số BLOCK (1 block = BLOCK_MS ms)
#     SILENCE_BLOCKS  = max(1, int(800  / BLOCK_MS))   # 400ms im lặng → flush
#     MAX_SPEECH_BLOCKS = max(1, int(8000 / BLOCK_MS)) # 4s nói liên tục → force flush
#     MAX_BACKLOG     = 10   # drop nếu queue > 10 block

#     def __init__(self, model: vosk.Model, audio_q: queue.Queue,
#                  result_q: queue.Queue):
#         super().__init__(daemon=True)
#         self.audio_q          = audio_q
#         self.result_q         = result_q
#         self.running          = True
#         self._model           = model   # giữ ref để tạo lại recognizer sau flush
#         self.rec              = vosk.KaldiRecognizer(model, SAMPLE_RATE)
#         self._silence_count   = 0
#         self._speech_count    = 0   # tổng block có tiếng kể từ lần flush cuối
#         self._speech_rms_sum  = 0.0
#         self._speech_rms_n    = 0

#     def _flush(self, reason: str):
#         log.debug(f"FLUSH [{reason}] speech_blocks={self._speech_count} silence_blocks={self._silence_count}")
#         res  = json.loads(self.rec.FinalResult())
#         text = res.get("text", "").strip()
#         log.debug(f"FLUSH result: '{text}'")
#         # Reset để nhận câu tiếp theo sạch
#         self.rec = vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
#         self._silence_count  = 0
#         self._speech_count   = 0
#         self._speech_rms_sum = 0.0
#         self._speech_rms_n   = 0
#         if text:
#             self.result_q.put({"type": "final", "text": text})
#         else:
#             log.debug("FLUSH: empty result, nothing pushed")

#     def run(self):
#         log.info("VoskWorker started")
#         noise_samples     = []
#         block_count       = 0
#         silence_threshold = 300
#         last_partial      = ""

#         while self.running:
#             # Drop backlog
#             backlog = self.audio_q.qsize()
#             if backlog > self.MAX_BACKLOG:
#                 dropped = 0
#                 while self.audio_q.qsize() > self.MAX_BACKLOG:
#                     try: self.audio_q.get_nowait(); dropped += 1
#                     except: break
#                 log.warning(f"DROP {dropped} blocks (backlog={backlog})")

#             try:
#                 data = self.audio_q.get(timeout=0.1)
#             except queue.Empty:
#                 continue

#             block_count += 1
#             pcm = np.frombuffer(data, dtype=np.int16)
#             rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))

#             # Calibrate noise floor
#             if len(noise_samples) < 50:
#                 noise_samples.append(rms)
#                 silence_threshold = 300
#                 if len(noise_samples) == 50:
#                     floor = float(np.percentile(noise_samples, 30))
#                     silence_threshold = max(100, floor * 2)
#                     log.info(f"Noise calibrated: floor={floor:.0f} threshold={silence_threshold:.0f}")

#             if block_count % 10 == 0:
#                 log.debug(f"block={block_count} rms={rms:.0f} thresh={silence_threshold:.0f} queue={self.audio_q.qsize()}")

#             if self.rec.AcceptWaveform(data):
#                 # Câu hoàn chỉnh → final, xóa partial
#                 res  = json.loads(self.rec.Result())
#                 text = res.get("text", "").strip()
#                 log.info(f"FINAL: '{text}'")
#                 last_partial = ""
#                 self._silence_count = 0
#                 self._speech_count  = 0
#                 if text:
#                     self.result_q.put({"type": "final", "text": text})
#             else:
#                 # Partial — in ngay từng từ mới xuất hiện
#                 res     = json.loads(self.rec.PartialResult())
#                 partial = res.get("partial", "").strip()
#                 if partial != last_partial:
#                     log.debug(f"PARTIAL: '{partial}'")
#                     last_partial = partial
#                     self.result_q.put({"type": "partial", "text": partial})

#         log.info("VoskWorker stopped")

#     def stop(self):
#         log.info("VoskWorker.stop() called")
#         try:
#             res  = json.loads(self.rec.FinalResult())
#             text = res.get("text", "").strip()
#             if text:
#                 log.info(f"stop() final flush: '{text}'")
#                 self.result_q.put({"type": "final", "text": text})
#         except Exception as e:
#             log.error(f"stop() error: {e}")
#         self.running = False


# # ─── Custom Scrollbar ─────────────────────────────────────────────────────────

# class SlimScrollbar(tk.Canvas):
#     W = 7

#     def __init__(self, parent, command=None, **kw):
#         super().__init__(parent, width=self.W, bg=C["surface"],
#                          highlightthickness=0, **kw)
#         self._cmd    = command
#         self._lo     = 0.0
#         self._hi     = 1.0
#         self._drag_y = None
#         self.bind("<Configure>",       lambda e: self._draw())
#         self.bind("<ButtonPress-1>",   self._press)
#         self.bind("<B1-Motion>",       self._drag)
#         self.bind("<ButtonRelease-1>", self._release)
#         self.bind("<Enter>",           self._hover_on)
#         self.bind("<Leave>",           self._hover_off)

#     def set(self, lo, hi):
#         self._lo, self._hi = float(lo), float(hi)
#         self._draw()

#     def _draw(self, hover=False):
#         self.delete("all")
#         h = self.winfo_height()
#         w = self.winfo_width()
#         if h < 2 or (self._hi - self._lo) >= 1.0:
#             return
#         ty  = max(2,   int(self._lo * h))
#         by  = min(h-2, max(ty + 20, int(self._hi * h)))
#         col = C["scroll_hv"] if hover else C["scrollbar"]
#         pts = [2+3,ty, w-2-3,ty, w-2,ty, w-2,ty+3,
#                w-2,by-3, w-2,by, w-2-3,by, 2+3,by,
#                2,by, 2,by-3, 2,ty+3, 2,ty, 2+3,ty]
#         self.create_polygon(pts, smooth=True, fill=col)

#     def _press(self, e):   self._drag_y = e.y
#     def _release(self, e): self._drag_y = None
#     def _hover_on(self, e):  self._draw(hover=True)
#     def _hover_off(self, e): self._draw(hover=False)

#     def _drag(self, e):
#         if self._drag_y is None or not self._cmd:
#             return
#         h    = self.winfo_height()
#         diff = (e.y - self._drag_y) / max(h, 1)
#         self._drag_y = e.y
#         span = self._hi - self._lo
#         new  = max(0.0, min(1.0 - span, self._lo + diff))
#         self._cmd("moveto", new)


# # ─── App ─────────────────────────────────────────────────────────────────────

# class App(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("Russian STT")
#         self.geometry("860x660")
#         self.minsize(640, 480)
#         self.configure(bg=C["bg"])

#         # Model load một lần, dùng lại mãi
#         self._vosk_model    = None

#         self.worker         = None
#         self.stream         = None
#         # audio_q và result_q tạo một lần, dùng chung
#         self.audio_q        = queue.Queue()
#         self.result_q       = queue.Queue()
#         self.is_recording   = False
#         self.is_paused      = False
#         self.session_log    = []
#         self.word_count     = 0
#         self._session_start = 0.0
#         self._partial_start = None

#         self._setup_fonts()
#         self._setup_styles()
#         self._build_ui()
#         self._load_model_async()
#         self.after(30, self._poll_results)

#     # ── Fonts & styles ────────────────────────────────────────────────────────

#     def _setup_fonts(self):
#         self.fn_title = tkfont.Font(family="Segoe UI", size=14, weight="bold")
#         self.fn_sub   = tkfont.Font(family="Segoe UI", size=9)
#         self.fn_body  = tkfont.Font(family="Segoe UI", size=15)
#         self.fn_small = tkfont.Font(family="Segoe UI", size=9)
#         self.fn_btn   = tkfont.Font(family="Segoe UI", size=10, weight="bold")
#         self.fn_stat  = tkfont.Font(family="Segoe UI", size=13, weight="bold")

#     def _setup_styles(self):
#         s = ttk.Style()
#         s.theme_use("clam")
#         s.configure("Vol.Horizontal.TProgressbar",
#                     troughcolor=C["panel"], background=C["green"],
#                     bordercolor=C["panel"], lightcolor=C["green"],
#                     darkcolor=C["green"], thickness=4)

#     # ── UI ────────────────────────────────────────────────────────────────────

#     def _build_ui(self):
#         # Header
#         hdr = tk.Frame(self, bg=C["surface"], padx=20, pady=14)
#         hdr.pack(fill="x")

#         left = tk.Frame(hdr, bg=C["surface"])
#         left.pack(side="left")
#         tk.Label(left, text="Russian STT", font=self.fn_title,
#                  bg=C["surface"], fg=C["text"]).pack(anchor="w")
#         tk.Label(left, text="Vosk  ·  real-time  ·  offline  ·  vosk-model-ru-0.42",
#                  font=self.fn_sub, bg=C["surface"], fg=C["muted"]).pack(anchor="w")

#         right = tk.Frame(hdr, bg=C["surface"])
#         right.pack(side="right")
#         tk.Label(right, text="MIC", font=self.fn_small,
#                  bg=C["surface"], fg=C["muted"]).pack(anchor="e")
#         self.vol_var = tk.DoubleVar(value=0)
#         ttk.Progressbar(right, variable=self.vol_var, maximum=100,
#                         length=120, style="Vol.Horizontal.TProgressbar"
#                         ).pack(anchor="e", pady=(3, 0))

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Status bar
#         sbar = tk.Frame(self, bg=C["panel"], padx=16, pady=7)
#         sbar.pack(fill="x")
#         self._dot = tk.Label(sbar, text="●", font=self.fn_small,
#                              bg=C["panel"], fg=C["muted"])
#         self._dot.pack(side="left", padx=(0, 7))
#         self.status_var = tk.StringVar(value="Đang tải model Vosk…")
#         tk.Label(sbar, textvariable=self.status_var, font=self.fn_small,
#                  bg=C["panel"], fg=C["muted"], anchor="w").pack(side="left")

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Toolbar — pack TRƯỚC txt_wrap
#         self._build_toolbar()

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Text area
#         txt_wrap = tk.Frame(self, bg=C["bg"], padx=16, pady=12)
#         txt_wrap.pack(fill="both", expand=True)

#         lbl_row = tk.Frame(txt_wrap, bg=C["bg"])
#         lbl_row.pack(fill="x", pady=(0, 6))
#         tk.Label(lbl_row, text="Văn bản nhận dạng",
#                  font=self.fn_small, bg=C["bg"], fg=C["muted"]).pack(side="left")
#         self._word_lbl = tk.Label(lbl_row, text="0 từ",
#                                   font=self.fn_small, bg=C["bg"], fg=C["muted"])
#         self._word_lbl.pack(side="right")

#         box = tk.Frame(txt_wrap, bg=C["surface"],
#                        highlightbackground=C["border"], highlightthickness=1)
#         box.pack(fill="both", expand=True)

#         self.text_area = tk.Text(
#             box,
#             font=self.fn_body,
#             bg=C["surface"], fg=C["text"],
#             insertbackground=C["text"],
#             selectbackground=C["accent"],
#             relief="flat", bd=0,
#             wrap=tk.WORD,
#             padx=18, pady=14,
#             spacing1=2, spacing3=8,
#             cursor="arrow",
#             state="disabled",
#         )
#         self.text_area.pack(side="left", fill="both", expand=True)

#         sb = SlimScrollbar(box, command=self.text_area.yview)
#         sb.pack(side="right", fill="y", padx=(0, 3), pady=6)
#         self.text_area.configure(yscrollcommand=sb.set)

#         self.text_area.tag_config("final",   foreground=C["text"],  font=self.fn_body)
#         self.text_area.tag_config("partial", foreground=C["partial"], font=self.fn_body)
#         self.text_area.tag_config("pause",   foreground=C["amber"],   font=self.fn_small)

#     def _build_toolbar(self):
#         bar = tk.Frame(self, bg=C["surface"], padx=16, pady=10)
#         bar.pack(fill="x")

#         self.rec_btn = self._btn(bar, "⏺  Ghi âm", C["red"],
#                                  self.toggle_recording, state="disabled")
#         self.rec_btn.pack(side="left", padx=(0, 6))

#         self.pause_btn = self._btn(bar, "⏸  Tạm dừng", C["amber"],
#                                    self.toggle_pause, state="disabled")
#         self.pause_btn.pack(side="left", padx=(0, 18))

#         tk.Frame(bar, bg=C["border"], width=1, height=22).pack(side="left", padx=(0, 18))

#         self._btn(bar, "💾  Lưu", C["panel"], self.save_session).pack(side="left", padx=(0, 6))
#         self._btn(bar, "🗑  Xóa", C["panel"], self.clear_text ).pack(side="left")

#         stat = tk.Frame(bar, bg=C["surface"])
#         stat.pack(side="right")
#         self._time_var = tk.StringVar(value="00:00")
#         tk.Label(stat, textvariable=self._time_var, font=self.fn_stat,
#                  bg=C["surface"], fg=C["accent"]).pack()
#         tk.Label(stat, text="phiên", font=self.fn_small,
#                  bg=C["surface"], fg=C["muted"]).pack()

#     def _btn(self, parent, text, bg, cmd, state="normal"):
#         b = tk.Button(parent, text=text, font=self.fn_btn,
#                       bg=bg, fg=C["text"], activebackground=bg,
#                       activeforeground=C["text"], relief="flat",
#                       padx=14, pady=6, cursor="hand2",
#                       command=cmd, state=state, bd=0, highlightthickness=0)
#         def _hl(c): return "#{:02x}{:02x}{:02x}".format(
#             *[min(255, int(c.lstrip("#")[i:i+2], 16) + 22) for i in (0, 2, 4)])
#         b.bind("<Enter>", lambda e: b.config(bg=_hl(bg)))
#         b.bind("<Leave>", lambda e: b.config(bg=bg))
#         return b

#     # ── Model load — chỉ chạy MỘT LẦN ───────────────────────────────────────

#     def _load_model_async(self):
#         self.status_var.set("Đang tải model Vosk… (chỉ lần đầu)")
#         self._dot.config(fg=C["amber"])

#         def _load():
#             try:
#                 vosk.SetLogLevel(-1)
#                 model = vosk.Model(MODEL_PATH)
#                 self._vosk_model = model
#                 self.after(0, self._on_ready)
#             except Exception as e:
#                 err = str(e)
#                 self.after(0, lambda: self._on_error(err))

#         threading.Thread(target=_load, daemon=True).start()

#     def _on_ready(self):
#         self.status_var.set("Sẵn sàng  ·  Vosk  ·  real-time")
#         self._dot.config(fg=C["green"])
#         self.rec_btn.config(state="normal")

#     def _on_error(self, err):
#         self.status_var.set(f"Lỗi tải model: {err[:60]}")
#         self._dot.config(fg=C["red"])
#         messagebox.showerror("Không tải được model",
#                              f"Kiểm tra thư mục:\n{MODEL_PATH}\n\n{err}")

#     # ── Recording ─────────────────────────────────────────────────────────────

#     def toggle_recording(self):
#         if self.is_recording: self.stop_recording()
#         else:                  self.start_recording()

#     def start_recording(self):
#         self.is_recording   = True
#         self.is_paused      = False
#         self._session_start = time.time()

#         # Xả queue cũ còn sót
#         while not self.audio_q.empty():
#             self.audio_q.get_nowait()
#         while not self.result_q.empty():
#             self.result_q.get_nowait()

#         # Worker mới — chỉ tạo KaldiRecognizer, KHÔNG load model lại
#         self.worker = VoskWorker(self._vosk_model, self.audio_q, self.result_q)
#         self.worker.start()

#         self.rec_btn.config(text="⏹  Dừng", bg=C["green"])
#         self.pause_btn.config(state="normal")
#         self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
#         self._dot.config(fg=C["red"])
#         self._tick()

#         _cb_count = [0]
#         def _cb(indata, frames, t, status):
#             if status:
#                 log.warning(f"sounddevice status: {status}")
#             if self.is_paused:
#                 return
#             # Boost cố định 6x — đủ để Vosk nghe, không clip với mic thường
#             samples = indata[:, 0].astype(np.float32)
#             samples = np.clip(samples * 6.0, -1.0, 1.0)
#             pcm     = (samples * 32767).astype("int16").tobytes()
#             self.audio_q.put_nowait(pcm)
#             _cb_count[0] += 1
#             if _cb_count[0] % 25 == 0:  # log mỗi 2s
#                 log.debug(f"audio_cb: block={_cb_count[0]} queue={self.audio_q.qsize()}")
#             rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
#             self.after(0, lambda: self.vol_var.set(min(100, rms * 2500)))

#         self.stream = sd.InputStream(
#             samplerate=SAMPLE_RATE, channels=1,
#             dtype="float32", blocksize=BLOCK_SIZE,
#             callback=_cb,
#         )
#         self.stream.start()

#     def stop_recording(self):
#         self.is_recording = False
#         self.is_paused    = False
#         if self.stream:
#             self.stream.stop(); self.stream.close(); self.stream = None
#         if self.worker:
#             self.worker.stop(); self.worker = None
#         self.vol_var.set(0)
#         self.rec_btn.config(text="⏺  Ghi âm", bg=C["red"])
#         self.pause_btn.config(state="disabled", text="⏸  Tạm dừng", bg=C["amber"])
#         self.status_var.set("Đã dừng  ·  Nhấn để ghi âm tiếp")
#         self._dot.config(fg=C["muted"])
#         self._clear_partial()

#     def toggle_pause(self):
#         self.is_paused = not self.is_paused
#         if self.is_paused:
#             self.pause_btn.config(text="▶  Tiếp tục", bg=C["accent"])
#             self.status_var.set("Tạm dừng")
#             self._dot.config(fg=C["amber"])
#             self.vol_var.set(0)
#             self._clear_partial()
#             self._insert_pause_marker()
#         else:
#             self.pause_btn.config(text="⏸  Tạm dừng", bg=C["amber"])
#             self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
#             self._dot.config(fg=C["red"])

#     # ── Timer ─────────────────────────────────────────────────────────────────

#     def _tick(self):
#         if not self.is_recording:
#             return
#         e = int(time.time() - self._session_start)
#         self._time_var.set(f"{e//60:02d}:{e%60:02d}")
#         self.after(1000, self._tick)

#     # ── Results ───────────────────────────────────────────────────────────────

#     def _poll_results(self):
#         try:
#             while True:
#                 msg = self.result_q.get_nowait()
#                 if msg["type"] == "partial":
#                     self._show_partial(msg["text"])
#                 else:
#                     self._commit_final(msg["text"])
#         except queue.Empty:
#             pass
#         self.after(30, self._poll_results)

#     def _show_partial(self, text: str):
#         """Xóa partial cũ, hiện partial mới màu xám."""
#         self.text_area.config(state="normal")
#         if self._partial_start:
#             self.text_area.delete(self._partial_start, "end")
#         if text:
#             self._partial_start = self.text_area.index("end-1c")
#             self.text_area.insert("end", " " + text, "partial")
#             self.text_area.see("end")
#         else:
#             self._partial_start = None
#         self.text_area.config(state="disabled")

#     def _commit_final(self, text: str):
#         """Xóa partial, ghi final màu trắng + xuống dòng."""
#         log.info(f"UI commit: '{text}'")
#         self.text_area.config(state="normal")
#         if self._partial_start:
#             self.text_area.delete(self._partial_start, "end")
#             self._partial_start = None
#         self.text_area.insert("end", text + " ", "final")
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")
#         self.session_log.append(text)
#         self.word_count += len(text.split())
#         self._word_lbl.config(text=f"{self.word_count} từ")

#     def _clear_partial(self):
#         if self._partial_start:
#             self.text_area.config(state="normal")
#             self.text_area.delete(self._partial_start, "end")
#             self.text_area.config(state="disabled")
#             self._partial_start = None

#     def _insert_pause_marker(self):
#         ts = datetime.datetime.now().strftime("%H:%M:%S")
#         self.text_area.config(state="normal")
#         self.text_area.insert("end", f"\n— tạm dừng {ts} —\n", "pause")
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")

#     # ── Helpers ───────────────────────────────────────────────────────────────

#     def clear_text(self):
#         self.text_area.config(state="normal")
#         self.text_area.delete("1.0", "end")
#         self.text_area.config(state="disabled")
#         self.session_log.clear()
#         self.word_count     = 0
#         self._partial_start = None
#         self._word_lbl.config(text="0 từ")

#     def save_session(self):
#         if not self.session_log:
#             messagebox.showinfo("Thông báo", "Chưa có nội dung để lưu.")
#             return
#         default = f"session_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
#         path = filedialog.asksaveasfilename(
#             defaultextension=".txt",
#             filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
#             initialfile=default, title="Lưu phiên ghi âm",
#         )
#         if not path:
#             return
#         with open(path, "w", encoding="utf-8") as f:
#             f.write(f"# Russian STT — {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
#             f.write(f"# Từ: {self.word_count}\n\n")
#             f.write(" ".join(self.session_log))
#         messagebox.showinfo("Đã lưu", f"Lưu thành công:\n{path}")

#     def on_closing(self):
#         if self.is_recording:
#             self.stop_recording()
#         self.destroy()


# # ─── Entry point ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     app = App()
#     app.protocol("WM_DELETE_WINDOW", app.on_closing)
#     app.mainloop()
# ______________________________________ OK 3 ______________________________________
# """
# Russian STT — Vosk real-time, từng từ hiện ngay
# Cấu trúc thư mục:
#   STT/
#   └── server/
#       ├── app.py
#       └── model/   ← vosk-model-ru-0.42
# """

# import os
# import threading
# import queue
# import json
# import logging
# import tkinter as tk
# import tkinter.font as tkfont
# from tkinter import ttk, messagebox, filedialog
# import sounddevice as sd
# import vosk
# import datetime
# import time
# import numpy as np

# # ─── Debug logger ─────────────────────────────────────────────────────────────
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s.%(msecs)03d [%(threadName)s] %(levelname)s — %(message)s",
#     datefmt="%H:%M:%S",
#     handlers=[
#         logging.FileHandler("debug.log", encoding="utf-8", mode="w"),
#         logging.StreamHandler(),   # in ra terminal luôn
#     ]
# )
# log = logging.getLogger("STT")

# # ─── Cấu hình ────────────────────────────────────────────────────────────────

# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH  = os.path.join(_SCRIPT_DIR, "model")  # vosk-model-ru-0.42 (model lớn)
# SAMPLE_RATE = 16000
# BLOCK_MS    = 80
# BLOCK_SIZE  = int(SAMPLE_RATE * BLOCK_MS / 1000)

# # ─── Màu sắc ─────────────────────────────────────────────────────────────────

# C = {
#     "bg":        "#0f1117",
#     "surface":   "#1a1d27",
#     "panel":     "#21253a",
#     "border":    "#2d3148",
#     "text":      "#e8eaf0",
#     "muted":     "#5a6070",
#     "accent":    "#6c8fff",
#     "green":     "#3ecf72",
#     "red":       "#ef4444",
#     "amber":     "#f59e0b",
#     "partial":   "#94a3b8",
#     "scrollbar": "#2d3148",
#     "scroll_hv": "#4a5080",
# }

# # ─── Vosk Worker ─────────────────────────────────────────────────────────────

# class VoskWorker(threading.Thread):
#     """
#     Nhận model đã load sẵn — KHÔNG load lại model mỗi lần record.
#     Flush khi:
#       1. Vosk tự AcceptWaveform = True (câu hoàn chỉnh)
#       2. Im lặng >= 400ms  (người dừng nói)
#       3. Nói liên tục >= 4s (force flush để không tích lũy quá lâu)
#     """
#     # Tất cả tính theo số BLOCK (1 block = BLOCK_MS ms)
#     SILENCE_BLOCKS  = max(1, int(800  / BLOCK_MS))   # 400ms im lặng → flush
#     MAX_SPEECH_BLOCKS = max(1, int(8000 / BLOCK_MS)) # 4s nói liên tục → force flush
#     MAX_BACKLOG     = 10   # drop nếu queue > 10 block

#     def __init__(self, model: vosk.Model, audio_q: queue.Queue,
#                  result_q: queue.Queue):
#         super().__init__(daemon=True)
#         self.audio_q          = audio_q
#         self.result_q         = result_q
#         self.running          = True
#         self._model           = model   # giữ ref để tạo lại recognizer sau flush
#         self.rec              = vosk.KaldiRecognizer(model, SAMPLE_RATE)
#         self._silence_count   = 0
#         self._speech_count    = 0   # tổng block có tiếng kể từ lần flush cuối
#         self._speech_rms_sum  = 0.0
#         self._speech_rms_n    = 0

#     def _flush(self, reason: str):
#         log.debug(f"FLUSH [{reason}] speech_blocks={self._speech_count} silence_blocks={self._silence_count}")
#         res  = json.loads(self.rec.FinalResult())
#         text = res.get("text", "").strip()
#         log.debug(f"FLUSH result: '{text}'")
#         # Reset để nhận câu tiếp theo sạch
#         self.rec = vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
#         self._silence_count  = 0
#         self._speech_count   = 0
#         self._speech_rms_sum = 0.0
#         self._speech_rms_n   = 0
#         if text:
#             self.result_q.put({"type": "final", "text": text})
#         else:
#             log.debug("FLUSH: empty result, nothing pushed")

#     def run(self):
#         log.info("VoskWorker started")
#         noise_samples     = []
#         block_count       = 0
#         silence_threshold = 300
#         last_partial      = ""

#         while self.running:
#             # Drop backlog
#             backlog = self.audio_q.qsize()
#             if backlog > self.MAX_BACKLOG:
#                 dropped = 0
#                 while self.audio_q.qsize() > self.MAX_BACKLOG:
#                     try: self.audio_q.get_nowait(); dropped += 1
#                     except: break
#                 log.warning(f"DROP {dropped} blocks (backlog={backlog})")

#             try:
#                 data = self.audio_q.get(timeout=0.1)
#             except queue.Empty:
#                 continue

#             block_count += 1
#             pcm = np.frombuffer(data, dtype=np.int16)
#             rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))

#             # Calibrate noise floor
#             if len(noise_samples) < 50:
#                 noise_samples.append(rms)
#                 silence_threshold = 300
#                 if len(noise_samples) == 50:
#                     floor = float(np.percentile(noise_samples, 30))
#                     silence_threshold = max(100, floor * 2)
#                     log.info(f"Noise calibrated: floor={floor:.0f} threshold={silence_threshold:.0f}")

#             if block_count % 10 == 0:
#                 log.debug(f"block={block_count} rms={rms:.0f} thresh={silence_threshold:.0f} queue={self.audio_q.qsize()}")

#             if self.rec.AcceptWaveform(data):
#                 # Câu hoàn chỉnh → final, xóa partial
#                 res  = json.loads(self.rec.Result())
#                 text = res.get("text", "").strip()
#                 log.info(f"FINAL: '{text}'")
#                 last_partial = ""
#                 self._silence_count = 0
#                 self._speech_count  = 0
#                 if text:
#                     self.result_q.put({"type": "final", "text": text})
#             else:
#                 # Partial — in ngay từng từ mới xuất hiện
#                 res     = json.loads(self.rec.PartialResult())
#                 partial = res.get("partial", "").strip()
#                 if partial != last_partial:
#                     log.debug(f"PARTIAL: '{partial}'")
#                     last_partial = partial
#                     self.result_q.put({"type": "partial", "text": partial})

#         log.info("VoskWorker stopped")

#     def stop(self):
#         log.info("VoskWorker.stop() called")
#         try:
#             res  = json.loads(self.rec.FinalResult())
#             text = res.get("text", "").strip()
#             if text:
#                 log.info(f"stop() final flush: '{text}'")
#                 self.result_q.put({"type": "final", "text": text})
#         except Exception as e:
#             log.error(f"stop() error: {e}")
#         self.running = False


# # ─── Custom Scrollbar ─────────────────────────────────────────────────────────

# class SlimScrollbar(tk.Canvas):
#     W = 7

#     def __init__(self, parent, command=None, **kw):
#         super().__init__(parent, width=self.W, bg=C["surface"],
#                          highlightthickness=0, **kw)
#         self._cmd    = command
#         self._lo     = 0.0
#         self._hi     = 1.0
#         self._drag_y = None
#         self.bind("<Configure>",       lambda e: self._draw())
#         self.bind("<ButtonPress-1>",   self._press)
#         self.bind("<B1-Motion>",       self._drag)
#         self.bind("<ButtonRelease-1>", self._release)
#         self.bind("<Enter>",           self._hover_on)
#         self.bind("<Leave>",           self._hover_off)

#     def set(self, lo, hi):
#         self._lo, self._hi = float(lo), float(hi)
#         self._draw()

#     def _draw(self, hover=False):
#         self.delete("all")
#         h = self.winfo_height()
#         w = self.winfo_width()
#         if h < 2 or (self._hi - self._lo) >= 1.0:
#             return
#         ty  = max(2,   int(self._lo * h))
#         by  = min(h-2, max(ty + 20, int(self._hi * h)))
#         col = C["scroll_hv"] if hover else C["scrollbar"]
#         pts = [2+3,ty, w-2-3,ty, w-2,ty, w-2,ty+3,
#                w-2,by-3, w-2,by, w-2-3,by, 2+3,by,
#                2,by, 2,by-3, 2,ty+3, 2,ty, 2+3,ty]
#         self.create_polygon(pts, smooth=True, fill=col)

#     def _press(self, e):   self._drag_y = e.y
#     def _release(self, e): self._drag_y = None
#     def _hover_on(self, e):  self._draw(hover=True)
#     def _hover_off(self, e): self._draw(hover=False)

#     def _drag(self, e):
#         if self._drag_y is None or not self._cmd:
#             return
#         h    = self.winfo_height()
#         diff = (e.y - self._drag_y) / max(h, 1)
#         self._drag_y = e.y
#         span = self._hi - self._lo
#         new  = max(0.0, min(1.0 - span, self._lo + diff))
#         self._cmd("moveto", new)


# # ─── App ─────────────────────────────────────────────────────────────────────

# class App(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("Russian STT")
#         self.geometry("860x660")
#         self.minsize(640, 480)
#         self.configure(bg=C["bg"])

#         # Model load một lần, dùng lại mãi
#         self._vosk_model    = None

#         self.worker         = None
#         self.stream         = None
#         # audio_q và result_q tạo một lần, dùng chung
#         self.audio_q        = queue.Queue()
#         self.result_q       = queue.Queue()
#         self.is_recording   = False
#         self.is_paused      = False
#         self.session_log    = []
#         self.word_count     = 0
#         self._session_start = 0.0
#         self._partial_start = None

#         self._setup_fonts()
#         self._setup_styles()
#         self._build_ui()
#         self._load_model_async()
#         self.after(30, self._poll_results)

#     # ── Fonts & styles ────────────────────────────────────────────────────────

#     def _setup_fonts(self):
#         self.fn_title = tkfont.Font(family="Segoe UI", size=14, weight="bold")
#         self.fn_sub   = tkfont.Font(family="Segoe UI", size=9)
#         self.fn_body  = tkfont.Font(family="Segoe UI", size=15)
#         self.fn_small = tkfont.Font(family="Segoe UI", size=9)
#         self.fn_btn   = tkfont.Font(family="Segoe UI", size=10, weight="bold")
#         self.fn_stat  = tkfont.Font(family="Segoe UI", size=13, weight="bold")

#     def _setup_styles(self):
#         s = ttk.Style()
#         s.theme_use("clam")
#         s.configure("Vol.Horizontal.TProgressbar",
#                     troughcolor=C["panel"], background=C["green"],
#                     bordercolor=C["panel"], lightcolor=C["green"],
#                     darkcolor=C["green"], thickness=4)

#     # ── UI ────────────────────────────────────────────────────────────────────

#     def _build_ui(self):
#         # Header
#         hdr = tk.Frame(self, bg=C["surface"], padx=20, pady=14)
#         hdr.pack(fill="x")

#         left = tk.Frame(hdr, bg=C["surface"])
#         left.pack(side="left")
#         tk.Label(left, text="Russian STT", font=self.fn_title,
#                  bg=C["surface"], fg=C["text"]).pack(anchor="w")
#         tk.Label(left, text="Vosk  ·  real-time  ·  offline  ·  vosk-model-ru-0.42",
#                  font=self.fn_sub, bg=C["surface"], fg=C["muted"]).pack(anchor="w")

#         right = tk.Frame(hdr, bg=C["surface"])
#         right.pack(side="right")
#         tk.Label(right, text="MIC", font=self.fn_small,
#                  bg=C["surface"], fg=C["muted"]).pack(anchor="e")
#         self.vol_var = tk.DoubleVar(value=0)
#         ttk.Progressbar(right, variable=self.vol_var, maximum=100,
#                         length=120, style="Vol.Horizontal.TProgressbar"
#                         ).pack(anchor="e", pady=(3, 0))

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Status bar
#         sbar = tk.Frame(self, bg=C["panel"], padx=16, pady=7)
#         sbar.pack(fill="x")
#         self._dot = tk.Label(sbar, text="●", font=self.fn_small,
#                              bg=C["panel"], fg=C["muted"])
#         self._dot.pack(side="left", padx=(0, 7))
#         self.status_var = tk.StringVar(value="Đang tải model Vosk…")
#         tk.Label(sbar, textvariable=self.status_var, font=self.fn_small,
#                  bg=C["panel"], fg=C["muted"], anchor="w").pack(side="left")

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Toolbar — pack TRƯỚC txt_wrap
#         self._build_toolbar()

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Text area
#         txt_wrap = tk.Frame(self, bg=C["bg"], padx=16, pady=12)
#         txt_wrap.pack(fill="both", expand=True)

#         lbl_row = tk.Frame(txt_wrap, bg=C["bg"])
#         lbl_row.pack(fill="x", pady=(0, 6))
#         tk.Label(lbl_row, text="Văn bản nhận dạng",
#                  font=self.fn_small, bg=C["bg"], fg=C["muted"]).pack(side="left")
#         self._word_lbl = tk.Label(lbl_row, text="0 từ",
#                                   font=self.fn_small, bg=C["bg"], fg=C["muted"])
#         self._word_lbl.pack(side="right")

#         box = tk.Frame(txt_wrap, bg=C["surface"],
#                        highlightbackground=C["border"], highlightthickness=1)
#         box.pack(fill="both", expand=True)

#         self.text_area = tk.Text(
#             box,
#             font=self.fn_body,
#             bg=C["surface"], fg=C["text"],
#             insertbackground=C["text"],
#             selectbackground=C["accent"],
#             relief="flat", bd=0,
#             wrap=tk.WORD,
#             padx=18, pady=14,
#             spacing1=2, spacing3=8,
#             cursor="arrow",
#             state="disabled",
#         )
#         self.text_area.pack(side="left", fill="both", expand=True)

#         sb = SlimScrollbar(box, command=self.text_area.yview)
#         sb.pack(side="right", fill="y", padx=(0, 3), pady=6)
#         self.text_area.configure(yscrollcommand=sb.set)

#         self.text_area.tag_config("final",   foreground=C["text"],  font=self.fn_body)
#         self.text_area.tag_config("partial", foreground=C["partial"], font=self.fn_body)
#         self.text_area.tag_config("pause",   foreground=C["amber"],   font=self.fn_small)

#     def _build_toolbar(self):
#         bar = tk.Frame(self, bg=C["surface"], padx=16, pady=10)
#         bar.pack(fill="x")

#         self.rec_btn = self._btn(bar, "⏺  Ghi âm", C["red"],
#                                  self.toggle_recording, state="disabled")
#         self.rec_btn.pack(side="left", padx=(0, 6))

#         self.pause_btn = self._btn(bar, "⏸  Tạm dừng", C["amber"],
#                                    self.toggle_pause, state="disabled")
#         self.pause_btn.pack(side="left", padx=(0, 18))

#         tk.Frame(bar, bg=C["border"], width=1, height=22).pack(side="left", padx=(0, 18))

#         self._btn(bar, "💾  Lưu", C["panel"], self.save_session).pack(side="left", padx=(0, 6))
#         self._btn(bar, "🗑  Xóa", C["panel"], self.clear_text ).pack(side="left")

#         stat = tk.Frame(bar, bg=C["surface"])
#         stat.pack(side="right")
#         self._time_var = tk.StringVar(value="00:00")
#         tk.Label(stat, textvariable=self._time_var, font=self.fn_stat,
#                  bg=C["surface"], fg=C["accent"]).pack()
#         tk.Label(stat, text="phiên", font=self.fn_small,
#                  bg=C["surface"], fg=C["muted"]).pack()

#     def _btn(self, parent, text, bg, cmd, state="normal"):
#         b = tk.Button(parent, text=text, font=self.fn_btn,
#                       bg=bg, fg=C["text"], activebackground=bg,
#                       activeforeground=C["text"], relief="flat",
#                       padx=14, pady=6, cursor="hand2",
#                       command=cmd, state=state, bd=0, highlightthickness=0)
#         def _hl(c): return "#{:02x}{:02x}{:02x}".format(
#             *[min(255, int(c.lstrip("#")[i:i+2], 16) + 22) for i in (0, 2, 4)])
#         b.bind("<Enter>", lambda e: b.config(bg=_hl(bg)))
#         b.bind("<Leave>", lambda e: b.config(bg=bg))
#         return b

#     # ── Model load — chỉ chạy MỘT LẦN ───────────────────────────────────────

#     def _load_model_async(self):
#         self.status_var.set("Đang tải model Vosk… (chỉ lần đầu)")
#         self._dot.config(fg=C["amber"])

#         def _load():
#             try:
#                 vosk.SetLogLevel(-1)
#                 model = vosk.Model(MODEL_PATH)
#                 self._vosk_model = model
#                 self.after(0, self._on_ready)
#             except Exception as e:
#                 err = str(e)
#                 self.after(0, lambda: self._on_error(err))

#         threading.Thread(target=_load, daemon=True).start()

#     def _on_ready(self):
#         self.status_var.set("Sẵn sàng  ·  Vosk  ·  real-time")
#         self._dot.config(fg=C["green"])
#         self.rec_btn.config(state="normal")

#     def _on_error(self, err):
#         self.status_var.set(f"Lỗi tải model: {err[:60]}")
#         self._dot.config(fg=C["red"])
#         messagebox.showerror("Không tải được model",
#                              f"Kiểm tra thư mục:\n{MODEL_PATH}\n\n{err}")

#     # ── Recording ─────────────────────────────────────────────────────────────

#     def toggle_recording(self):
#         if self.is_recording: self.stop_recording()
#         else:                  self.start_recording()

#     def start_recording(self):
#         self.is_recording   = True
#         self.is_paused      = False
#         self._session_start = time.time()

#         # Xả queue cũ còn sót
#         while not self.audio_q.empty():
#             self.audio_q.get_nowait()
#         while not self.result_q.empty():
#             self.result_q.get_nowait()

#         # Worker mới — chỉ tạo KaldiRecognizer, KHÔNG load model lại
#         self.worker = VoskWorker(self._vosk_model, self.audio_q, self.result_q)
#         self.worker.start()

#         self.rec_btn.config(text="⏹  Dừng", bg=C["green"])
#         self.pause_btn.config(state="normal")
#         self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
#         self._dot.config(fg=C["red"])
#         self._tick()

#         _cb_count = [0]
#         def _cb(indata, frames, t, status):
#             if status:
#                 log.warning(f"sounddevice status: {status}")
#             if self.is_paused:
#                 return
#             # Normalize nhẹ — chỉ boost nếu quá nhỏ, không clip
#             samples = indata[:, 0].astype(np.float32)
#             peak    = float(np.abs(samples).max()) + 1e-9
#             # Chỉ boost nếu peak < 0.3, cap tối đa 4x để không clip
#             if peak < 0.3:
#                 gain = min(0.3 / peak, 4.0)
#                 samples = samples * gain
#             pcm = (samples * 32767).astype("int16").tobytes()
#             self.audio_q.put_nowait(pcm)
#             _cb_count[0] += 1
#             if _cb_count[0] % 25 == 0:  # log mỗi 2s
#                 log.debug(f"audio_cb: block={_cb_count[0]} queue={self.audio_q.qsize()}")
#             rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
#             self.after(0, lambda: self.vol_var.set(min(100, rms * 2500)))

#         self.stream = sd.InputStream(
#             samplerate=SAMPLE_RATE, channels=1,
#             dtype="float32", blocksize=BLOCK_SIZE,
#             callback=_cb,
#         )
#         self.stream.start()

#     def stop_recording(self):
#         self.is_recording = False
#         self.is_paused    = False
#         if self.stream:
#             self.stream.stop(); self.stream.close(); self.stream = None
#         if self.worker:
#             self.worker.stop(); self.worker = None
#         self.vol_var.set(0)
#         self.rec_btn.config(text="⏺  Ghi âm", bg=C["red"])
#         self.pause_btn.config(state="disabled", text="⏸  Tạm dừng", bg=C["amber"])
#         self.status_var.set("Đã dừng  ·  Nhấn để ghi âm tiếp")
#         self._dot.config(fg=C["muted"])
#         self._clear_partial()

#     def toggle_pause(self):
#         self.is_paused = not self.is_paused
#         if self.is_paused:
#             self.pause_btn.config(text="▶  Tiếp tục", bg=C["accent"])
#             self.status_var.set("Tạm dừng")
#             self._dot.config(fg=C["amber"])
#             self.vol_var.set(0)
#             self._clear_partial()
#             self._insert_pause_marker()
#         else:
#             self.pause_btn.config(text="⏸  Tạm dừng", bg=C["amber"])
#             self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
#             self._dot.config(fg=C["red"])

#     # ── Timer ─────────────────────────────────────────────────────────────────

#     def _tick(self):
#         if not self.is_recording:
#             return
#         e = int(time.time() - self._session_start)
#         self._time_var.set(f"{e//60:02d}:{e%60:02d}")
#         self.after(1000, self._tick)

#     # ── Results ───────────────────────────────────────────────────────────────

#     def _poll_results(self):
#         try:
#             while True:
#                 msg = self.result_q.get_nowait()
#                 if msg["type"] == "partial":
#                     self._show_partial(msg["text"])
#                 else:
#                     self._commit_final(msg["text"])
#         except queue.Empty:
#             pass
#         self.after(30, self._poll_results)

#     def _show_partial(self, text: str):
#         """Xóa partial cũ, hiện partial mới màu xám."""
#         self.text_area.config(state="normal")
#         if self._partial_start:
#             self.text_area.delete(self._partial_start, "end")
#         if text:
#             self._partial_start = self.text_area.index("end-1c")
#             self.text_area.insert("end", " " + text, "partial")
#             self.text_area.see("end")
#         else:
#             self._partial_start = None
#         self.text_area.config(state="disabled")

#     def _commit_final(self, text: str):
#         """Xóa partial, ghi final màu trắng + xuống dòng."""
#         log.info(f"UI commit: '{text}'")
#         self.text_area.config(state="normal")
#         if self._partial_start:
#             self.text_area.delete(self._partial_start, "end")
#             self._partial_start = None
#         self.text_area.insert("end", text + " ", "final")
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")
#         self.session_log.append(text)
#         self.word_count += len(text.split())
#         self._word_lbl.config(text=f"{self.word_count} từ")

#     def _clear_partial(self):
#         if self._partial_start:
#             self.text_area.config(state="normal")
#             self.text_area.delete(self._partial_start, "end")
#             self.text_area.config(state="disabled")
#             self._partial_start = None

#     def _insert_pause_marker(self):
#         ts = datetime.datetime.now().strftime("%H:%M:%S")
#         self.text_area.config(state="normal")
#         self.text_area.insert("end", f"\n— tạm dừng {ts} —\n", "pause")
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")

#     # ── Helpers ───────────────────────────────────────────────────────────────

#     def clear_text(self):
#         self.text_area.config(state="normal")
#         self.text_area.delete("1.0", "end")
#         self.text_area.config(state="disabled")
#         self.session_log.clear()
#         self.word_count     = 0
#         self._partial_start = None
#         self._word_lbl.config(text="0 từ")

#     def save_session(self):
#         if not self.session_log:
#             messagebox.showinfo("Thông báo", "Chưa có nội dung để lưu.")
#             return
#         default = f"session_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
#         path = filedialog.asksaveasfilename(
#             defaultextension=".txt",
#             filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
#             initialfile=default, title="Lưu phiên ghi âm",
#         )
#         if not path:
#             return
#         with open(path, "w", encoding="utf-8") as f:
#             f.write(f"# Russian STT — {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
#             f.write(f"# Từ: {self.word_count}\n\n")
#             f.write(" ".join(self.session_log))
#         messagebox.showinfo("Đã lưu", f"Lưu thành công:\n{path}")

#     def on_closing(self):
#         if self.is_recording:
#             self.stop_recording()
#         self.destroy()


# # ─── Entry point ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     app = App()
#     app.protocol("WM_DELETE_WINDOW", app.on_closing)
#     app.mainloop()

# ------------------------------ OK 2 ____________________________________________
# """
# Russian STT — Vosk real-time, từng từ hiện ngay
# Cấu trúc thư mục:
#   STT/
#   └── server/
#       ├── app.py
#       └── model/   ← vosk-model-ru-0.42
# """

# import os
# import threading
# import queue
# import json
# import logging
# import tkinter as tk
# import tkinter.font as tkfont
# from tkinter import ttk, messagebox, filedialog
# import sounddevice as sd
# import vosk
# import datetime
# import time
# import numpy as np

# # ─── Debug logger ─────────────────────────────────────────────────────────────
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s.%(msecs)03d [%(threadName)s] %(levelname)s — %(message)s",
#     datefmt="%H:%M:%S",
#     handlers=[
#         logging.FileHandler("debug.log", encoding="utf-8", mode="w"),
#         logging.StreamHandler(),   # in ra terminal luôn
#     ]
# )
# log = logging.getLogger("STT")

# # ─── Cấu hình ────────────────────────────────────────────────────────────────

# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH  = os.path.join(_SCRIPT_DIR, "model")  # vosk-model-ru-0.42 (model lớn)
# SAMPLE_RATE = 16000
# BLOCK_MS    = 80
# BLOCK_SIZE  = int(SAMPLE_RATE * BLOCK_MS / 1000)

# # ─── Màu sắc ─────────────────────────────────────────────────────────────────

# C = {
#     "bg":        "#0f1117",
#     "surface":   "#1a1d27",
#     "panel":     "#21253a",
#     "border":    "#2d3148",
#     "text":      "#e8eaf0",
#     "muted":     "#5a6070",
#     "accent":    "#6c8fff",
#     "green":     "#3ecf72",
#     "red":       "#ef4444",
#     "amber":     "#f59e0b",
#     "partial":   "#94a3b8",
#     "scrollbar": "#2d3148",
#     "scroll_hv": "#4a5080",
# }

# # ─── Vosk Worker ─────────────────────────────────────────────────────────────

# class VoskWorker(threading.Thread):
#     """
#     Nhận model đã load sẵn — KHÔNG load lại model mỗi lần record.
#     Flush khi:
#       1. Vosk tự AcceptWaveform = True (câu hoàn chỉnh)
#       2. Im lặng >= 400ms  (người dừng nói)
#       3. Nói liên tục >= 4s (force flush để không tích lũy quá lâu)
#     """
#     # Tất cả tính theo số BLOCK (1 block = BLOCK_MS ms)
#     SILENCE_BLOCKS  = max(1, int(800  / BLOCK_MS))   # 400ms im lặng → flush
#     MAX_SPEECH_BLOCKS = max(1, int(8000 / BLOCK_MS)) # 4s nói liên tục → force flush
#     MAX_BACKLOG     = 10   # drop nếu queue > 10 block

#     def __init__(self, model: vosk.Model, audio_q: queue.Queue,
#                  result_q: queue.Queue):
#         super().__init__(daemon=True)
#         self.audio_q          = audio_q
#         self.result_q         = result_q
#         self.running          = True
#         self._model           = model   # giữ ref để tạo lại recognizer sau flush
#         self.rec              = vosk.KaldiRecognizer(model, SAMPLE_RATE)
#         self._silence_count   = 0
#         self._speech_count    = 0   # tổng block có tiếng kể từ lần flush cuối
#         self._speech_rms_sum  = 0.0
#         self._speech_rms_n    = 0

#     def _flush(self, reason: str):
#         log.debug(f"FLUSH [{reason}] speech_blocks={self._speech_count} silence_blocks={self._silence_count}")
#         res  = json.loads(self.rec.FinalResult())
#         text = res.get("text", "").strip()
#         log.debug(f"FLUSH result: '{text}'")
#         # Reset để nhận câu tiếp theo sạch
#         self.rec = vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
#         self._silence_count  = 0
#         self._speech_count   = 0
#         self._speech_rms_sum = 0.0
#         self._speech_rms_n   = 0
#         if text:
#             self.result_q.put({"type": "final", "text": text})
#         else:
#             log.debug("FLUSH: empty result, nothing pushed")

#     def run(self):
#         log.info("VoskWorker started")
#         noise_samples     = []
#         block_count       = 0
#         silence_threshold = 300
#         last_partial      = ""

#         while self.running:
#             # Drop backlog
#             backlog = self.audio_q.qsize()
#             if backlog > self.MAX_BACKLOG:
#                 dropped = 0
#                 while self.audio_q.qsize() > self.MAX_BACKLOG:
#                     try: self.audio_q.get_nowait(); dropped += 1
#                     except: break
#                 log.warning(f"DROP {dropped} blocks (backlog={backlog})")

#             try:
#                 data = self.audio_q.get(timeout=0.1)
#             except queue.Empty:
#                 continue

#             block_count += 1
#             pcm = np.frombuffer(data, dtype=np.int16)
#             rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))

#             # Calibrate noise floor
#             if len(noise_samples) < 50:
#                 noise_samples.append(rms)
#                 silence_threshold = 300
#                 if len(noise_samples) == 50:
#                     floor = float(np.percentile(noise_samples, 30))
#                     silence_threshold = max(100, floor * 2)
#                     log.info(f"Noise calibrated: floor={floor:.0f} threshold={silence_threshold:.0f}")

#             if block_count % 10 == 0:
#                 log.debug(f"block={block_count} rms={rms:.0f} thresh={silence_threshold:.0f} queue={self.audio_q.qsize()}")

#             if self.rec.AcceptWaveform(data):
#                 # Câu hoàn chỉnh → final, xóa partial
#                 res  = json.loads(self.rec.Result())
#                 text = res.get("text", "").strip()
#                 log.info(f"FINAL: '{text}'")
#                 last_partial = ""
#                 self._silence_count = 0
#                 self._speech_count  = 0
#                 if text:
#                     self.result_q.put({"type": "final", "text": text})
#             else:
#                 # Partial — in ngay từng từ mới xuất hiện
#                 res     = json.loads(self.rec.PartialResult())
#                 partial = res.get("partial", "").strip()
#                 if partial != last_partial:
#                     log.debug(f"PARTIAL: '{partial}'")
#                     last_partial = partial
#                     self.result_q.put({"type": "partial", "text": partial})

#         log.info("VoskWorker stopped")

#     def stop(self):
#         log.info("VoskWorker.stop() called")
#         try:
#             res  = json.loads(self.rec.FinalResult())
#             text = res.get("text", "").strip()
#             if text:
#                 log.info(f"stop() final flush: '{text}'")
#                 self.result_q.put({"type": "final", "text": text})
#         except Exception as e:
#             log.error(f"stop() error: {e}")
#         self.running = False


# # ─── Custom Scrollbar ─────────────────────────────────────────────────────────

# class SlimScrollbar(tk.Canvas):
#     W = 7

#     def __init__(self, parent, command=None, **kw):
#         super().__init__(parent, width=self.W, bg=C["surface"],
#                          highlightthickness=0, **kw)
#         self._cmd    = command
#         self._lo     = 0.0
#         self._hi     = 1.0
#         self._drag_y = None
#         self.bind("<Configure>",       lambda e: self._draw())
#         self.bind("<ButtonPress-1>",   self._press)
#         self.bind("<B1-Motion>",       self._drag)
#         self.bind("<ButtonRelease-1>", self._release)
#         self.bind("<Enter>",           self._hover_on)
#         self.bind("<Leave>",           self._hover_off)

#     def set(self, lo, hi):
#         self._lo, self._hi = float(lo), float(hi)
#         self._draw()

#     def _draw(self, hover=False):
#         self.delete("all")
#         h = self.winfo_height()
#         w = self.winfo_width()
#         if h < 2 or (self._hi - self._lo) >= 1.0:
#             return
#         ty  = max(2,   int(self._lo * h))
#         by  = min(h-2, max(ty + 20, int(self._hi * h)))
#         col = C["scroll_hv"] if hover else C["scrollbar"]
#         pts = [2+3,ty, w-2-3,ty, w-2,ty, w-2,ty+3,
#                w-2,by-3, w-2,by, w-2-3,by, 2+3,by,
#                2,by, 2,by-3, 2,ty+3, 2,ty, 2+3,ty]
#         self.create_polygon(pts, smooth=True, fill=col)

#     def _press(self, e):   self._drag_y = e.y
#     def _release(self, e): self._drag_y = None
#     def _hover_on(self, e):  self._draw(hover=True)
#     def _hover_off(self, e): self._draw(hover=False)

#     def _drag(self, e):
#         if self._drag_y is None or not self._cmd:
#             return
#         h    = self.winfo_height()
#         diff = (e.y - self._drag_y) / max(h, 1)
#         self._drag_y = e.y
#         span = self._hi - self._lo
#         new  = max(0.0, min(1.0 - span, self._lo + diff))
#         self._cmd("moveto", new)


# # ─── App ─────────────────────────────────────────────────────────────────────

# class App(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("Russian STT")
#         self.geometry("860x660")
#         self.minsize(640, 480)
#         self.configure(bg=C["bg"])

#         # Model load một lần, dùng lại mãi
#         self._vosk_model    = None

#         self.worker         = None
#         self.stream         = None
#         # audio_q và result_q tạo một lần, dùng chung
#         self.audio_q        = queue.Queue()
#         self.result_q       = queue.Queue()
#         self.is_recording   = False
#         self.is_paused      = False
#         self.session_log    = []
#         self.word_count     = 0
#         self._session_start = 0.0
#         self._partial_start = None

#         self._setup_fonts()
#         self._setup_styles()
#         self._build_ui()
#         self._load_model_async()
#         self.after(30, self._poll_results)

#     # ── Fonts & styles ────────────────────────────────────────────────────────

#     def _setup_fonts(self):
#         self.fn_title = tkfont.Font(family="Segoe UI", size=14, weight="bold")
#         self.fn_sub   = tkfont.Font(family="Segoe UI", size=9)
#         self.fn_body  = tkfont.Font(family="Segoe UI", size=15)
#         self.fn_small = tkfont.Font(family="Segoe UI", size=9)
#         self.fn_btn   = tkfont.Font(family="Segoe UI", size=10, weight="bold")
#         self.fn_stat  = tkfont.Font(family="Segoe UI", size=13, weight="bold")

#     def _setup_styles(self):
#         s = ttk.Style()
#         s.theme_use("clam")
#         s.configure("Vol.Horizontal.TProgressbar",
#                     troughcolor=C["panel"], background=C["green"],
#                     bordercolor=C["panel"], lightcolor=C["green"],
#                     darkcolor=C["green"], thickness=4)

#     # ── UI ────────────────────────────────────────────────────────────────────

#     def _build_ui(self):
#         # Header
#         hdr = tk.Frame(self, bg=C["surface"], padx=20, pady=14)
#         hdr.pack(fill="x")

#         left = tk.Frame(hdr, bg=C["surface"])
#         left.pack(side="left")
#         tk.Label(left, text="Russian STT", font=self.fn_title,
#                  bg=C["surface"], fg=C["text"]).pack(anchor="w")
#         tk.Label(left, text="Vosk  ·  real-time  ·  offline  ·  vosk-model-ru-0.42",
#                  font=self.fn_sub, bg=C["surface"], fg=C["muted"]).pack(anchor="w")

#         right = tk.Frame(hdr, bg=C["surface"])
#         right.pack(side="right")
#         tk.Label(right, text="MIC", font=self.fn_small,
#                  bg=C["surface"], fg=C["muted"]).pack(anchor="e")
#         self.vol_var = tk.DoubleVar(value=0)
#         ttk.Progressbar(right, variable=self.vol_var, maximum=100,
#                         length=120, style="Vol.Horizontal.TProgressbar"
#                         ).pack(anchor="e", pady=(3, 0))

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Status bar
#         sbar = tk.Frame(self, bg=C["panel"], padx=16, pady=7)
#         sbar.pack(fill="x")
#         self._dot = tk.Label(sbar, text="●", font=self.fn_small,
#                              bg=C["panel"], fg=C["muted"])
#         self._dot.pack(side="left", padx=(0, 7))
#         self.status_var = tk.StringVar(value="Đang tải model Vosk…")
#         tk.Label(sbar, textvariable=self.status_var, font=self.fn_small,
#                  bg=C["panel"], fg=C["muted"], anchor="w").pack(side="left")

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Toolbar — pack TRƯỚC txt_wrap
#         self._build_toolbar()

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Text area
#         txt_wrap = tk.Frame(self, bg=C["bg"], padx=16, pady=12)
#         txt_wrap.pack(fill="both", expand=True)

#         lbl_row = tk.Frame(txt_wrap, bg=C["bg"])
#         lbl_row.pack(fill="x", pady=(0, 6))
#         tk.Label(lbl_row, text="Văn bản nhận dạng",
#                  font=self.fn_small, bg=C["bg"], fg=C["muted"]).pack(side="left")
#         self._word_lbl = tk.Label(lbl_row, text="0 từ",
#                                   font=self.fn_small, bg=C["bg"], fg=C["muted"])
#         self._word_lbl.pack(side="right")

#         box = tk.Frame(txt_wrap, bg=C["surface"],
#                        highlightbackground=C["border"], highlightthickness=1)
#         box.pack(fill="both", expand=True)

#         self.text_area = tk.Text(
#             box,
#             font=self.fn_body,
#             bg=C["surface"], fg=C["text"],
#             insertbackground=C["text"],
#             selectbackground=C["accent"],
#             relief="flat", bd=0,
#             wrap=tk.WORD,
#             padx=18, pady=14,
#             spacing1=2, spacing3=8,
#             cursor="arrow",
#             state="disabled",
#         )
#         self.text_area.pack(side="left", fill="both", expand=True)

#         sb = SlimScrollbar(box, command=self.text_area.yview)
#         sb.pack(side="right", fill="y", padx=(0, 3), pady=6)
#         self.text_area.configure(yscrollcommand=sb.set)

#         self.text_area.tag_config("final",   foreground=C["text"],  font=self.fn_body)
#         self.text_area.tag_config("partial", foreground=C["partial"], font=self.fn_body)
#         self.text_area.tag_config("pause",   foreground=C["amber"],   font=self.fn_small)

#     def _build_toolbar(self):
#         bar = tk.Frame(self, bg=C["surface"], padx=16, pady=10)
#         bar.pack(fill="x")

#         self.rec_btn = self._btn(bar, "⏺  Ghi âm", C["red"],
#                                  self.toggle_recording, state="disabled")
#         self.rec_btn.pack(side="left", padx=(0, 6))

#         self.pause_btn = self._btn(bar, "⏸  Tạm dừng", C["amber"],
#                                    self.toggle_pause, state="disabled")
#         self.pause_btn.pack(side="left", padx=(0, 18))

#         tk.Frame(bar, bg=C["border"], width=1, height=22).pack(side="left", padx=(0, 18))

#         self._btn(bar, "💾  Lưu", C["panel"], self.save_session).pack(side="left", padx=(0, 6))
#         self._btn(bar, "🗑  Xóa", C["panel"], self.clear_text ).pack(side="left")

#         stat = tk.Frame(bar, bg=C["surface"])
#         stat.pack(side="right")
#         self._time_var = tk.StringVar(value="00:00")
#         tk.Label(stat, textvariable=self._time_var, font=self.fn_stat,
#                  bg=C["surface"], fg=C["accent"]).pack()
#         tk.Label(stat, text="phiên", font=self.fn_small,
#                  bg=C["surface"], fg=C["muted"]).pack()

#     def _btn(self, parent, text, bg, cmd, state="normal"):
#         b = tk.Button(parent, text=text, font=self.fn_btn,
#                       bg=bg, fg=C["text"], activebackground=bg,
#                       activeforeground=C["text"], relief="flat",
#                       padx=14, pady=6, cursor="hand2",
#                       command=cmd, state=state, bd=0, highlightthickness=0)
#         def _hl(c): return "#{:02x}{:02x}{:02x}".format(
#             *[min(255, int(c.lstrip("#")[i:i+2], 16) + 22) for i in (0, 2, 4)])
#         b.bind("<Enter>", lambda e: b.config(bg=_hl(bg)))
#         b.bind("<Leave>", lambda e: b.config(bg=bg))
#         return b

#     # ── Model load — chỉ chạy MỘT LẦN ───────────────────────────────────────

#     def _load_model_async(self):
#         self.status_var.set("Đang tải model Vosk… (chỉ lần đầu)")
#         self._dot.config(fg=C["amber"])

#         def _load():
#             try:
#                 vosk.SetLogLevel(-1)
#                 model = vosk.Model(MODEL_PATH)
#                 self._vosk_model = model
#                 self.after(0, self._on_ready)
#             except Exception as e:
#                 err = str(e)
#                 self.after(0, lambda: self._on_error(err))

#         threading.Thread(target=_load, daemon=True).start()

#     def _on_ready(self):
#         self.status_var.set("Sẵn sàng  ·  Vosk  ·  real-time")
#         self._dot.config(fg=C["green"])
#         self.rec_btn.config(state="normal")

#     def _on_error(self, err):
#         self.status_var.set(f"Lỗi tải model: {err[:60]}")
#         self._dot.config(fg=C["red"])
#         messagebox.showerror("Không tải được model",
#                              f"Kiểm tra thư mục:\n{MODEL_PATH}\n\n{err}")

#     # ── Recording ─────────────────────────────────────────────────────────────

#     def toggle_recording(self):
#         if self.is_recording: self.stop_recording()
#         else:                  self.start_recording()

#     def start_recording(self):
#         self.is_recording   = True
#         self.is_paused      = False
#         self._session_start = time.time()

#         # Xả queue cũ còn sót
#         while not self.audio_q.empty():
#             self.audio_q.get_nowait()
#         while not self.result_q.empty():
#             self.result_q.get_nowait()

#         # Worker mới — chỉ tạo KaldiRecognizer, KHÔNG load model lại
#         self.worker = VoskWorker(self._vosk_model, self.audio_q, self.result_q)
#         self.worker.start()

#         self.rec_btn.config(text="⏹  Dừng", bg=C["green"])
#         self.pause_btn.config(state="normal")
#         self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
#         self._dot.config(fg=C["red"])
#         self._tick()

#         _cb_count = [0]
#         def _cb(indata, frames, t, status):
#             if status:
#                 log.warning(f"sounddevice status: {status}")
#             if self.is_paused:
#                 return
#             # Normalize audio lên RMS ~3000 để Vosk nhận tốt hơn
#             samples = indata[:, 0].astype(np.float32)
#             rms_f   = float(np.sqrt(np.mean(samples ** 2))) + 1e-9
#             target  = 0.1  # float32 target (~3000/32767)
#             gain    = min(target / rms_f, 20.0)  # cap gain tối đa 20x
#             samples = np.clip(samples * gain, -1.0, 1.0)
#             pcm     = (samples * 32767).astype("int16").tobytes()
#             self.audio_q.put_nowait(pcm)
#             _cb_count[0] += 1
#             if _cb_count[0] % 25 == 0:  # log mỗi 2s
#                 log.debug(f"audio_cb: block={_cb_count[0]} queue={self.audio_q.qsize()}")
#             rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
#             self.after(0, lambda: self.vol_var.set(min(100, rms * 2500)))

#         self.stream = sd.InputStream(
#             samplerate=SAMPLE_RATE, channels=1,
#             dtype="float32", blocksize=BLOCK_SIZE,
#             callback=_cb,
#         )
#         self.stream.start()

#     def stop_recording(self):
#         self.is_recording = False
#         self.is_paused    = False
#         if self.stream:
#             self.stream.stop(); self.stream.close(); self.stream = None
#         if self.worker:
#             self.worker.stop(); self.worker = None
#         self.vol_var.set(0)
#         self.rec_btn.config(text="⏺  Ghi âm", bg=C["red"])
#         self.pause_btn.config(state="disabled", text="⏸  Tạm dừng", bg=C["amber"])
#         self.status_var.set("Đã dừng  ·  Nhấn để ghi âm tiếp")
#         self._dot.config(fg=C["muted"])
#         self._clear_partial()

#     def toggle_pause(self):
#         self.is_paused = not self.is_paused
#         if self.is_paused:
#             self.pause_btn.config(text="▶  Tiếp tục", bg=C["accent"])
#             self.status_var.set("Tạm dừng")
#             self._dot.config(fg=C["amber"])
#             self.vol_var.set(0)
#             self._clear_partial()
#             self._insert_pause_marker()
#         else:
#             self.pause_btn.config(text="⏸  Tạm dừng", bg=C["amber"])
#             self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
#             self._dot.config(fg=C["red"])

#     # ── Timer ─────────────────────────────────────────────────────────────────

#     def _tick(self):
#         if not self.is_recording:
#             return
#         e = int(time.time() - self._session_start)
#         self._time_var.set(f"{e//60:02d}:{e%60:02d}")
#         self.after(1000, self._tick)

#     # ── Results ───────────────────────────────────────────────────────────────

#     def _poll_results(self):
#         try:
#             while True:
#                 msg = self.result_q.get_nowait()
#                 if msg["type"] == "partial":
#                     self._show_partial(msg["text"])
#                 else:
#                     self._commit_final(msg["text"])
#         except queue.Empty:
#             pass
#         self.after(30, self._poll_results)

#     def _show_partial(self, text: str):
#         """Xóa partial cũ, hiện partial mới màu xám."""
#         self.text_area.config(state="normal")
#         if self._partial_start:
#             self.text_area.delete(self._partial_start, "end")
#         if text:
#             self._partial_start = self.text_area.index("end-1c")
#             self.text_area.insert("end", text, "partial")
#             self.text_area.see("end")
#         else:
#             self._partial_start = None
#         self.text_area.config(state="disabled")

#     def _commit_final(self, text: str):
#         """Xóa partial, ghi final màu trắng + xuống dòng."""
#         log.info(f"UI commit: '{text}'")
#         self.text_area.config(state="normal")
#         if self._partial_start:
#             self.text_area.delete(self._partial_start, "end")
#             self._partial_start = None
#         self.text_area.insert("end", text + "\n", "final")
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")
#         self.session_log.append(text)
#         self.word_count += len(text.split())
#         self._word_lbl.config(text=f"{self.word_count} từ")

#     def _clear_partial(self):
#         if self._partial_start:
#             self.text_area.config(state="normal")
#             self.text_area.delete(self._partial_start, "end")
#             self.text_area.config(state="disabled")
#             self._partial_start = None

#     def _insert_pause_marker(self):
#         ts = datetime.datetime.now().strftime("%H:%M:%S")
#         self.text_area.config(state="normal")
#         self.text_area.insert("end", f"\n— tạm dừng {ts} —\n", "pause")
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")

#     # ── Helpers ───────────────────────────────────────────────────────────────

#     def clear_text(self):
#         self.text_area.config(state="normal")
#         self.text_area.delete("1.0", "end")
#         self.text_area.config(state="disabled")
#         self.session_log.clear()
#         self.word_count     = 0
#         self._partial_start = None
#         self._word_lbl.config(text="0 từ")

#     def save_session(self):
#         if not self.session_log:
#             messagebox.showinfo("Thông báo", "Chưa có nội dung để lưu.")
#             return
#         default = f"session_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
#         path = filedialog.asksaveasfilename(
#             defaultextension=".txt",
#             filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
#             initialfile=default, title="Lưu phiên ghi âm",
#         )
#         if not path:
#             return
#         with open(path, "w", encoding="utf-8") as f:
#             f.write(f"# Russian STT — {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
#             f.write(f"# Từ: {self.word_count}\n\n")
#             f.write(" ".join(self.session_log))
#         messagebox.showinfo("Đã lưu", f"Lưu thành công:\n{path}")

#     def on_closing(self):
#         if self.is_recording:
#             self.stop_recording()
#         self.destroy()


# # ─── Entry point ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     app = App()
#     app.protocol("WM_DELETE_WINDOW", app.on_closing)
#     app.mainloop()


# """
# Russian STT — Vosk real-time, từng từ hiện ngay
# Cấu trúc thư mục:
#   STT/
#   └── server/
#       ├── app.py
#       └── model/   ← vosk-model-ru-0.42
# """

# import os
# import threading
# import queue
# import json
# import logging
# import tkinter as tk
# import tkinter.font as tkfont
# from tkinter import ttk, messagebox, filedialog
# import sounddevice as sd
# import vosk
# import datetime
# import time
# import numpy as np

# # ─── Debug logger ─────────────────────────────────────────────────────────────
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s.%(msecs)03d [%(threadName)s] %(levelname)s — %(message)s",
#     datefmt="%H:%M:%S",
#     handlers=[
#         logging.FileHandler("debug.log", encoding="utf-8", mode="w"),
#         logging.StreamHandler(),   # in ra terminal luôn
#     ]
# )
# log = logging.getLogger("STT")

# # ─── Cấu hình ────────────────────────────────────────────────────────────────

# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH  = os.path.join(_SCRIPT_DIR, "model")
# SAMPLE_RATE = 16000
# BLOCK_MS    = 80
# BLOCK_SIZE  = int(SAMPLE_RATE * BLOCK_MS / 1000)

# # ─── Màu sắc ─────────────────────────────────────────────────────────────────

# C = {
#     "bg":        "#0f1117",
#     "surface":   "#1a1d27",
#     "panel":     "#21253a",
#     "border":    "#2d3148",
#     "text":      "#e8eaf0",
#     "muted":     "#5a6070",
#     "accent":    "#6c8fff",
#     "green":     "#3ecf72",
#     "red":       "#ef4444",
#     "amber":     "#f59e0b",
#     "partial":   "#94a3b8",
#     "scrollbar": "#2d3148",
#     "scroll_hv": "#4a5080",
# }

# # ─── Vosk Worker ─────────────────────────────────────────────────────────────

# class VoskWorker(threading.Thread):
#     """
#     Nhận model đã load sẵn — KHÔNG load lại model mỗi lần record.
#     Flush khi:
#       1. Vosk tự AcceptWaveform = True (câu hoàn chỉnh)
#       2. Im lặng >= 400ms  (người dừng nói)
#       3. Nói liên tục >= 4s (force flush để không tích lũy quá lâu)
#     """
#     # Tất cả tính theo số BLOCK (1 block = BLOCK_MS ms)
#     SILENCE_BLOCKS  = max(1, int(500  / BLOCK_MS))   # 400ms im lặng → flush
#     MAX_SPEECH_BLOCKS = max(1, int(3000 / BLOCK_MS)) # 4s nói liên tục → force flush
#     MAX_BACKLOG     = 10   # drop nếu queue > 10 block

#     def __init__(self, model: vosk.Model, audio_q: queue.Queue,
#                  result_q: queue.Queue):
#         super().__init__(daemon=True)
#         self.audio_q          = audio_q
#         self.result_q         = result_q
#         self.running          = True
#         self._model           = model   # giữ ref để tạo lại recognizer sau flush
#         self.rec              = vosk.KaldiRecognizer(model, SAMPLE_RATE)
#         self._silence_count   = 0
#         self._speech_count    = 0   # tổng block có tiếng kể từ lần flush cuối
#         self._speech_rms_sum  = 0.0
#         self._speech_rms_n    = 0

#     def _flush(self, reason: str):
#         log.debug(f"FLUSH [{reason}] speech_blocks={self._speech_count} silence_blocks={self._silence_count}")
#         res  = json.loads(self.rec.FinalResult())
#         text = res.get("text", "").strip()
#         log.debug(f"FLUSH result: '{text}'")
#         # Reset để nhận câu tiếp theo sạch
#         self.rec = vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
#         self._silence_count  = 0
#         self._speech_count   = 0
#         self._speech_rms_sum = 0.0
#         self._speech_rms_n   = 0
#         if text:
#             self.result_q.put({"type": "final", "text": text})
#         else:
#             log.debug("FLUSH: empty result, nothing pushed")

#     def run(self):
#         log.info("VoskWorker started")
#         noise_samples     = []
#         block_count       = 0
#         silence_threshold = 300

#         while self.running:
#             # Drop backlog
#             backlog = self.audio_q.qsize()
#             if backlog > self.MAX_BACKLOG:
#                 dropped = 0
#                 while self.audio_q.qsize() > self.MAX_BACKLOG:
#                     try: self.audio_q.get_nowait(); dropped += 1
#                     except: break
#                 log.warning(f"DROP {dropped} blocks (backlog={backlog})")

#             try:
#                 data = self.audio_q.get(timeout=0.1)
#             except queue.Empty:
#                 continue

#             block_count += 1
#             pcm = np.frombuffer(data, dtype=np.int16)
#             rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))

#             # Calibrate noise floor
#             if len(noise_samples) < 50:
#                 noise_samples.append(rms)
#                 silence_threshold = 300
#                 if len(noise_samples) == 50:
#                     floor = float(np.percentile(noise_samples, 30))
#                     silence_threshold = max(100, floor * 2)   # nhạy hơn: *2 thay vì *3, min 100
#                     log.info(f"Noise calibrated: floor={floor:.0f} threshold={silence_threshold:.0f}")
            
#             is_speech = rms > silence_threshold

#             # Log mỗi 10 block (~800ms) để không spam
#             if block_count % 10 == 0:
#                 log.debug(f"block={block_count} rms={rms:.0f} thresh={silence_threshold:.0f} "
#                           f"is_speech={is_speech} speech_cnt={self._speech_count} "
#                           f"silence_cnt={self._silence_count} queue={self.audio_q.qsize()}")

#             if is_speech:
#                 self._silence_count = 0
#                 self._speech_count += 1
#             else:
#                 self._silence_count += 1

#             accepted = self.rec.AcceptWaveform(data)
#             if accepted:
#                 # KHÔNG Reset() — để Vosk tự quản lý state
#                 res  = json.loads(self.rec.Result())
#                 text = res.get("text", "").strip()
#                 log.info(f"VOSK_ACCEPT: '{text}' (speech_blocks={self._speech_count})")
#                 self._silence_count = 0
#                 self._speech_count  = 0
#                 if text:
#                     self.result_q.put({"type": "final", "text": text})

#             elif self._speech_count > 0 and self._silence_count >= self.SILENCE_BLOCKS:
#                 self._flush("SILENCE")

#             elif self._speech_count >= self.MAX_SPEECH_BLOCKS:
#                 self._flush("MAX_SPEECH")

#         log.info("VoskWorker stopped")

#     def stop(self):
#         log.info("VoskWorker.stop() called")
#         try:
#             res  = json.loads(self.rec.FinalResult())
#             text = res.get("text", "").strip()
#             if text:
#                 log.info(f"stop() final flush: '{text}'")
#                 self.result_q.put({"type": "final", "text": text})
#         except Exception as e:
#             log.error(f"stop() error: {e}")
#         self.running = False


# # ─── Custom Scrollbar ─────────────────────────────────────────────────────────

# class SlimScrollbar(tk.Canvas):
#     W = 7

#     def __init__(self, parent, command=None, **kw):
#         super().__init__(parent, width=self.W, bg=C["surface"],
#                          highlightthickness=0, **kw)
#         self._cmd    = command
#         self._lo     = 0.0
#         self._hi     = 1.0
#         self._drag_y = None
#         self.bind("<Configure>",       lambda e: self._draw())
#         self.bind("<ButtonPress-1>",   self._press)
#         self.bind("<B1-Motion>",       self._drag)
#         self.bind("<ButtonRelease-1>", self._release)
#         self.bind("<Enter>",           self._hover_on)
#         self.bind("<Leave>",           self._hover_off)

#     def set(self, lo, hi):
#         self._lo, self._hi = float(lo), float(hi)
#         self._draw()

#     def _draw(self, hover=False):
#         self.delete("all")
#         h = self.winfo_height()
#         w = self.winfo_width()
#         if h < 2 or (self._hi - self._lo) >= 1.0:
#             return
#         ty  = max(2,   int(self._lo * h))
#         by  = min(h-2, max(ty + 20, int(self._hi * h)))
#         col = C["scroll_hv"] if hover else C["scrollbar"]
#         pts = [2+3,ty, w-2-3,ty, w-2,ty, w-2,ty+3,
#                w-2,by-3, w-2,by, w-2-3,by, 2+3,by,
#                2,by, 2,by-3, 2,ty+3, 2,ty, 2+3,ty]
#         self.create_polygon(pts, smooth=True, fill=col)

#     def _press(self, e):   self._drag_y = e.y
#     def _release(self, e): self._drag_y = None
#     def _hover_on(self, e):  self._draw(hover=True)
#     def _hover_off(self, e): self._draw(hover=False)

#     def _drag(self, e):
#         if self._drag_y is None or not self._cmd:
#             return
#         h    = self.winfo_height()
#         diff = (e.y - self._drag_y) / max(h, 1)
#         self._drag_y = e.y
#         span = self._hi - self._lo
#         new  = max(0.0, min(1.0 - span, self._lo + diff))
#         self._cmd("moveto", new)


# # ─── App ─────────────────────────────────────────────────────────────────────

# class App(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("Russian STT")
#         self.geometry("860x660")
#         self.minsize(640, 480)
#         self.configure(bg=C["bg"])

#         # Model load một lần, dùng lại mãi
#         self._vosk_model    = None

#         self.worker         = None
#         self.stream         = None
#         # audio_q và result_q tạo một lần, dùng chung
#         self.audio_q        = queue.Queue()
#         self.result_q       = queue.Queue()
#         self.is_recording   = False
#         self.is_paused      = False
#         self.session_log    = []
#         self.word_count     = 0
#         self._session_start = 0.0
#         self._partial_start = None

#         self._setup_fonts()
#         self._setup_styles()
#         self._build_ui()
#         self._load_model_async()
#         self.after(30, self._poll_results)

#     # ── Fonts & styles ────────────────────────────────────────────────────────

#     def _setup_fonts(self):
#         self.fn_title = tkfont.Font(family="Segoe UI", size=14, weight="bold")
#         self.fn_sub   = tkfont.Font(family="Segoe UI", size=9)
#         self.fn_body  = tkfont.Font(family="Segoe UI", size=15)
#         self.fn_small = tkfont.Font(family="Segoe UI", size=9)
#         self.fn_btn   = tkfont.Font(family="Segoe UI", size=10, weight="bold")
#         self.fn_stat  = tkfont.Font(family="Segoe UI", size=13, weight="bold")

#     def _setup_styles(self):
#         s = ttk.Style()
#         s.theme_use("clam")
#         s.configure("Vol.Horizontal.TProgressbar",
#                     troughcolor=C["panel"], background=C["green"],
#                     bordercolor=C["panel"], lightcolor=C["green"],
#                     darkcolor=C["green"], thickness=4)

#     # ── UI ────────────────────────────────────────────────────────────────────

#     def _build_ui(self):
#         # Header
#         hdr = tk.Frame(self, bg=C["surface"], padx=20, pady=14)
#         hdr.pack(fill="x")

#         left = tk.Frame(hdr, bg=C["surface"])
#         left.pack(side="left")
#         tk.Label(left, text="Russian STT", font=self.fn_title,
#                  bg=C["surface"], fg=C["text"]).pack(anchor="w")
#         tk.Label(left, text="Vosk  ·  real-time  ·  offline  ·  vosk-model-ru-0.42",
#                  font=self.fn_sub, bg=C["surface"], fg=C["muted"]).pack(anchor="w")

#         right = tk.Frame(hdr, bg=C["surface"])
#         right.pack(side="right")
#         tk.Label(right, text="MIC", font=self.fn_small,
#                  bg=C["surface"], fg=C["muted"]).pack(anchor="e")
#         self.vol_var = tk.DoubleVar(value=0)
#         ttk.Progressbar(right, variable=self.vol_var, maximum=100,
#                         length=120, style="Vol.Horizontal.TProgressbar"
#                         ).pack(anchor="e", pady=(3, 0))

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Status bar
#         sbar = tk.Frame(self, bg=C["panel"], padx=16, pady=7)
#         sbar.pack(fill="x")
#         self._dot = tk.Label(sbar, text="●", font=self.fn_small,
#                              bg=C["panel"], fg=C["muted"])
#         self._dot.pack(side="left", padx=(0, 7))
#         self.status_var = tk.StringVar(value="Đang tải model Vosk…")
#         tk.Label(sbar, textvariable=self.status_var, font=self.fn_small,
#                  bg=C["panel"], fg=C["muted"], anchor="w").pack(side="left")

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Toolbar — pack TRƯỚC txt_wrap
#         self._build_toolbar()

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Text area
#         txt_wrap = tk.Frame(self, bg=C["bg"], padx=16, pady=12)
#         txt_wrap.pack(fill="both", expand=True)

#         lbl_row = tk.Frame(txt_wrap, bg=C["bg"])
#         lbl_row.pack(fill="x", pady=(0, 6))
#         tk.Label(lbl_row, text="Văn bản nhận dạng",
#                  font=self.fn_small, bg=C["bg"], fg=C["muted"]).pack(side="left")
#         self._word_lbl = tk.Label(lbl_row, text="0 từ",
#                                   font=self.fn_small, bg=C["bg"], fg=C["muted"])
#         self._word_lbl.pack(side="right")

#         box = tk.Frame(txt_wrap, bg=C["surface"],
#                        highlightbackground=C["border"], highlightthickness=1)
#         box.pack(fill="both", expand=True)

#         self.text_area = tk.Text(
#             box,
#             font=self.fn_body,
#             bg=C["surface"], fg=C["text"],
#             insertbackground=C["text"],
#             selectbackground=C["accent"],
#             relief="flat", bd=0,
#             wrap=tk.WORD,
#             padx=18, pady=14,
#             spacing1=2, spacing3=8,
#             cursor="arrow",
#             state="disabled",
#         )
#         self.text_area.pack(side="left", fill="both", expand=True)

#         sb = SlimScrollbar(box, command=self.text_area.yview)
#         sb.pack(side="right", fill="y", padx=(0, 3), pady=6)
#         self.text_area.configure(yscrollcommand=sb.set)

#         self.text_area.tag_config("final",   foreground=C["text"],  font=self.fn_body)
#         self.text_area.tag_config("partial", foreground=C["partial"], font=self.fn_body)
#         self.text_area.tag_config("pause",   foreground=C["amber"],   font=self.fn_small)

#     def _build_toolbar(self):
#         bar = tk.Frame(self, bg=C["surface"], padx=16, pady=10)
#         bar.pack(fill="x")

#         self.rec_btn = self._btn(bar, "⏺  Ghi âm", C["red"],
#                                  self.toggle_recording, state="disabled")
#         self.rec_btn.pack(side="left", padx=(0, 6))

#         self.pause_btn = self._btn(bar, "⏸  Tạm dừng", C["amber"],
#                                    self.toggle_pause, state="disabled")
#         self.pause_btn.pack(side="left", padx=(0, 18))

#         tk.Frame(bar, bg=C["border"], width=1, height=22).pack(side="left", padx=(0, 18))

#         self._btn(bar, "💾  Lưu", C["panel"], self.save_session).pack(side="left", padx=(0, 6))
#         self._btn(bar, "🗑  Xóa", C["panel"], self.clear_text ).pack(side="left")

#         stat = tk.Frame(bar, bg=C["surface"])
#         stat.pack(side="right")
#         self._time_var = tk.StringVar(value="00:00")
#         tk.Label(stat, textvariable=self._time_var, font=self.fn_stat,
#                  bg=C["surface"], fg=C["accent"]).pack()
#         tk.Label(stat, text="phiên", font=self.fn_small,
#                  bg=C["surface"], fg=C["muted"]).pack()

#     def _btn(self, parent, text, bg, cmd, state="normal"):
#         b = tk.Button(parent, text=text, font=self.fn_btn,
#                       bg=bg, fg=C["text"], activebackground=bg,
#                       activeforeground=C["text"], relief="flat",
#                       padx=14, pady=6, cursor="hand2",
#                       command=cmd, state=state, bd=0, highlightthickness=0)
#         def _hl(c): return "#{:02x}{:02x}{:02x}".format(
#             *[min(255, int(c.lstrip("#")[i:i+2], 16) + 22) for i in (0, 2, 4)])
#         b.bind("<Enter>", lambda e: b.config(bg=_hl(bg)))
#         b.bind("<Leave>", lambda e: b.config(bg=bg))
#         return b

#     # ── Model load — chỉ chạy MỘT LẦN ───────────────────────────────────────

#     def _load_model_async(self):
#         self.status_var.set("Đang tải model Vosk… (chỉ lần đầu)")
#         self._dot.config(fg=C["amber"])

#         def _load():
#             try:
#                 vosk.SetLogLevel(-1)
#                 model = vosk.Model(MODEL_PATH)
#                 self._vosk_model = model
#                 self.after(0, self._on_ready)
#             except Exception as e:
#                 err = str(e)
#                 self.after(0, lambda: self._on_error(err))

#         threading.Thread(target=_load, daemon=True).start()

#     def _on_ready(self):
#         self.status_var.set("Sẵn sàng  ·  Vosk  ·  real-time")
#         self._dot.config(fg=C["green"])
#         self.rec_btn.config(state="normal")

#     def _on_error(self, err):
#         self.status_var.set(f"Lỗi tải model: {err[:60]}")
#         self._dot.config(fg=C["red"])
#         messagebox.showerror("Không tải được model",
#                              f"Kiểm tra thư mục:\n{MODEL_PATH}\n\n{err}")

#     # ── Recording ─────────────────────────────────────────────────────────────

#     def toggle_recording(self):
#         if self.is_recording: self.stop_recording()
#         else:                  self.start_recording()

#     def start_recording(self):
#         self.is_recording   = True
#         self.is_paused      = False
#         self._session_start = time.time()

#         # Xả queue cũ còn sót
#         while not self.audio_q.empty():
#             self.audio_q.get_nowait()
#         while not self.result_q.empty():
#             self.result_q.get_nowait()

#         # Worker mới — chỉ tạo KaldiRecognizer, KHÔNG load model lại
#         self.worker = VoskWorker(self._vosk_model, self.audio_q, self.result_q)
#         self.worker.start()

#         self.rec_btn.config(text="⏹  Dừng", bg=C["green"])
#         self.pause_btn.config(state="normal")
#         self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
#         self._dot.config(fg=C["red"])
#         self._tick()

#         _cb_count = [0]
#         def _cb(indata, frames, t, status):
#             if status:
#                 log.warning(f"sounddevice status: {status}")
#             if self.is_paused:
#                 return
#             pcm = (indata[:, 0] * 32767).astype("int16").tobytes()
#             self.audio_q.put_nowait(pcm)
#             _cb_count[0] += 1
#             if _cb_count[0] % 25 == 0:  # log mỗi 2s
#                 log.debug(f"audio_cb: block={_cb_count[0]} queue={self.audio_q.qsize()}")
#             rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
#             self.after(0, lambda: self.vol_var.set(min(100, rms * 2500)))

#         self.stream = sd.InputStream(
#             samplerate=SAMPLE_RATE, channels=1,
#             dtype="float32", blocksize=BLOCK_SIZE,
#             callback=_cb,
#         )
#         self.stream.start()

#     def stop_recording(self):
#         self.is_recording = False
#         self.is_paused    = False
#         if self.stream:
#             self.stream.stop(); self.stream.close(); self.stream = None
#         if self.worker:
#             self.worker.stop(); self.worker = None
#         self.vol_var.set(0)
#         self.rec_btn.config(text="⏺  Ghi âm", bg=C["red"])
#         self.pause_btn.config(state="disabled", text="⏸  Tạm dừng", bg=C["amber"])
#         self.status_var.set("Đã dừng  ·  Nhấn để ghi âm tiếp")
#         self._dot.config(fg=C["muted"])
#         self._clear_partial()

#     def toggle_pause(self):
#         self.is_paused = not self.is_paused
#         if self.is_paused:
#             self.pause_btn.config(text="▶  Tiếp tục", bg=C["accent"])
#             self.status_var.set("Tạm dừng")
#             self._dot.config(fg=C["amber"])
#             self.vol_var.set(0)
#             self._clear_partial()
#             self._insert_pause_marker()
#         else:
#             self.pause_btn.config(text="⏸  Tạm dừng", bg=C["amber"])
#             self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
#             self._dot.config(fg=C["red"])

#     # ── Timer ─────────────────────────────────────────────────────────────────

#     def _tick(self):
#         if not self.is_recording:
#             return
#         e = int(time.time() - self._session_start)
#         self._time_var.set(f"{e//60:02d}:{e%60:02d}")
#         self.after(1000, self._tick)

#     # ── Results ───────────────────────────────────────────────────────────────

#     def _poll_results(self):
#         try:
#             while True:
#                 msg = self.result_q.get_nowait()
#                 if msg["type"] == "final":
#                     self._commit_final(msg["text"])
#         except queue.Empty:
#             pass
#         self.after(30, self._poll_results)

#     def _commit_final(self, text: str):
#         log.info(f"UI commit: '{text}'")
#         self.text_area.config(state="normal")
#         self.text_area.insert("end", text + "\n", "final")
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")
#         self.session_log.append(text)
#         self.word_count += len(text.split())
#         self._word_lbl.config(text=f"{self.word_count} từ")

#     def _clear_partial(self):
#         pass  # không còn partial

#     def _insert_pause_marker(self):
#         ts = datetime.datetime.now().strftime("%H:%M:%S")
#         self.text_area.config(state="normal")
#         self.text_area.insert("end", f"\n— tạm dừng {ts} —\n", "pause")
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")

#     # ── Helpers ───────────────────────────────────────────────────────────────

#     def clear_text(self):
#         self.text_area.config(state="normal")
#         self.text_area.delete("1.0", "end")
#         self.text_area.config(state="disabled")
#         self.session_log.clear()
#         self.word_count     = 0
#         self._partial_start = None
#         self._word_lbl.config(text="0 từ")

#     def save_session(self):
#         if not self.session_log:
#             messagebox.showinfo("Thông báo", "Chưa có nội dung để lưu.")
#             return
#         default = f"session_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
#         path = filedialog.asksaveasfilename(
#             defaultextension=".txt",
#             filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
#             initialfile=default, title="Lưu phiên ghi âm",
#         )
#         if not path:
#             return
#         with open(path, "w", encoding="utf-8") as f:
#             f.write(f"# Russian STT — {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
#             f.write(f"# Từ: {self.word_count}\n\n")
#             f.write(" ".join(self.session_log))
#         messagebox.showinfo("Đã lưu", f"Lưu thành công:\n{path}")

#     def on_closing(self):
#         if self.is_recording:
#             self.stop_recording()
#         self.destroy()


# # ─── Entry point ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     app = App()
#     app.protocol("WM_DELETE_WINDOW", app.on_closing)
#     app.mainloop()

# """
# Russian STT — Vosk real-time, từng từ hiện ngay
# Cấu trúc thư mục:
#   STT/
#   └── server/
#       ├── app.py
#       └── model/   ← vosk-model-ru-0.42
# """

# import os
# import threading
# import queue
# import json
# import tkinter as tk
# import tkinter.font as tkfont
# from tkinter import ttk, messagebox, filedialog
# import sounddevice as sd
# import vosk
# import datetime
# import time
# import numpy as np

# # ─── Cấu hình ────────────────────────────────────────────────────────────────

# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH  = os.path.join(_SCRIPT_DIR, "model")
# SAMPLE_RATE = 16000
# BLOCK_MS    = 80
# BLOCK_SIZE  = int(SAMPLE_RATE * BLOCK_MS / 1000)

# # ─── Màu sắc ─────────────────────────────────────────────────────────────────

# C = {
#     "bg":        "#0f1117",
#     "surface":   "#1a1d27",
#     "panel":     "#21253a",
#     "border":    "#2d3148",
#     "text":      "#e8eaf0",
#     "muted":     "#5a6070",
#     "accent":    "#6c8fff",
#     "green":     "#3ecf72",
#     "red":       "#ef4444",
#     "amber":     "#f59e0b",
#     "partial":   "#94a3b8",
#     "scrollbar": "#2d3148",
#     "scroll_hv": "#4a5080",
# }

# # ─── Vosk Worker ─────────────────────────────────────────────────────────────

# class VoskWorker(threading.Thread):
#     """
#     Nhận model đã load sẵn — KHÔNG load lại model mỗi lần record.
#     Flush khi:
#       1. Vosk tự AcceptWaveform = True (câu hoàn chỉnh)
#       2. Im lặng >= 400ms  (người dừng nói)
#       3. Nói liên tục >= 4s (force flush để không tích lũy quá lâu)
#     """
#     # Tất cả tính theo số BLOCK (1 block = BLOCK_MS ms)
#     SILENCE_BLOCKS  = max(1, int(400  / BLOCK_MS))   # 400ms im lặng → flush
#     MAX_SPEECH_BLOCKS = max(1, int(4000 / BLOCK_MS)) # 4s nói liên tục → force flush
#     MAX_BACKLOG     = 10   # drop nếu queue > 10 block

#     def __init__(self, model: vosk.Model, audio_q: queue.Queue,
#                  result_q: queue.Queue):
#         super().__init__(daemon=True)
#         self.audio_q          = audio_q
#         self.result_q         = result_q
#         self.running          = True
#         self.rec              = vosk.KaldiRecognizer(model, SAMPLE_RATE)
#         self._silence_count   = 0
#         self._speech_count    = 0   # tổng block có tiếng kể từ lần flush cuối
#         self._speech_rms_sum  = 0.0
#         self._speech_rms_n    = 0

#     def _flush(self):
#         res  = json.loads(self.rec.FinalResult())
#         text = res.get("text", "").strip()
#         self.rec.Reset()
#         self._silence_count  = 0
#         self._speech_count   = 0
#         self._speech_rms_sum = 0.0
#         self._speech_rms_n   = 0
#         if text:
#             self.result_q.put({"type": "final", "text": text})

#     def run(self):
#         # Tự động hiệu chỉnh ngưỡng silence từ 50 block đầu (noise floor)
#         noise_samples = []

#         while self.running:
#             # Drop backlog
#             while self.audio_q.qsize() > self.MAX_BACKLOG:
#                 try: self.audio_q.get_nowait()
#                 except: break

#             try:
#                 data = self.audio_q.get(timeout=0.1)
#             except queue.Empty:
#                 continue

#             pcm = np.frombuffer(data, dtype=np.int16)
#             rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))

#             # Thu thập noise floor 50 block đầu
#             if len(noise_samples) < 50:
#                 noise_samples.append(rms)
#                 silence_threshold = 300  # default
#             else:
#                 # Ngưỡng = noise floor * 3, tối thiểu 200
#                 silence_threshold = max(200, np.percentile(noise_samples, 30) * 3)

#             is_speech = rms > silence_threshold

#             if is_speech:
#                 self._silence_count = 0
#                 self._speech_count += 1
#             else:
#                 self._silence_count += 1

#             # Feed vào Vosk
#             if self.rec.AcceptWaveform(data):
#                 # Vosk tự nhận ra hết câu
#                 res  = json.loads(self.rec.Result())
#                 text = res.get("text", "").strip()
#                 self.rec.Reset()
#                 self._silence_count  = 0
#                 self._speech_count   = 0
#                 if text:
#                     self.result_q.put({"type": "final", "text": text})

#             elif self._speech_count > 0 and self._silence_count >= self.SILENCE_BLOCKS:
#                 # Dừng nói đủ lâu → flush
#                 self._flush()

#             elif self._speech_count >= self.MAX_SPEECH_BLOCKS:
#                 # Nói quá lâu không dừng → force flush
#                 self._flush()

#     def stop(self):
#         try:
#             res  = json.loads(self.rec.FinalResult())
#             text = res.get("text", "").strip()
#             if text:
#                 self.result_q.put({"type": "final", "text": text})
#         except Exception:
#             pass
#         self.running = False


# # ─── Custom Scrollbar ─────────────────────────────────────────────────────────

# class SlimScrollbar(tk.Canvas):
#     W = 7

#     def __init__(self, parent, command=None, **kw):
#         super().__init__(parent, width=self.W, bg=C["surface"],
#                          highlightthickness=0, **kw)
#         self._cmd    = command
#         self._lo     = 0.0
#         self._hi     = 1.0
#         self._drag_y = None
#         self.bind("<Configure>",       lambda e: self._draw())
#         self.bind("<ButtonPress-1>",   self._press)
#         self.bind("<B1-Motion>",       self._drag)
#         self.bind("<ButtonRelease-1>", self._release)
#         self.bind("<Enter>",           self._hover_on)
#         self.bind("<Leave>",           self._hover_off)

#     def set(self, lo, hi):
#         self._lo, self._hi = float(lo), float(hi)
#         self._draw()

#     def _draw(self, hover=False):
#         self.delete("all")
#         h = self.winfo_height()
#         w = self.winfo_width()
#         if h < 2 or (self._hi - self._lo) >= 1.0:
#             return
#         ty  = max(2,   int(self._lo * h))
#         by  = min(h-2, max(ty + 20, int(self._hi * h)))
#         col = C["scroll_hv"] if hover else C["scrollbar"]
#         pts = [2+3,ty, w-2-3,ty, w-2,ty, w-2,ty+3,
#                w-2,by-3, w-2,by, w-2-3,by, 2+3,by,
#                2,by, 2,by-3, 2,ty+3, 2,ty, 2+3,ty]
#         self.create_polygon(pts, smooth=True, fill=col)

#     def _press(self, e):   self._drag_y = e.y
#     def _release(self, e): self._drag_y = None
#     def _hover_on(self, e):  self._draw(hover=True)
#     def _hover_off(self, e): self._draw(hover=False)

#     def _drag(self, e):
#         if self._drag_y is None or not self._cmd:
#             return
#         h    = self.winfo_height()
#         diff = (e.y - self._drag_y) / max(h, 1)
#         self._drag_y = e.y
#         span = self._hi - self._lo
#         new  = max(0.0, min(1.0 - span, self._lo + diff))
#         self._cmd("moveto", new)


# # ─── App ─────────────────────────────────────────────────────────────────────

# class App(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("Russian STT")
#         self.geometry("860x660")
#         self.minsize(640, 480)
#         self.configure(bg=C["bg"])

#         # Model load một lần, dùng lại mãi
#         self._vosk_model    = None

#         self.worker         = None
#         self.stream         = None
#         # audio_q và result_q tạo một lần, dùng chung
#         self.audio_q        = queue.Queue()
#         self.result_q       = queue.Queue()
#         self.is_recording   = False
#         self.is_paused      = False
#         self.session_log    = []
#         self.word_count     = 0
#         self._session_start = 0.0
#         self._partial_start = None

#         self._setup_fonts()
#         self._setup_styles()
#         self._build_ui()
#         self._load_model_async()
#         self.after(30, self._poll_results)

#     # ── Fonts & styles ────────────────────────────────────────────────────────

#     def _setup_fonts(self):
#         self.fn_title = tkfont.Font(family="Segoe UI", size=14, weight="bold")
#         self.fn_sub   = tkfont.Font(family="Segoe UI", size=9)
#         self.fn_body  = tkfont.Font(family="Segoe UI", size=15)
#         self.fn_small = tkfont.Font(family="Segoe UI", size=9)
#         self.fn_btn   = tkfont.Font(family="Segoe UI", size=10, weight="bold")
#         self.fn_stat  = tkfont.Font(family="Segoe UI", size=13, weight="bold")

#     def _setup_styles(self):
#         s = ttk.Style()
#         s.theme_use("clam")
#         s.configure("Vol.Horizontal.TProgressbar",
#                     troughcolor=C["panel"], background=C["green"],
#                     bordercolor=C["panel"], lightcolor=C["green"],
#                     darkcolor=C["green"], thickness=4)

#     # ── UI ────────────────────────────────────────────────────────────────────

#     def _build_ui(self):
#         # Header
#         hdr = tk.Frame(self, bg=C["surface"], padx=20, pady=14)
#         hdr.pack(fill="x")

#         left = tk.Frame(hdr, bg=C["surface"])
#         left.pack(side="left")
#         tk.Label(left, text="Russian STT", font=self.fn_title,
#                  bg=C["surface"], fg=C["text"]).pack(anchor="w")
#         tk.Label(left, text="Vosk  ·  real-time  ·  offline  ·  vosk-model-ru-0.42",
#                  font=self.fn_sub, bg=C["surface"], fg=C["muted"]).pack(anchor="w")

#         right = tk.Frame(hdr, bg=C["surface"])
#         right.pack(side="right")
#         tk.Label(right, text="MIC", font=self.fn_small,
#                  bg=C["surface"], fg=C["muted"]).pack(anchor="e")
#         self.vol_var = tk.DoubleVar(value=0)
#         ttk.Progressbar(right, variable=self.vol_var, maximum=100,
#                         length=120, style="Vol.Horizontal.TProgressbar"
#                         ).pack(anchor="e", pady=(3, 0))

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Status bar
#         sbar = tk.Frame(self, bg=C["panel"], padx=16, pady=7)
#         sbar.pack(fill="x")
#         self._dot = tk.Label(sbar, text="●", font=self.fn_small,
#                              bg=C["panel"], fg=C["muted"])
#         self._dot.pack(side="left", padx=(0, 7))
#         self.status_var = tk.StringVar(value="Đang tải model Vosk…")
#         tk.Label(sbar, textvariable=self.status_var, font=self.fn_small,
#                  bg=C["panel"], fg=C["muted"], anchor="w").pack(side="left")

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Toolbar — pack TRƯỚC txt_wrap
#         self._build_toolbar()

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Text area
#         txt_wrap = tk.Frame(self, bg=C["bg"], padx=16, pady=12)
#         txt_wrap.pack(fill="both", expand=True)

#         lbl_row = tk.Frame(txt_wrap, bg=C["bg"])
#         lbl_row.pack(fill="x", pady=(0, 6))
#         tk.Label(lbl_row, text="Văn bản nhận dạng",
#                  font=self.fn_small, bg=C["bg"], fg=C["muted"]).pack(side="left")
#         self._word_lbl = tk.Label(lbl_row, text="0 từ",
#                                   font=self.fn_small, bg=C["bg"], fg=C["muted"])
#         self._word_lbl.pack(side="right")

#         box = tk.Frame(txt_wrap, bg=C["surface"],
#                        highlightbackground=C["border"], highlightthickness=1)
#         box.pack(fill="both", expand=True)

#         self.text_area = tk.Text(
#             box,
#             font=self.fn_body,
#             bg=C["surface"], fg=C["text"],
#             insertbackground=C["text"],
#             selectbackground=C["accent"],
#             relief="flat", bd=0,
#             wrap=tk.WORD,
#             padx=18, pady=14,
#             spacing1=2, spacing3=8,
#             cursor="arrow",
#             state="disabled",
#         )
#         self.text_area.pack(side="left", fill="both", expand=True)

#         sb = SlimScrollbar(box, command=self.text_area.yview)
#         sb.pack(side="right", fill="y", padx=(0, 3), pady=6)
#         self.text_area.configure(yscrollcommand=sb.set)

#         self.text_area.tag_config("final",   foreground=C["text"],  font=self.fn_body)
#         self.text_area.tag_config("partial", foreground=C["partial"], font=self.fn_body)
#         self.text_area.tag_config("pause",   foreground=C["amber"],   font=self.fn_small)

#     def _build_toolbar(self):
#         bar = tk.Frame(self, bg=C["surface"], padx=16, pady=10)
#         bar.pack(fill="x")

#         self.rec_btn = self._btn(bar, "⏺  Ghi âm", C["red"],
#                                  self.toggle_recording, state="disabled")
#         self.rec_btn.pack(side="left", padx=(0, 6))

#         self.pause_btn = self._btn(bar, "⏸  Tạm dừng", C["amber"],
#                                    self.toggle_pause, state="disabled")
#         self.pause_btn.pack(side="left", padx=(0, 18))

#         tk.Frame(bar, bg=C["border"], width=1, height=22).pack(side="left", padx=(0, 18))

#         self._btn(bar, "💾  Lưu", C["panel"], self.save_session).pack(side="left", padx=(0, 6))
#         self._btn(bar, "🗑  Xóa", C["panel"], self.clear_text ).pack(side="left")

#         stat = tk.Frame(bar, bg=C["surface"])
#         stat.pack(side="right")
#         self._time_var = tk.StringVar(value="00:00")
#         tk.Label(stat, textvariable=self._time_var, font=self.fn_stat,
#                  bg=C["surface"], fg=C["accent"]).pack()
#         tk.Label(stat, text="phiên", font=self.fn_small,
#                  bg=C["surface"], fg=C["muted"]).pack()

#     def _btn(self, parent, text, bg, cmd, state="normal"):
#         b = tk.Button(parent, text=text, font=self.fn_btn,
#                       bg=bg, fg=C["text"], activebackground=bg,
#                       activeforeground=C["text"], relief="flat",
#                       padx=14, pady=6, cursor="hand2",
#                       command=cmd, state=state, bd=0, highlightthickness=0)
#         def _hl(c): return "#{:02x}{:02x}{:02x}".format(
#             *[min(255, int(c.lstrip("#")[i:i+2], 16) + 22) for i in (0, 2, 4)])
#         b.bind("<Enter>", lambda e: b.config(bg=_hl(bg)))
#         b.bind("<Leave>", lambda e: b.config(bg=bg))
#         return b

#     # ── Model load — chỉ chạy MỘT LẦN ───────────────────────────────────────

#     def _load_model_async(self):
#         self.status_var.set("Đang tải model Vosk… (chỉ lần đầu)")
#         self._dot.config(fg=C["amber"])

#         def _load():
#             try:
#                 vosk.SetLogLevel(-1)
#                 model = vosk.Model(MODEL_PATH)
#                 self._vosk_model = model
#                 self.after(0, self._on_ready)
#             except Exception as e:
#                 err = str(e)
#                 self.after(0, lambda: self._on_error(err))

#         threading.Thread(target=_load, daemon=True).start()

#     def _on_ready(self):
#         self.status_var.set("Sẵn sàng  ·  Vosk  ·  real-time")
#         self._dot.config(fg=C["green"])
#         self.rec_btn.config(state="normal")

#     def _on_error(self, err):
#         self.status_var.set(f"Lỗi tải model: {err[:60]}")
#         self._dot.config(fg=C["red"])
#         messagebox.showerror("Không tải được model",
#                              f"Kiểm tra thư mục:\n{MODEL_PATH}\n\n{err}")

#     # ── Recording ─────────────────────────────────────────────────────────────

#     def toggle_recording(self):
#         if self.is_recording: self.stop_recording()
#         else:                  self.start_recording()

#     def start_recording(self):
#         self.is_recording   = True
#         self.is_paused      = False
#         self._session_start = time.time()

#         # Xả queue cũ còn sót
#         while not self.audio_q.empty():
#             self.audio_q.get_nowait()
#         while not self.result_q.empty():
#             self.result_q.get_nowait()

#         # Worker mới — chỉ tạo KaldiRecognizer, KHÔNG load model lại
#         self.worker = VoskWorker(self._vosk_model, self.audio_q, self.result_q)
#         self.worker.start()

#         self.rec_btn.config(text="⏹  Dừng", bg=C["green"])
#         self.pause_btn.config(state="normal")
#         self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
#         self._dot.config(fg=C["red"])
#         self._tick()

#         def _cb(indata, frames, t, status):
#             if self.is_paused:
#                 return
#             pcm = (indata[:, 0] * 32767).astype("int16").tobytes()
#             self.audio_q.put_nowait(pcm)
#             rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
#             self.after(0, lambda: self.vol_var.set(min(100, rms * 2500)))

#         self.stream = sd.InputStream(
#             samplerate=SAMPLE_RATE, channels=1,
#             dtype="float32", blocksize=BLOCK_SIZE,
#             callback=_cb,
#         )
#         self.stream.start()

#     def stop_recording(self):
#         self.is_recording = False
#         self.is_paused    = False
#         if self.stream:
#             self.stream.stop(); self.stream.close(); self.stream = None
#         if self.worker:
#             self.worker.stop(); self.worker = None
#         self.vol_var.set(0)
#         self.rec_btn.config(text="⏺  Ghi âm", bg=C["red"])
#         self.pause_btn.config(state="disabled", text="⏸  Tạm dừng", bg=C["amber"])
#         self.status_var.set("Đã dừng  ·  Nhấn để ghi âm tiếp")
#         self._dot.config(fg=C["muted"])
#         self._clear_partial()

#     def toggle_pause(self):
#         self.is_paused = not self.is_paused
#         if self.is_paused:
#             self.pause_btn.config(text="▶  Tiếp tục", bg=C["accent"])
#             self.status_var.set("Tạm dừng")
#             self._dot.config(fg=C["amber"])
#             self.vol_var.set(0)
#             self._clear_partial()
#             self._insert_pause_marker()
#         else:
#             self.pause_btn.config(text="⏸  Tạm dừng", bg=C["amber"])
#             self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
#             self._dot.config(fg=C["red"])

#     # ── Timer ─────────────────────────────────────────────────────────────────

#     def _tick(self):
#         if not self.is_recording:
#             return
#         e = int(time.time() - self._session_start)
#         self._time_var.set(f"{e//60:02d}:{e%60:02d}")
#         self.after(1000, self._tick)

#     # ── Results ───────────────────────────────────────────────────────────────

#     def _poll_results(self):
#         try:
#             while True:
#                 msg = self.result_q.get_nowait()
#                 if msg["type"] == "final":
#                     self._commit_final(msg["text"])
#         except queue.Empty:
#             pass
#         self.after(30, self._poll_results)

#     def _commit_final(self, text: str):
#         self.text_area.config(state="normal")
#         self.text_area.insert("end", text + "\n", "final")
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")
#         self.session_log.append(text)
#         self.word_count += len(text.split())
#         self._word_lbl.config(text=f"{self.word_count} từ")

#     def _clear_partial(self):
#         pass  # không còn partial

#     def _insert_pause_marker(self):
#         ts = datetime.datetime.now().strftime("%H:%M:%S")
#         self.text_area.config(state="normal")
#         self.text_area.insert("end", f"\n— tạm dừng {ts} —\n", "pause")
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")

#     # ── Helpers ───────────────────────────────────────────────────────────────

#     def clear_text(self):
#         self.text_area.config(state="normal")
#         self.text_area.delete("1.0", "end")
#         self.text_area.config(state="disabled")
#         self.session_log.clear()
#         self.word_count     = 0
#         self._partial_start = None
#         self._word_lbl.config(text="0 từ")

#     def save_session(self):
#         if not self.session_log:
#             messagebox.showinfo("Thông báo", "Chưa có nội dung để lưu.")
#             return
#         default = f"session_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
#         path = filedialog.asksaveasfilename(
#             defaultextension=".txt",
#             filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
#             initialfile=default, title="Lưu phiên ghi âm",
#         )
#         if not path:
#             return
#         with open(path, "w", encoding="utf-8") as f:
#             f.write(f"# Russian STT — {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
#             f.write(f"# Từ: {self.word_count}\n\n")
#             f.write(" ".join(self.session_log))
#         messagebox.showinfo("Đã lưu", f"Lưu thành công:\n{path}")

#     def on_closing(self):
#         if self.is_recording:
#             self.stop_recording()
#         self.destroy()


# # ─── Entry point ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     app = App()
#     app.protocol("WM_DELETE_WINDOW", app.on_closing)
#     app.mainloop()

# ______________________________ Big model ______________________________________
# """
# Russian STT — Vosk real-time, từng từ hiện ngay
# Cấu trúc thư mục:
#   STT/
#   └── server/
#       ├── app.py
#       └── model/   ← vosk-model-ru-0.42
# """

# import os
# import threading
# import queue
# import json
# import tkinter as tk
# import tkinter.font as tkfont
# from tkinter import ttk, messagebox, filedialog
# import sounddevice as sd
# import vosk
# import datetime
# import time
# import numpy as np

# # ─── Cấu hình ────────────────────────────────────────────────────────────────

# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH  = os.path.join(_SCRIPT_DIR, "model")
# SAMPLE_RATE = 16000
# BLOCK_MS    = 80
# BLOCK_SIZE  = int(SAMPLE_RATE * BLOCK_MS / 1000)

# # ─── Màu sắc ─────────────────────────────────────────────────────────────────

# C = {
#     "bg":        "#0f1117",
#     "surface":   "#1a1d27",
#     "panel":     "#21253a",
#     "border":    "#2d3148",
#     "text":      "#e8eaf0",
#     "muted":     "#5a6070",
#     "accent":    "#6c8fff",
#     "green":     "#3ecf72",
#     "red":       "#ef4444",
#     "amber":     "#f59e0b",
#     "partial":   "#94a3b8",
#     "scrollbar": "#2d3148",
#     "scroll_hv": "#4a5080",
# }

# # ─── Vosk Worker ─────────────────────────────────────────────────────────────

# class VoskWorker(threading.Thread):
#     """
#     Nhận model đã load sẵn — KHÔNG load lại model mỗi lần record.
#     Chỉ tạo KaldiRecognizer mới (rất nhanh, ~ms).
#     """
#     def __init__(self, model: vosk.Model, audio_q: queue.Queue,
#                  result_q: queue.Queue):
#         super().__init__(daemon=True)
#         self.audio_q      = audio_q
#         self.result_q     = result_q
#         self.running      = True
#         self.rec          = vosk.KaldiRecognizer(model, SAMPLE_RATE)
#         self.rec.SetWords(True)
#         self._last_partial = ""   # tránh push partial trùng lặp

#     def run(self):
#         while self.running:
#             try:
#                 data = self.audio_q.get(timeout=0.1)
#             except queue.Empty:
#                 continue

#             if self.rec.AcceptWaveform(data):
#                 # Câu hoàn chỉnh — reset partial tracker
#                 res  = json.loads(self.rec.Result())
#                 text = res.get("text", "").strip()
#                 self._last_partial = ""
#                 if text:
#                     self.result_q.put({"type": "final", "text": text})
#             else:
#                 res  = json.loads(self.rec.PartialResult())
#                 text = res.get("partial", "").strip()
#                 # Chỉ push khi partial THAY ĐỔI và không rỗng
#                 if text and text != self._last_partial:
#                     self._last_partial = text
#                     self.result_q.put({"type": "partial", "text": text})

#     def stop(self):
#         self.running = False


# # ─── Custom Scrollbar ─────────────────────────────────────────────────────────

# class SlimScrollbar(tk.Canvas):
#     W = 7

#     def __init__(self, parent, command=None, **kw):
#         super().__init__(parent, width=self.W, bg=C["surface"],
#                          highlightthickness=0, **kw)
#         self._cmd    = command
#         self._lo     = 0.0
#         self._hi     = 1.0
#         self._drag_y = None
#         self.bind("<Configure>",       lambda e: self._draw())
#         self.bind("<ButtonPress-1>",   self._press)
#         self.bind("<B1-Motion>",       self._drag)
#         self.bind("<ButtonRelease-1>", self._release)
#         self.bind("<Enter>",           self._hover_on)
#         self.bind("<Leave>",           self._hover_off)

#     def set(self, lo, hi):
#         self._lo, self._hi = float(lo), float(hi)
#         self._draw()

#     def _draw(self, hover=False):
#         self.delete("all")
#         h = self.winfo_height()
#         w = self.winfo_width()
#         if h < 2 or (self._hi - self._lo) >= 1.0:
#             return
#         ty  = max(2,   int(self._lo * h))
#         by  = min(h-2, max(ty + 20, int(self._hi * h)))
#         col = C["scroll_hv"] if hover else C["scrollbar"]
#         pts = [2+3,ty, w-2-3,ty, w-2,ty, w-2,ty+3,
#                w-2,by-3, w-2,by, w-2-3,by, 2+3,by,
#                2,by, 2,by-3, 2,ty+3, 2,ty, 2+3,ty]
#         self.create_polygon(pts, smooth=True, fill=col)

#     def _press(self, e):   self._drag_y = e.y
#     def _release(self, e): self._drag_y = None
#     def _hover_on(self, e):  self._draw(hover=True)
#     def _hover_off(self, e): self._draw(hover=False)

#     def _drag(self, e):
#         if self._drag_y is None or not self._cmd:
#             return
#         h    = self.winfo_height()
#         diff = (e.y - self._drag_y) / max(h, 1)
#         self._drag_y = e.y
#         span = self._hi - self._lo
#         new  = max(0.0, min(1.0 - span, self._lo + diff))
#         self._cmd("moveto", new)


# # ─── App ─────────────────────────────────────────────────────────────────────

# class App(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("Russian STT")
#         self.geometry("860x660")
#         self.minsize(640, 480)
#         self.configure(bg=C["bg"])

#         # Model load một lần, dùng lại mãi
#         self._vosk_model    = None

#         self.worker         = None
#         self.stream         = None
#         # audio_q và result_q tạo một lần, dùng chung
#         self.audio_q        = queue.Queue()
#         self.result_q       = queue.Queue()
#         self.is_recording   = False
#         self.is_paused      = False
#         self.session_log    = []
#         self.word_count     = 0
#         self._session_start = 0.0
#         self._partial_start = None

#         self._setup_fonts()
#         self._setup_styles()
#         self._build_ui()
#         self._load_model_async()
#         self.after(30, self._poll_results)

#     # ── Fonts & styles ────────────────────────────────────────────────────────

#     def _setup_fonts(self):
#         self.fn_title = tkfont.Font(family="Segoe UI", size=14, weight="bold")
#         self.fn_sub   = tkfont.Font(family="Segoe UI", size=9)
#         self.fn_body  = tkfont.Font(family="Segoe UI", size=15)
#         self.fn_small = tkfont.Font(family="Segoe UI", size=9)
#         self.fn_btn   = tkfont.Font(family="Segoe UI", size=10, weight="bold")
#         self.fn_stat  = tkfont.Font(family="Segoe UI", size=13, weight="bold")

#     def _setup_styles(self):
#         s = ttk.Style()
#         s.theme_use("clam")
#         s.configure("Vol.Horizontal.TProgressbar",
#                     troughcolor=C["panel"], background=C["green"],
#                     bordercolor=C["panel"], lightcolor=C["green"],
#                     darkcolor=C["green"], thickness=4)

#     # ── UI ────────────────────────────────────────────────────────────────────

#     def _build_ui(self):
#         # Header
#         hdr = tk.Frame(self, bg=C["surface"], padx=20, pady=14)
#         hdr.pack(fill="x")

#         left = tk.Frame(hdr, bg=C["surface"])
#         left.pack(side="left")
#         tk.Label(left, text="Russian STT", font=self.fn_title,
#                  bg=C["surface"], fg=C["text"]).pack(anchor="w")
#         tk.Label(left, text="Vosk  ·  real-time  ·  offline  ·  vosk-model-ru-0.42",
#                  font=self.fn_sub, bg=C["surface"], fg=C["muted"]).pack(anchor="w")

#         right = tk.Frame(hdr, bg=C["surface"])
#         right.pack(side="right")
#         tk.Label(right, text="MIC", font=self.fn_small,
#                  bg=C["surface"], fg=C["muted"]).pack(anchor="e")
#         self.vol_var = tk.DoubleVar(value=0)
#         ttk.Progressbar(right, variable=self.vol_var, maximum=100,
#                         length=120, style="Vol.Horizontal.TProgressbar"
#                         ).pack(anchor="e", pady=(3, 0))

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Status bar
#         sbar = tk.Frame(self, bg=C["panel"], padx=16, pady=7)
#         sbar.pack(fill="x")
#         self._dot = tk.Label(sbar, text="●", font=self.fn_small,
#                              bg=C["panel"], fg=C["muted"])
#         self._dot.pack(side="left", padx=(0, 7))
#         self.status_var = tk.StringVar(value="Đang tải model Vosk…")
#         tk.Label(sbar, textvariable=self.status_var, font=self.fn_small,
#                  bg=C["panel"], fg=C["muted"], anchor="w").pack(side="left")

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Toolbar — pack TRƯỚC txt_wrap
#         self._build_toolbar()

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Text area
#         txt_wrap = tk.Frame(self, bg=C["bg"], padx=16, pady=12)
#         txt_wrap.pack(fill="both", expand=True)

#         lbl_row = tk.Frame(txt_wrap, bg=C["bg"])
#         lbl_row.pack(fill="x", pady=(0, 6))
#         tk.Label(lbl_row, text="Văn bản nhận dạng",
#                  font=self.fn_small, bg=C["bg"], fg=C["muted"]).pack(side="left")
#         self._word_lbl = tk.Label(lbl_row, text="0 từ",
#                                   font=self.fn_small, bg=C["bg"], fg=C["muted"])
#         self._word_lbl.pack(side="right")

#         box = tk.Frame(txt_wrap, bg=C["surface"],
#                        highlightbackground=C["border"], highlightthickness=1)
#         box.pack(fill="both", expand=True)

#         self.text_area = tk.Text(
#             box,
#             font=self.fn_body,
#             bg=C["surface"], fg=C["text"],
#             insertbackground=C["text"],
#             selectbackground=C["accent"],
#             relief="flat", bd=0,
#             wrap=tk.WORD,
#             padx=18, pady=14,
#             spacing1=2, spacing3=8,
#             cursor="arrow",
#             state="disabled",
#         )
#         self.text_area.pack(side="left", fill="both", expand=True)

#         sb = SlimScrollbar(box, command=self.text_area.yview)
#         sb.pack(side="right", fill="y", padx=(0, 3), pady=6)
#         self.text_area.configure(yscrollcommand=sb.set)

#         self.text_area.tag_config("final",   foreground=C["text"],  font=self.fn_body)
#         self.text_area.tag_config("partial", foreground=C["partial"], font=self.fn_body)
#         self.text_area.tag_config("pause",   foreground=C["amber"],   font=self.fn_small)

#     def _build_toolbar(self):
#         bar = tk.Frame(self, bg=C["surface"], padx=16, pady=10)
#         bar.pack(fill="x")

#         self.rec_btn = self._btn(bar, "⏺  Ghi âm", C["red"],
#                                  self.toggle_recording, state="disabled")
#         self.rec_btn.pack(side="left", padx=(0, 6))

#         self.pause_btn = self._btn(bar, "⏸  Tạm dừng", C["amber"],
#                                    self.toggle_pause, state="disabled")
#         self.pause_btn.pack(side="left", padx=(0, 18))

#         tk.Frame(bar, bg=C["border"], width=1, height=22).pack(side="left", padx=(0, 18))

#         self._btn(bar, "💾  Lưu", C["panel"], self.save_session).pack(side="left", padx=(0, 6))
#         self._btn(bar, "🗑  Xóa", C["panel"], self.clear_text ).pack(side="left")

#         stat = tk.Frame(bar, bg=C["surface"])
#         stat.pack(side="right")
#         self._time_var = tk.StringVar(value="00:00")
#         tk.Label(stat, textvariable=self._time_var, font=self.fn_stat,
#                  bg=C["surface"], fg=C["accent"]).pack()
#         tk.Label(stat, text="phiên", font=self.fn_small,
#                  bg=C["surface"], fg=C["muted"]).pack()

#     def _btn(self, parent, text, bg, cmd, state="normal"):
#         b = tk.Button(parent, text=text, font=self.fn_btn,
#                       bg=bg, fg=C["text"], activebackground=bg,
#                       activeforeground=C["text"], relief="flat",
#                       padx=14, pady=6, cursor="hand2",
#                       command=cmd, state=state, bd=0, highlightthickness=0)
#         def _hl(c): return "#{:02x}{:02x}{:02x}".format(
#             *[min(255, int(c.lstrip("#")[i:i+2], 16) + 22) for i in (0, 2, 4)])
#         b.bind("<Enter>", lambda e: b.config(bg=_hl(bg)))
#         b.bind("<Leave>", lambda e: b.config(bg=bg))
#         return b

#     # ── Model load — chỉ chạy MỘT LẦN ───────────────────────────────────────

#     def _load_model_async(self):
#         self.status_var.set("Đang tải model Vosk… (chỉ lần đầu)")
#         self._dot.config(fg=C["amber"])

#         def _load():
#             try:
#                 vosk.SetLogLevel(-1)
#                 model = vosk.Model(MODEL_PATH)
#                 self._vosk_model = model
#                 self.after(0, self._on_ready)
#             except Exception as e:
#                 err = str(e)
#                 self.after(0, lambda: self._on_error(err))

#         threading.Thread(target=_load, daemon=True).start()

#     def _warmup(self, model):
#         """Chạy silent audio qua Vosk để JIT compile xong trước khi dùng."""
#         try:
#             rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
#             silence = (np.zeros(SAMPLE_RATE * 2, dtype=np.int16)).tobytes()  # 2s silence
#             rec.AcceptWaveform(silence)
#             rec.FinalResult()
#         except Exception:
#             pass

#     def _on_ready(self):
#         # Warmup trong background — xong mới enable nút
#         def _do_warmup():
#             self._warmup(self._vosk_model)
#             self.after(0, self._on_warmup_done)
#         self.status_var.set("Khởi động model… (chỉ lần đầu)")
#         threading.Thread(target=_do_warmup, daemon=True).start()

#     def _on_warmup_done(self):
#         self.status_var.set("Sẵn sàng  ·  Vosk ru-0.42  ·  real-time")
#         self._dot.config(fg=C["green"])
#         self.rec_btn.config(state="normal")

#     def _on_error(self, err):
#         self.status_var.set(f"Lỗi tải model: {err[:60]}")
#         self._dot.config(fg=C["red"])
#         messagebox.showerror("Không tải được model",
#                              f"Kiểm tra thư mục:\n{MODEL_PATH}\n\n{err}")

#     # ── Recording ─────────────────────────────────────────────────────────────

#     def toggle_recording(self):
#         if self.is_recording: self.stop_recording()
#         else:                  self.start_recording()

#     def start_recording(self):
#         self.is_recording   = True
#         self.is_paused      = False
#         self._session_start = time.time()

#         # Xả queue cũ còn sót
#         while not self.audio_q.empty():
#             self.audio_q.get_nowait()
#         while not self.result_q.empty():
#             self.result_q.get_nowait()

#         # Worker mới — chỉ tạo KaldiRecognizer, KHÔNG load model lại
#         self.worker = VoskWorker(self._vosk_model, self.audio_q, self.result_q)
#         self.worker.start()

#         self.rec_btn.config(text="⏹  Dừng", bg=C["green"])
#         self.pause_btn.config(state="normal")
#         self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
#         self._dot.config(fg=C["red"])
#         self._tick()

#         def _cb(indata, frames, t, status):
#             if self.is_paused:
#                 return
#             pcm = (indata[:, 0] * 32767).astype("int16").tobytes()
#             self.audio_q.put_nowait(pcm)
#             rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
#             self.after(0, lambda: self.vol_var.set(min(100, rms * 2500)))

#         self.stream = sd.InputStream(
#             samplerate=SAMPLE_RATE, channels=1,
#             dtype="float32", blocksize=BLOCK_SIZE,
#             callback=_cb,
#         )
#         self.stream.start()

#     def stop_recording(self):
#         self.is_recording = False
#         self.is_paused    = False
#         if self.stream:
#             self.stream.stop(); self.stream.close(); self.stream = None
#         if self.worker:
#             self.worker.stop(); self.worker = None
#         self.vol_var.set(0)
#         self.rec_btn.config(text="⏺  Ghi âm", bg=C["red"])
#         self.pause_btn.config(state="disabled", text="⏸  Tạm dừng", bg=C["amber"])
#         self.status_var.set("Đã dừng  ·  Nhấn để ghi âm tiếp")
#         self._dot.config(fg=C["muted"])
#         self._clear_partial()

#     def toggle_pause(self):
#         self.is_paused = not self.is_paused
#         if self.is_paused:
#             self.pause_btn.config(text="▶  Tiếp tục", bg=C["accent"])
#             self.status_var.set("Tạm dừng")
#             self._dot.config(fg=C["amber"])
#             self.vol_var.set(0)
#             self._clear_partial()
#             self._insert_pause_marker()
#         else:
#             self.pause_btn.config(text="⏸  Tạm dừng", bg=C["amber"])
#             self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
#             self._dot.config(fg=C["red"])

#     # ── Timer ─────────────────────────────────────────────────────────────────

#     def _tick(self):
#         if not self.is_recording:
#             return
#         e = int(time.time() - self._session_start)
#         self._time_var.set(f"{e//60:02d}:{e%60:02d}")
#         self.after(1000, self._tick)

#     # ── Results ───────────────────────────────────────────────────────────────

#     def _poll_results(self):
#         try:
#             while True:
#                 msg = self.result_q.get_nowait()
#                 if msg["type"] == "partial":
#                     self._update_partial(msg["text"])
#                 else:
#                     self._commit_final(msg["text"])
#         except queue.Empty:
#             pass
#         self.after(30, self._poll_results)

#     def _update_partial(self, text: str):
#         self.text_area.config(state="normal")
#         if self._partial_start:
#             self.text_area.delete(self._partial_start, "end")
#         if text:
#             self._partial_start = self.text_area.index("end-1c")
#             self.text_area.insert("end", text, "partial")
#         else:
#             self._partial_start = None
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")

#     def _commit_final(self, text: str):
#         self.text_area.config(state="normal")
#         if self._partial_start:
#             self.text_area.delete(self._partial_start, "end")
#             self._partial_start = None
#         self.text_area.insert("end", text + " ", "final")
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")
#         self.session_log.append(text)
#         self.word_count += len(text.split())
#         self._word_lbl.config(text=f"{self.word_count} từ")

#     def _clear_partial(self):
#         if self._partial_start:
#             self.text_area.config(state="normal")
#             self.text_area.delete(self._partial_start, "end")
#             self.text_area.config(state="disabled")
#             self._partial_start = None

#     def _insert_pause_marker(self):
#         ts = datetime.datetime.now().strftime("%H:%M:%S")
#         self.text_area.config(state="normal")
#         self.text_area.insert("end", f"\n— tạm dừng {ts} — \n", "pause")
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")

#     # ── Helpers ───────────────────────────────────────────────────────────────

#     def clear_text(self):
#         self.text_area.config(state="normal")
#         self.text_area.delete("1.0", "end")
#         self.text_area.config(state="disabled")
#         self.session_log.clear()
#         self.word_count     = 0
#         self._partial_start = None
#         self._word_lbl.config(text="0 từ")

#     def save_session(self):
#         if not self.session_log:
#             messagebox.showinfo("Thông báo", "Chưa có nội dung để lưu.")
#             return
#         default = f"session_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
#         path = filedialog.asksaveasfilename(
#             defaultextension=".txt",
#             filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
#             initialfile=default, title="Lưu phiên ghi âm",
#         )
#         if not path:
#             return
#         with open(path, "w", encoding="utf-8") as f:
#             f.write(f"# Russian STT — {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
#             f.write(f"# Từ: {self.word_count}\n\n")
#             f.write(" ".join(self.session_log))
#         messagebox.showinfo("Đã lưu", f"Lưu thành công:\n{path}")

#     def on_closing(self):
#         if self.is_recording:
#             self.stop_recording()
#         self.destroy()


# # ─── Entry point ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     app = App()
#     app.protocol("WM_DELETE_WINDOW", app.on_closing)
#     app.mainloop()


# _______________________________ OK ____________________________________________
# """
# Russian STT — Vosk real-time, từng từ hiện ngay
# Cấu trúc thư mục:
#   STT/
#   └── server/
#       ├── app.py
#       └── model/   ← vosk-model-ru-0.42
# """

# import os
# import threading
# import queue
# import json
# import tkinter as tk
# import tkinter.font as tkfont
# from tkinter import ttk, messagebox, filedialog
# import sounddevice as sd
# import vosk
# import datetime
# import time
# import numpy as np

# # ─── Cấu hình ────────────────────────────────────────────────────────────────

# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH  = os.path.join(_SCRIPT_DIR, "model")
# SAMPLE_RATE = 16000
# # BLOCK_MS    = 20
# # BLOCK_SIZE  = int(SAMPLE_RATE * BLOCK_MS / 1000)
# BLOCK_MS  = 80
# BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)   # 1280 samples

# # ─── Màu sắc ─────────────────────────────────────────────────────────────────

# C = {
#     "bg":        "#0f1117",
#     "surface":   "#1a1d27",
#     "panel":     "#21253a",
#     "border":    "#2d3148",
#     "text":      "#e8eaf0",
#     "muted":     "#5a6070",
#     "accent":    "#6c8fff",
#     "green":     "#3ecf72",
#     "red":       "#ef4444",
#     "amber":     "#f59e0b",
#     "partial":   "#94a3b8",
#     "scrollbar": "#2d3148",
#     "scroll_hv": "#4a5080",
# }

# # ─── Vosk Worker ─────────────────────────────────────────────────────────────

# class VoskWorker(threading.Thread):
#     """
#     Nhận model đã load sẵn — KHÔNG load lại model mỗi lần record.
#     Chỉ tạo KaldiRecognizer mới (rất nhanh, ~ms).
#     """
#     def __init__(self, model: vosk.Model, audio_q: queue.Queue,
#                  result_q: queue.Queue):
#         super().__init__(daemon=True)
#         self.audio_q  = audio_q
#         self.result_q = result_q
#         self.running  = True
#         self.rec      = vosk.KaldiRecognizer(model, SAMPLE_RATE)
#         self.rec.SetWords(True)

#     def run(self):
#         while self.running:
#             try:
#                 data = self.audio_q.get(timeout=0.1)
#             except queue.Empty:
#                 continue

#             if self.rec.AcceptWaveform(data):
#                 res  = json.loads(self.rec.Result())
#                 text = res.get("text", "").strip()
#                 if text:
#                     self.result_q.put({"type": "final", "text": text})
#             else:
#                 res  = json.loads(self.rec.PartialResult())
#                 text = res.get("partial", "").strip()
#                 self.result_q.put({"type": "partial", "text": text})

#     def stop(self):
#         self.running = False


# # ─── Custom Scrollbar ─────────────────────────────────────────────────────────

# class SlimScrollbar(tk.Canvas):
#     W = 7

#     def __init__(self, parent, command=None, **kw):
#         super().__init__(parent, width=self.W, bg=C["surface"],
#                          highlightthickness=0, **kw)
#         self._cmd    = command
#         self._lo     = 0.0
#         self._hi     = 1.0
#         self._drag_y = None
#         self.bind("<Configure>",       lambda e: self._draw())
#         self.bind("<ButtonPress-1>",   self._press)
#         self.bind("<B1-Motion>",       self._drag)
#         self.bind("<ButtonRelease-1>", self._release)
#         self.bind("<Enter>",           self._hover_on)
#         self.bind("<Leave>",           self._hover_off)

#     def set(self, lo, hi):
#         self._lo, self._hi = float(lo), float(hi)
#         self._draw()

#     def _draw(self, hover=False):
#         self.delete("all")
#         h = self.winfo_height()
#         w = self.winfo_width()
#         if h < 2 or (self._hi - self._lo) >= 1.0:
#             return
#         ty  = max(2,   int(self._lo * h))
#         by  = min(h-2, max(ty + 20, int(self._hi * h)))
#         col = C["scroll_hv"] if hover else C["scrollbar"]
#         pts = [2+3,ty, w-2-3,ty, w-2,ty, w-2,ty+3,
#                w-2,by-3, w-2,by, w-2-3,by, 2+3,by,
#                2,by, 2,by-3, 2,ty+3, 2,ty, 2+3,ty]
#         self.create_polygon(pts, smooth=True, fill=col)

#     def _press(self, e):   self._drag_y = e.y
#     def _release(self, e): self._drag_y = None
#     def _hover_on(self, e):  self._draw(hover=True)
#     def _hover_off(self, e): self._draw(hover=False)

#     def _drag(self, e):
#         if self._drag_y is None or not self._cmd:
#             return
#         h    = self.winfo_height()
#         diff = (e.y - self._drag_y) / max(h, 1)
#         self._drag_y = e.y
#         span = self._hi - self._lo
#         new  = max(0.0, min(1.0 - span, self._lo + diff))
#         self._cmd("moveto", new)


# # ─── App ─────────────────────────────────────────────────────────────────────

# class App(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("Russian STT")
#         self.geometry("860x660")
#         self.minsize(640, 480)
#         self.configure(bg=C["bg"])

#         # Model load một lần, dùng lại mãi
#         self._vosk_model    = None

#         self.worker         = None
#         self.stream         = None
#         # audio_q và result_q tạo một lần, dùng chung
#         self.audio_q        = queue.Queue()
#         self.result_q       = queue.Queue()
#         self.is_recording   = False
#         self.is_paused      = False
#         self.session_log    = []
#         self.word_count     = 0
#         self._session_start = 0.0
#         self._partial_start = None

#         self._setup_fonts()
#         self._setup_styles()
#         self._build_ui()
#         self._load_model_async()
#         self.after(30, self._poll_results)

#     # ── Fonts & styles ────────────────────────────────────────────────────────

#     def _setup_fonts(self):
#         self.fn_title = tkfont.Font(family="Segoe UI", size=14, weight="bold")
#         self.fn_sub   = tkfont.Font(family="Segoe UI", size=9)
#         self.fn_body  = tkfont.Font(family="Segoe UI", size=15)
#         self.fn_small = tkfont.Font(family="Segoe UI", size=9)
#         self.fn_btn   = tkfont.Font(family="Segoe UI", size=10, weight="bold")
#         self.fn_stat  = tkfont.Font(family="Segoe UI", size=13, weight="bold")

#     def _setup_styles(self):
#         s = ttk.Style()
#         s.theme_use("clam")
#         s.configure("Vol.Horizontal.TProgressbar",
#                     troughcolor=C["panel"], background=C["green"],
#                     bordercolor=C["panel"], lightcolor=C["green"],
#                     darkcolor=C["green"], thickness=4)

#     # ── UI ────────────────────────────────────────────────────────────────────

#     def _build_ui(self):
#         # Header
#         hdr = tk.Frame(self, bg=C["surface"], padx=20, pady=14)
#         hdr.pack(fill="x")

#         left = tk.Frame(hdr, bg=C["surface"])
#         left.pack(side="left")
#         tk.Label(left, text="Russian STT", font=self.fn_title,
#                  bg=C["surface"], fg=C["text"]).pack(anchor="w")
#         tk.Label(left, text="Vosk  ·  real-time  ·  offline  ·  vosk-model-ru-0.42",
#                  font=self.fn_sub, bg=C["surface"], fg=C["muted"]).pack(anchor="w")

#         right = tk.Frame(hdr, bg=C["surface"])
#         right.pack(side="right")
#         tk.Label(right, text="MIC", font=self.fn_small,
#                  bg=C["surface"], fg=C["muted"]).pack(anchor="e")
#         self.vol_var = tk.DoubleVar(value=0)
#         ttk.Progressbar(right, variable=self.vol_var, maximum=100,
#                         length=120, style="Vol.Horizontal.TProgressbar"
#                         ).pack(anchor="e", pady=(3, 0))

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Status bar
#         sbar = tk.Frame(self, bg=C["panel"], padx=16, pady=7)
#         sbar.pack(fill="x")
#         self._dot = tk.Label(sbar, text="●", font=self.fn_small,
#                              bg=C["panel"], fg=C["muted"])
#         self._dot.pack(side="left", padx=(0, 7))
#         self.status_var = tk.StringVar(value="Đang tải model Vosk…")
#         tk.Label(sbar, textvariable=self.status_var, font=self.fn_small,
#                  bg=C["panel"], fg=C["muted"], anchor="w").pack(side="left")

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Toolbar — pack TRƯỚC txt_wrap
#         self._build_toolbar()

#         tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

#         # Text area
#         txt_wrap = tk.Frame(self, bg=C["bg"], padx=16, pady=12)
#         txt_wrap.pack(fill="both", expand=True)

#         lbl_row = tk.Frame(txt_wrap, bg=C["bg"])
#         lbl_row.pack(fill="x", pady=(0, 6))
#         tk.Label(lbl_row, text="Văn bản nhận dạng",
#                  font=self.fn_small, bg=C["bg"], fg=C["muted"]).pack(side="left")
#         self._word_lbl = tk.Label(lbl_row, text="0 từ",
#                                   font=self.fn_small, bg=C["bg"], fg=C["muted"])
#         self._word_lbl.pack(side="right")

#         box = tk.Frame(txt_wrap, bg=C["surface"],
#                        highlightbackground=C["border"], highlightthickness=1)
#         box.pack(fill="both", expand=True)

#         self.text_area = tk.Text(
#             box,
#             font=self.fn_body,
#             bg=C["surface"], fg=C["text"],
#             insertbackground=C["text"],
#             selectbackground=C["accent"],
#             relief="flat", bd=0,
#             wrap=tk.WORD,
#             padx=18, pady=14,
#             spacing1=2, spacing3=8,
#             cursor="arrow",
#             state="disabled",
#         )
#         self.text_area.pack(side="left", fill="both", expand=True)

#         sb = SlimScrollbar(box, command=self.text_area.yview)
#         sb.pack(side="right", fill="y", padx=(0, 3), pady=6)
#         self.text_area.configure(yscrollcommand=sb.set)

#         self.text_area.tag_config("final",   foreground=C["text"],  font=self.fn_body)
#         self.text_area.tag_config("partial", foreground=C["partial"], font=self.fn_body)
#         self.text_area.tag_config("pause",   foreground=C["amber"],   font=self.fn_small)

#     def _build_toolbar(self):
#         bar = tk.Frame(self, bg=C["surface"], padx=16, pady=10)
#         bar.pack(fill="x")

#         self.rec_btn = self._btn(bar, "⏺  Ghi âm", C["red"],
#                                  self.toggle_recording, state="disabled")
#         self.rec_btn.pack(side="left", padx=(0, 6))

#         self.pause_btn = self._btn(bar, "⏸  Tạm dừng", C["amber"],
#                                    self.toggle_pause, state="disabled")
#         self.pause_btn.pack(side="left", padx=(0, 18))

#         tk.Frame(bar, bg=C["border"], width=1, height=22).pack(side="left", padx=(0, 18))

#         self._btn(bar, "💾  Lưu", C["panel"], self.save_session).pack(side="left", padx=(0, 6))
#         self._btn(bar, "🗑  Xóa", C["panel"], self.clear_text ).pack(side="left")

#         stat = tk.Frame(bar, bg=C["surface"])
#         stat.pack(side="right")
#         self._time_var = tk.StringVar(value="00:00")
#         tk.Label(stat, textvariable=self._time_var, font=self.fn_stat,
#                  bg=C["surface"], fg=C["accent"]).pack()
#         tk.Label(stat, text="phiên", font=self.fn_small,
#                  bg=C["surface"], fg=C["muted"]).pack()

#     def _btn(self, parent, text, bg, cmd, state="normal"):
#         b = tk.Button(parent, text=text, font=self.fn_btn,
#                       bg=bg, fg=C["text"], activebackground=bg,
#                       activeforeground=C["text"], relief="flat",
#                       padx=14, pady=6, cursor="hand2",
#                       command=cmd, state=state, bd=0, highlightthickness=0)
#         def _hl(c): return "#{:02x}{:02x}{:02x}".format(
#             *[min(255, int(c.lstrip("#")[i:i+2], 16) + 22) for i in (0, 2, 4)])
#         b.bind("<Enter>", lambda e: b.config(bg=_hl(bg)))
#         b.bind("<Leave>", lambda e: b.config(bg=bg))
#         return b

#     # ── Model load — chỉ chạy MỘT LẦN ───────────────────────────────────────

#     def _load_model_async(self):
#         self.status_var.set("Đang tải model Vosk… (chỉ lần đầu)")
#         self._dot.config(fg=C["amber"])

#         def _load():
#             try:
#                 vosk.SetLogLevel(-1)
#                 model = vosk.Model(MODEL_PATH)
#                 self._vosk_model = model
#                 self.after(0, self._on_ready)
#             except Exception as e:
#                 err = str(e)
#                 self.after(0, lambda: self._on_error(err))

#         threading.Thread(target=_load, daemon=True).start()

#     def _on_ready(self):
#         self.status_var.set("Sẵn sàng  ·  Vosk ru-0.42  ·  real-time")
#         self._dot.config(fg=C["green"])
#         self.rec_btn.config(state="normal")

#     def _on_error(self, err):
#         self.status_var.set(f"Lỗi tải model: {err[:60]}")
#         self._dot.config(fg=C["red"])
#         messagebox.showerror("Không tải được model",
#                              f"Kiểm tra thư mục:\n{MODEL_PATH}\n\n{err}")

#     # ── Recording ─────────────────────────────────────────────────────────────

#     def toggle_recording(self):
#         if self.is_recording: self.stop_recording()
#         else:                  self.start_recording()

#     def start_recording(self):
#         self.is_recording   = True
#         self.is_paused      = False
#         self._session_start = time.time()

#         # Xả queue cũ còn sót
#         while not self.audio_q.empty():
#             self.audio_q.get_nowait()
#         while not self.result_q.empty():
#             self.result_q.get_nowait()

#         # Worker mới — chỉ tạo KaldiRecognizer, KHÔNG load model lại
#         self.worker = VoskWorker(self._vosk_model, self.audio_q, self.result_q)
#         self.worker.start()

#         self.rec_btn.config(text="⏹  Dừng", bg=C["green"])
#         self.pause_btn.config(state="normal")
#         self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
#         self._dot.config(fg=C["red"])
#         self._tick()

#         def _cb(indata, frames, t, status):
#             if self.is_paused:
#                 return
#             pcm = (indata[:, 0] * 32767).astype("int16").tobytes()
#             self.audio_q.put_nowait(pcm)
#             rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
#             self.after(0, lambda: self.vol_var.set(min(100, rms * 2500)))

#         self.stream = sd.InputStream(
#             samplerate=SAMPLE_RATE, channels=1,
#             dtype="float32", blocksize=BLOCK_SIZE,
#             callback=_cb,
#         )
#         self.stream.start()

#     def stop_recording(self):
#         self.is_recording = False
#         self.is_paused    = False
#         if self.stream:
#             self.stream.stop(); self.stream.close(); self.stream = None
#         if self.worker:
#             self.worker.stop(); self.worker = None
#         self.vol_var.set(0)
#         self.rec_btn.config(text="⏺  Ghi âm", bg=C["red"])
#         self.pause_btn.config(state="disabled", text="⏸  Tạm dừng", bg=C["amber"])
#         self.status_var.set("Đã dừng  ·  Nhấn để ghi âm tiếp")
#         self._dot.config(fg=C["muted"])
#         self._clear_partial()

#     def toggle_pause(self):
#         self.is_paused = not self.is_paused
#         if self.is_paused:
#             self.pause_btn.config(text="▶  Tiếp tục", bg=C["accent"])
#             self.status_var.set("Tạm dừng")
#             self._dot.config(fg=C["amber"])
#             self.vol_var.set(0)
#             self._clear_partial()
#             self._insert_pause_marker()
#         else:
#             self.pause_btn.config(text="⏸  Tạm dừng", bg=C["amber"])
#             self.status_var.set("Đang ghi âm…  hãy nói tiếng Nga")
#             self._dot.config(fg=C["red"])

#     # ── Timer ─────────────────────────────────────────────────────────────────

#     def _tick(self):
#         if not self.is_recording:
#             return
#         e = int(time.time() - self._session_start)
#         self._time_var.set(f"{e//60:02d}:{e%60:02d}")
#         self.after(1000, self._tick)

#     # ── Results ───────────────────────────────────────────────────────────────

#     def _poll_results(self):
#         try:
#             while True:
#                 msg = self.result_q.get_nowait()
#                 if msg["type"] == "partial":
#                     self._update_partial(msg["text"])
#                 else:
#                     self._commit_final(msg["text"])
#         except queue.Empty:
#             pass
#         self.after(30, self._poll_results)

#     def _update_partial(self, text: str):
#         self.text_area.config(state="normal")
#         if self._partial_start:
#             self.text_area.delete(self._partial_start, "end")
#         if text:
#             self._partial_start = self.text_area.index("end-1c")
#             self.text_area.insert("end", text, "partial")
#         else:
#             self._partial_start = None
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")

#     def _commit_final(self, text: str):
#         self.text_area.config(state="normal")
#         if self._partial_start:
#             self.text_area.delete(self._partial_start, "end")
#             self._partial_start = None
#         self.text_area.insert("end", text + " ", "final")
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")
#         self.session_log.append(text)
#         self.word_count += len(text.split())
#         self._word_lbl.config(text=f"{self.word_count} từ")

#     def _clear_partial(self):
#         if self._partial_start:
#             self.text_area.config(state="normal")
#             self.text_area.delete(self._partial_start, "end")
#             self.text_area.config(state="disabled")
#             self._partial_start = None

#     def _insert_pause_marker(self):
#         ts = datetime.datetime.now().strftime("%H:%M:%S")
#         self.text_area.config(state="normal")
#         self.text_area.insert("end", f"\n— tạm dừng {ts} —\n", "pause")
#         self.text_area.see("end")
#         self.text_area.config(state="disabled")

#     # ── Helpers ───────────────────────────────────────────────────────────────

#     def clear_text(self):
#         self.text_area.config(state="normal")
#         self.text_area.delete("1.0", "end")
#         self.text_area.config(state="disabled")
#         self.session_log.clear()
#         self.word_count     = 0
#         self._partial_start = None
#         self._word_lbl.config(text="0 từ")

#     def save_session(self):
#         if not self.session_log:
#             messagebox.showinfo("Thông báo", "Chưa có nội dung để lưu.")
#             return
#         default = f"session_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
#         path = filedialog.asksaveasfilename(
#             defaultextension=".txt",
#             filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
#             initialfile=default, title="Lưu phiên ghi âm",
#         )
#         if not path:
#             return
#         with open(path, "w", encoding="utf-8") as f:
#             f.write(f"# Russian STT — {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
#             f.write(f"# Từ: {self.word_count}\n\n")
#             f.write(" ".join(self.session_log))
#         messagebox.showinfo("Đã lưu", f"Lưu thành công:\n{path}")

#     def on_closing(self):
#         if self.is_recording:
#             self.stop_recording()
#         self.destroy()


# # ─── Entry point ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     app = App()
#     app.protocol("WM_DELETE_WINDOW", app.on_closing)
#     app.mainloop()
