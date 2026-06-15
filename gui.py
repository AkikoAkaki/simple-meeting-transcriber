#!/usr/bin/env python3
"""
meeting-transcriber — gui.py
Minimal GUI. Follows system dark/light theme. UI language: EN / 中文.

Optional: pip install tkinterdnd2   (enables drag-and-drop)
Usage:    python gui.py
"""

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import config

TRANSCRIBE_SCRIPT = Path(__file__).parent / "transcribe.py"
W = 560   # fixed window width

# ── Drag-and-drop (optional) ──────────────────────────────────────────────────
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND = True
except ImportError:
    _DND = False

# ── System theme detection ────────────────────────────────────────────────────
def _system_is_dark() -> bool:
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            val, _ = winreg.QueryValueEx(key, "AppsUseDarkTheme")
            return bool(val)
        except Exception:
            pass
    return True

# ── Color themes ──────────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg":      "#1c1c1e",
        "bg2":     "#2c2c2e",
        "bg3":     "#3a3a3c",
        "fg":      "#f5f5f7",
        "fg_dim":  "#8e8e93",
        "accent":  "#0a84ff",
        "success": "#30d158",
        "danger":  "#ff453a",
    },
    "light": {
        "bg":      "#f2f2f7",
        "bg2":     "#ffffff",
        "bg3":     "#e5e5ea",
        "fg":      "#1c1c1e",
        "fg_dim":  "#6c6c70",
        "accent":  "#007aff",
        "success": "#34c759",
        "danger":  "#ff3b30",
    },
}

# ── i18n ──────────────────────────────────────────────────────────────────────
I18N = {
    "en": {
        "title":        "meeting-transcriber",
        "subtitle":     "Transcribe meetings with speaker labels",
        "drop_hint":    "Drop video here  ·  or click to browse",
        "no_file":      "No file selected",
        "xlang_label":  "Transcription language",
        "more":         "⚙  More settings  ▸",
        "less":         "⚙  More settings  ▾",
        "diarize":      "Speaker diarization",
        "model_lbl":    "Model",
        "device_lbl":   "Device",
        "speaker_lbl":  "Speakers",
        "transcribe":   "Transcribe",
        "running":      "Running…",
        "log_show":     "▸  Log",
        "log_hide":     "▾  Log",
        "open_btn":     "Open transcript →",
        "err_no_file":  "Please select a file first.",
        "err_missing":  "File not found.",
    },
    "zh": {
        "title":        "会议转录",
        "subtitle":     "自动转录会议录像，识别说话人",
        "drop_hint":    "将视频拖到此处  ·  或点击选择",
        "no_file":      "未选择文件",
        "xlang_label":  "转录语言",
        "more":         "⚙  更多设置  ▸",
        "less":         "⚙  更多设置  ▾",
        "diarize":      "说话人分离",
        "model_lbl":    "模型",
        "device_lbl":   "设备",
        "speaker_lbl":  "说话人数",
        "transcribe":   "开始转录",
        "running":      "转录中…",
        "log_show":     "▸  日志",
        "log_hide":     "▾  日志",
        "open_btn":     "打开转录文件 →",
        "err_no_file":  "请先选择视频文件。",
        "err_missing":  "文件不存在。",
    },
}

AUDIO_LANGS = {
    "Auto-detect / 自动": None,
    "English":  "en",
    "中文":     "zh",
    "日本語":   "ja",
    "한국어":   "ko",
    "Español":  "es",
    "Français": "fr",
    "Deutsch":  "de",
}
MODELS   = ["large-v3", "medium", "small", "base", "tiny"]
DEVICES  = ["auto", "cuda", "cpu"]
SPEAKERS = ["auto", "2", "3", "4", "5"]

FONT      = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 11, "bold")
FONT_HEAD = ("Segoe UI", 17, "bold")
FONT_TINY = ("Segoe UI", 9)
MONO      = ("Segoe UI", 9)

STEP_LABELS = {
    "[1/4]": "Converting audio…",
    "[2/4]": "Running Whisper…",
    "[3/4]": "Diarizing speakers…",
    "[4/4]": "Merging results…",
}


class App(TkinterDnD.Tk if _DND else tk.Tk):

    def __init__(self):
        super().__init__()
        self._ui_lang     = "en"
        self._theme_key   = "dark" if _system_is_dark() else "light"
        self._c           = THEMES[self._theme_key]
        self._video_path:  Path | None = None
        self._output_path: Path | None = None
        self._running         = False
        self._settings_open   = False
        self._log_open        = True
        self._divs: list[tk.Frame] = []

        self.title("meeting-transcriber")
        self.resizable(False, False)
        self.configure(bg=self._c["bg"])
        self._build()
        self._retranslate()
        self._retheme()
        self._autosize()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        c = self._c

        # Header
        self._f_hdr = tk.Frame(self, bg=c["bg"])
        self._f_hdr.pack(fill="x", padx=24, pady=(20, 14))

        self._f_text = tk.Frame(self._f_hdr, bg=c["bg"])
        self._f_text.pack(side="left", fill="both", expand=True)
        self._lbl_title    = tk.Label(self._f_text, bg=c["bg"], font=FONT_HEAD)
        self._lbl_title.pack(anchor="w")
        self._lbl_subtitle = tk.Label(self._f_text, bg=c["bg"], font=FONT)
        self._lbl_subtitle.pack(anchor="w", pady=(2, 0))

        self._f_ctrl = tk.Frame(self._f_hdr, bg=c["bg"])
        self._f_ctrl.pack(side="right", anchor="n", pady=4)
        self._btn_theme = tk.Button(self._f_ctrl, text="◐", width=3,
                                     command=self._toggle_theme,
                                     font=FONT, relief="flat", cursor="hand2", bd=0)
        self._btn_theme.pack(side="left", padx=(0, 4))
        self._btn_uilang = tk.Button(self._f_ctrl, width=4,
                                      command=self._toggle_ui_lang,
                                      font=FONT_TINY, relief="flat", cursor="hand2", bd=0)
        self._btn_uilang.pack(side="left")

        self._divs.append(self._mk_div())

        # Drop zone
        self._drop_zone = tk.Frame(self, cursor="hand2", bd=2, relief="flat")
        self._drop_zone.pack(fill="x", padx=24, pady=14)
        for w in (self._drop_zone,):
            w.bind("<Button-1>", lambda _: self._pick_file())

        self._lbl_drop = tk.Label(self._drop_zone, font=FONT, pady=20, cursor="hand2")
        self._lbl_drop.pack()
        self._lbl_drop.bind("<Button-1>", lambda _: self._pick_file())

        self._lbl_file = tk.Label(self._drop_zone, font=FONT_TINY, cursor="hand2")
        self._lbl_file.pack(pady=(0, 14))
        self._lbl_file.bind("<Button-1>", lambda _: self._pick_file())

        if _DND:
            self._drop_zone.drop_target_register(DND_FILES)
            self._drop_zone.dnd_bind("<<Drop>>", self._on_drop)

        self._divs.append(self._mk_div())

        # Transcription language row
        self._f_xlang = tk.Frame(self, bg=c["bg"])
        self._f_xlang.pack(fill="x", padx=24, pady=(10, 0))
        self._lbl_xlang = tk.Label(self._f_xlang, bg=c["bg"], font=FONT, width=22, anchor="w")
        self._lbl_xlang.pack(side="left")
        self._var_xlang = tk.StringVar(value="Auto-detect / 自动")
        ttk.Combobox(self._f_xlang, textvariable=self._var_xlang,
                     values=list(AUDIO_LANGS.keys()),
                     state="readonly", width=18, font=FONT).pack(side="left")

        # Settings toggle button
        self._btn_settings = tk.Button(self, command=self._toggle_settings,
                                        font=FONT_TINY, relief="flat", cursor="hand2",
                                        bd=0, anchor="w", padx=24, pady=8)
        self._btn_settings.pack(fill="x")

        # Settings panel (hidden by default)
        self._panel_settings = tk.Frame(self, bg=c["bg"])
        grid = tk.Frame(self._panel_settings, bg=c["bg"])
        grid.pack(fill="x", padx=24, pady=(0, 4))

        self._var_model    = tk.StringVar(value=config.WHISPER_MODEL)
        self._var_device   = tk.StringVar(value=config.DEVICE)
        self._var_speakers = tk.StringVar(value="auto")

        rows = [
            ("_lbl_model",   "model_lbl",   self._var_model,    MODELS),
            ("_lbl_device",  "device_lbl",  self._var_device,   DEVICES),
            ("_lbl_speaker", "speaker_lbl", self._var_speakers, SPEAKERS),
        ]
        for i, (attr, _, var, vals) in enumerate(rows):
            lbl = tk.Label(grid, bg=c["bg"], font=FONT, width=12, anchor="w")
            lbl.grid(row=i, column=0, sticky="w", pady=3)
            setattr(self, attr, lbl)
            ttk.Combobox(grid, textvariable=var, values=vals,
                         state="readonly", width=14, font=FONT).grid(
                row=i, column=1, sticky="w", padx=10, pady=3)

        self._var_diarize = tk.BooleanVar(value=True)
        self._cb_diarize = tk.Checkbutton(self._panel_settings,
                                           variable=self._var_diarize,
                                           font=FONT, cursor="hand2",
                                           bg=c["bg"], activebackground=c["bg"])
        self._cb_diarize.pack(anchor="w", padx=24, pady=(2, 8))

        self._divs.append(self._mk_div())

        # Action row
        self._f_action = tk.Frame(self, bg=c["bg"])
        self._f_action.pack(fill="x", padx=24, pady=12)
        self._btn_start = tk.Button(self._f_action, command=self._start,
                                     font=FONT_BOLD, relief="flat",
                                     padx=20, pady=8, cursor="hand2", bd=0)
        self._btn_start.pack(side="left")
        self._lbl_status = tk.Label(self._f_action, bg=c["bg"], font=FONT)
        self._lbl_status.pack(side="left", padx=14)

        # Progress bar (shown only while running)
        self._progress = ttk.Progressbar(self, mode="indeterminate", length=W - 48)

        # Log toggle button
        self._btn_log = tk.Button(self, command=self._toggle_log,
                                   font=FONT_TINY, relief="flat", cursor="hand2",
                                   bd=0, anchor="w", padx=24, pady=6)
        self._btn_log.pack(fill="x")

        # Log panel
        self._panel_log = tk.Frame(self, bg=c["bg"])
        self._f_log_inner = tk.Frame(self._panel_log, bg=c["bg2"])
        self._f_log_inner.pack(fill="x", padx=24, pady=(0, 8))
        self._log_text = tk.Text(self._f_log_inner, width=62, height=16,
                                  font=MONO, relief="flat", bd=0,
                                  state="disabled", wrap="word")
        sb = tk.Scrollbar(self._f_log_inner, command=self._log_text.yview, relief="flat")
        self._log_text.configure(yscrollcommand=sb.set)
        self._log_text.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        sb.pack(side="right", fill="y")
        self._panel_log.pack(fill="x")

        # Open button (shown after done)
        self._btn_open = tk.Button(self, command=self._open_output,
                                    font=FONT, relief="flat",
                                    padx=14, pady=6, cursor="hand2", bd=0)
        self._btn_open.pack(anchor="w", padx=24, pady=(0, 16))
        self._btn_open.pack_forget()

    def _mk_div(self) -> tk.Frame:
        f = tk.Frame(self, height=1)
        f.pack(fill="x", padx=24)
        return f

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _retheme(self):
        c = self._c

        self.configure(bg=c["bg"])
        for f in (self._f_hdr, self._f_text, self._f_ctrl,
                  self._f_xlang, self._f_action, self._panel_settings,
                  self._panel_log):
            f.configure(bg=c["bg"])
        self._panel_settings.winfo_children()[0].configure(bg=c["bg"])  # grid frame

        for div in self._divs:
            div.configure(bg=c["bg3"])

        self._lbl_title.configure(bg=c["bg"], fg=c["fg"])
        self._lbl_subtitle.configure(bg=c["bg"], fg=c["fg_dim"])

        for btn in (self._btn_theme, self._btn_uilang):
            btn.configure(bg=c["bg2"], fg=c["fg"],
                          activebackground=c["bg3"], activeforeground=c["fg"])

        self._drop_zone.configure(bg=c["bg2"],
                                   highlightbackground=c["bg3"],
                                   highlightthickness=2)
        self._lbl_drop.configure(bg=c["bg2"], fg=c["fg_dim"])
        self._lbl_file.configure(bg=c["bg2"],
                                  fg=c["fg"] if self._video_path else c["fg_dim"])

        self._lbl_xlang.configure(bg=c["bg"], fg=c["fg"])

        for btn in (self._btn_settings, self._btn_log):
            btn.configure(bg=c["bg"], fg=c["fg_dim"],
                          activebackground=c["bg"], activeforeground=c["fg"])

        for attr in ("_lbl_model", "_lbl_device", "_lbl_speaker"):
            lbl = getattr(self, attr)
            lbl.configure(bg=c["bg"], fg=c["fg"])

        self._cb_diarize.configure(bg=c["bg"], fg=c["fg"],
                                    selectcolor=c["bg2"],
                                    activebackground=c["bg"],
                                    activeforeground=c["fg"])

        self._lbl_status.configure(bg=c["bg"])
        self._btn_start.configure(
            bg=c["bg3"] if self._running else c["success"],
            fg="#ffffff",
            activebackground=c["bg3"] if self._running else c["success"],
            activeforeground="#ffffff",
        )

        self._f_log_inner.configure(bg=c["bg2"])
        self._log_text.configure(bg=c["bg2"], fg=c["fg"], insertbackground=c["fg"])

        self._btn_open.configure(bg=c["bg2"], fg=c["accent"],
                                  activebackground=c["bg3"],
                                  activeforeground=c["accent"])

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox",
                         fieldbackground=c["bg2"], background=c["bg2"],
                         foreground=c["fg"], selectbackground=c["bg3"],
                         selectforeground=c["fg"], bordercolor=c["bg3"],
                         arrowcolor=c["fg_dim"])
        style.configure("TProgressbar",
                         troughcolor=c["bg2"], background=c["accent"])
        style.configure("Vertical.TScrollbar",
                         background=c["bg3"], troughcolor=c["bg2"],
                         bordercolor=c["bg2"], arrowcolor=c["fg_dim"])

    def _toggle_theme(self):
        self._theme_key = "light" if self._theme_key == "dark" else "dark"
        self._c = THEMES[self._theme_key]
        self._retheme()

    # ── i18n ──────────────────────────────────────────────────────────────────

    def _t(self, key: str) -> str:
        return I18N[self._ui_lang].get(key, key)

    def _retranslate(self):
        t = self._t
        self._lbl_title.configure(text=t("title"))
        self._lbl_subtitle.configure(text=t("subtitle"))
        self._btn_uilang.configure(text="中文" if self._ui_lang == "en" else "EN")
        self._lbl_drop.configure(text=t("drop_hint"))
        if not self._video_path:
            self._lbl_file.configure(text=t("no_file"))
        self._lbl_xlang.configure(text=t("xlang_label"))
        self._btn_settings.configure(text=t("less") if self._settings_open else t("more"))
        self._lbl_model.configure(text=t("model_lbl"))
        self._lbl_device.configure(text=t("device_lbl"))
        self._lbl_speaker.configure(text=t("speaker_lbl"))
        self._cb_diarize.configure(text=t("diarize"))
        self._btn_start.configure(text=t("transcribe"))
        self._btn_log.configure(text=t("log_hide") if self._log_open else t("log_show"))
        self._btn_open.configure(text=t("open_btn"))

    def _toggle_ui_lang(self):
        self._ui_lang = "zh" if self._ui_lang == "en" else "en"
        self._retranslate()

    # ── File ──────────────────────────────────────────────────────────────────

    def _pick_file(self):
        exts = " ".join(f"*{e}" for e in config.WATCH_EXTENSIONS)
        path = filedialog.askopenfilename(
            title="Select video or audio file",
            filetypes=[("Video / Audio", exts), ("All files", "*.*")],
        )
        if path:
            self._set_file(Path(path))

    def _on_drop(self, event):
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        self._set_file(Path(raw))

    def _set_file(self, path: Path):
        self._video_path = path
        name = path.name
        self._lbl_file.configure(
            text=name if len(name) <= 46 else f"…{name[-44:]}",
            fg=self._c["fg"],
        )
        self._btn_open.pack_forget()
        self._lbl_status.configure(text="")

    # ── Transcription ─────────────────────────────────────────────────────────

    def _start(self):
        if self._running:
            return
        if not self._video_path:
            self._set_status(self._t("err_no_file"), "danger")
            return
        if not self._video_path.exists():
            self._set_status(self._t("err_missing"), "danger")
            return

        self._running = True
        self._btn_open.pack_forget()
        self._btn_start.configure(state="disabled", bg=self._c["bg3"])
        self._set_status(self._t("running"))
        self._clear_log()
        self._progress.pack(fill="x", padx=24, pady=(0, 4))
        self._progress.start(12)
        self._autosize()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        cmd = [sys.executable, str(TRANSCRIBE_SCRIPT), str(self._video_path)]

        lang = AUDIO_LANGS[self._var_xlang.get()]
        if lang:
            cmd += ["--language", lang]
        if not self._var_diarize.get():
            cmd += ["--transcribe-only"]

        model = self._var_model.get()
        if model != config.WHISPER_MODEL:
            cmd += ["--model", model]

        device = self._var_device.get()
        if device != config.DEVICE:
            cmd += ["--device", device]

        spk = self._var_speakers.get()
        if spk != "auto":
            cmd += ["--max-speakers", spk]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            line = line.rstrip()
            self.after(0, self._append_log, line)
            for marker, label in STEP_LABELS.items():
                if marker in line:
                    self.after(0, self._set_status, label)
                    break
        proc.wait()

        if proc.returncode == 0:
            self._output_path = config.TRANSCRIPT_DIR / f"{self._video_path.stem}.md"
            self.after(0, self._on_done)
        else:
            self.after(0, self._on_error, proc.returncode)

    def _on_done(self):
        self._progress.stop()
        self._progress.pack_forget()
        self._running = False
        self._set_status(f"✓  {self._output_path.name}", "success")
        self._btn_start.configure(state="normal", bg=self._c["success"])
        self._btn_open.pack(anchor="w", padx=24, pady=(0, 16))
        self._autosize()

    def _on_error(self, code: int):
        self._progress.stop()
        self._progress.pack_forget()
        self._running = False
        self._set_status(f"✗  exit {code} — see log", "danger")
        self._btn_start.configure(state="normal", bg=self._c["success"])
        if not self._log_open:
            self._toggle_log()
        self._autosize()

    def _open_output(self):
        if self._output_path and self._output_path.exists():
            os.startfile(str(self._output_path))

    # ── Panel toggles ─────────────────────────────────────────────────────────

    def _toggle_settings(self):
        self._settings_open = not self._settings_open
        if self._settings_open:
            self._panel_settings.pack(fill="x", after=self._btn_settings)
        else:
            self._panel_settings.pack_forget()
        self._retranslate()
        self._autosize()

    def _toggle_log(self):
        self._log_open = not self._log_open
        if self._log_open:
            self._panel_log.pack(fill="x", after=self._btn_log)
        else:
            self._panel_log.pack_forget()
        self._retranslate()
        self._autosize()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, text: str, level: str = "dim"):
        color = {"dim": self._c["fg_dim"],
                 "success": self._c["success"],
                 "danger": self._c["danger"]}.get(level, self._c["fg_dim"])
        self._lbl_status.configure(text=text, fg=color)

    def _append_log(self, line: str):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", line + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _clear_log(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    def _autosize(self):
        self.update_idletasks()
        self.geometry(f"{W}x{self.winfo_reqheight()}")


if __name__ == "__main__":
    App().mainloop()
