#!/usr/bin/env python3
"""
meeting-transcriber — gui.py
Minimal GUI. Follows system dark/light theme. UI language: EN / 中文.

Optional: pip install tkinterdnd2   (enables drag-and-drop)
Usage:    python gui.py
"""

import json
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import config

TRANSCRIBE_SCRIPT = Path(__file__).parent / "transcribe.py"
TOKEN_FILE        = Path(__file__).parent / "hf_token.txt"
SETTINGS_FILE     = Path(__file__).parent / "user_settings.json"
W = 560   # fixed window width

# ── Drag-and-drop (optional) ──────────────────────────────────────────────────
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND = True
except ImportError:
    _DND = False

# ── CJK font detection ───────────────────────────────────────────────────────
def _detect_cjk_font() -> str:
    """Return the best available sans-serif font with CJK coverage."""
    try:
        from tkinter import font as tkfont
        avail = set(tkfont.families())
        for cand in ("Microsoft YaHei UI", "Microsoft YaHei",
                     "PingFang SC", "Hiragino Sans GB",
                     "Source Han Sans SC", "Noto Sans CJK SC"):
            if cand in avail:
                return cand
    except Exception:
        pass
    # Platform-appropriate sans-serif fallback when no CJK font found
    if sys.platform == "darwin":
        return "Helvetica Neue"
    if sys.platform == "win32":
        return "Segoe UI"
    return "DejaVu Sans"   # Linux default

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
    elif sys.platform == "darwin":
        try:
            r = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=2)
            return r.stdout.strip().lower() == "dark"
        except Exception:
            pass
    else:  # Linux / other
        try:
            r = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                capture_output=True, text=True, timeout=2)
            return "dark" in r.stdout.lower()
        except Exception:
            pass
    return False

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
        "warn":    "#ff9f0a",
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
        "warn":    "#ff9500",
    },
}

# ── i18n ──────────────────────────────────────────────────────────────────────
I18N = {
    "en": {
        "title":         "meeting-transcriber",
        "subtitle":      "Transcribe meetings with speaker labels",
        "drop_hint":     "Drop video here  ·  or click to browse",
        "no_file":       "No file selected",
        "xlang_label":   "Transcription language",
        "more":          "⚙  More settings  ▸",
        "less":          "⚙  More settings  ▾",
        "diarize":       "Speaker diarization",
        "model_lbl":     "Model",
        "device_lbl":    "Device",
        "speaker_lbl":   "Speakers",
        "token_lbl":     "HF Token",
        "token_hint":    "paste token here",
        "token_save":    "Save",
        "token_saved":   "Saved ✓",
        "token_show":    "👁",
        "transcribe":    "Transcribe",
        "running":       "Running…",
        "cancel":        "Cancel",
        "log_show":      "▸  Log",
        "log_hide":      "▾  Log",
        "open_btn":      "Open transcript →",
        "err_no_file":   "Please select a file first.",
        "err_missing":   "File not found.",
        "outdir_lbl":    "Output folder",
        "browse_btn":    "Browse",
        "details_btn":   "Details ↓",
        "set_token_btn": "Set token ▸",
    },
    "zh": {
        "title":         "会议转录",
        "subtitle":      "自动转录会议录像，识别说话人",
        "drop_hint":     "将视频拖到此处  ·  或点击选择",
        "no_file":       "未选择文件",
        "xlang_label":   "转录语言",
        "more":          "⚙  更多设置  ▸",
        "less":          "⚙  更多设置  ▾",
        "diarize":       "说话人分离",
        "model_lbl":     "模型",
        "device_lbl":    "设备",
        "speaker_lbl":   "说话人数",
        "token_lbl":     "HF Token",
        "token_hint":    "在此粘贴 token",
        "token_save":    "保存",
        "token_saved":   "已保存 ✓",
        "token_show":    "👁",
        "transcribe":    "开始转录",
        "running":       "转录中…",
        "cancel":        "取消",
        "log_show":      "▸  日志",
        "log_hide":      "▾  日志",
        "open_btn":      "打开转录文件 →",
        "err_no_file":   "请先选择视频文件。",
        "err_missing":   "文件不存在。",
        "outdir_lbl":    "输出目录",
        "browse_btn":    "浏览",
        "details_btn":   "查看详情 ↓",
        "set_token_btn": "设置 Token ▸",
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

def _make_fonts(face: str):
    """Build the (FONT, FONT_BOLD, FONT_HEAD, FONT_TINY) tuple for a font face."""
    return (face, 10), (face, 11, "bold"), (face, 13, "bold"), (face, 9)

FONT, FONT_BOLD, FONT_HEAD, FONT_TINY = _make_fonts("Segoe UI")

STEP_LABELS = {
    "[1/4]":               "Step 1/4 — Converting audio…",
    "[2/4]":               "Step 2/4 — Loading Whisper model…",
    "Transcribing [":      "Step 2/4 — Transcribing…",
    "segments, up to":     "Step 2/4 — Transcribing…",
    "[3/4]":               "Step 3/4 — Loading diarization model…",
    "Running diarization": "Step 3/4 — Diarizing (may take 10–20 min)…",
    "[4/4]":               "Step 4/4 — Merging results…",
    "✓ Done":              "Done!",
    "ERROR:":              "Error — see log below",
    "FATAL:":              "Fatal error — see log below",
}


class App(TkinterDnD.Tk if _DND else tk.Tk):

    def __init__(self):
        super().__init__()

        # Update font globals to best available CJK-capable face (needs Tk to exist)
        global FONT, FONT_BOLD, FONT_HEAD, FONT_TINY
        FONT, FONT_BOLD, FONT_HEAD, FONT_TINY = _make_fonts(_detect_cjk_font())

        self._ui_lang     = "en"
        self._theme_key   = "dark" if _system_is_dark() else "light"
        self._c           = THEMES[self._theme_key]
        self._video_path:  Path | None = None
        self._output_path: Path | None = None
        self._running       = False
        self._proc: subprocess.Popen | None = None
        self._settings_open = False
        self._log_open      = True
        self._token_visible = False
        self._checks: dict  = {}
        self._divs: list[tk.Frame] = []

        # Output directory — persisted in user_settings.json
        _us = self._load_user_settings()
        self._var_outdir = tk.StringVar(
            value=_us.get("transcript_dir", str(config.TRANSCRIPT_DIR)))

        self.title("meeting-transcriber")
        self.resizable(False, False)
        self.configure(bg=self._c["bg"])
        self.option_add("*Font", FONT)
        self._build()
        self._retranslate()
        self._retheme()
        self._center()
        # Run dependency checks after window is rendered
        self.after(200, lambda: threading.Thread(
            target=self._run_startup_checks, daemon=True).start())

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        c = self._c

        # ── Header ──
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

        # ── Banner (hidden until checks run) ──
        self._banner = tk.Frame(self, bg=c["bg3"])
        self._banner_lbl = tk.Label(self._banner, font=FONT_TINY,
                                     bg=c["bg3"], fg=c["fg"], anchor="w")
        self._banner_lbl.pack(side="left", padx=(10, 4), pady=7)
        self._banner_btn = tk.Button(self._banner, font=FONT_TINY,
                                      relief="flat", cursor="hand2", bd=0,
                                      padx=6, pady=2)
        self._banner_btn.pack(side="right", padx=(4, 10), pady=5)
        # Banner is NOT packed here — _update_banner() does it

        # ── Drop zone ──
        self._drop_zone = tk.Frame(self, cursor="hand2", bd=2, relief="flat")
        self._drop_zone.pack(fill="x", padx=24, pady=14)
        self._drop_zone.bind("<Button-1>", lambda _: self._pick_file())

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

        # ── Transcription language row ──
        self._f_xlang = tk.Frame(self, bg=c["bg"])
        self._f_xlang.pack(fill="x", padx=24, pady=(10, 0))
        self._lbl_xlang = tk.Label(self._f_xlang, bg=c["bg"], font=FONT, width=22, anchor="w")
        self._lbl_xlang.pack(side="left")
        self._var_xlang = tk.StringVar(value="Auto-detect / 自动")
        ttk.Combobox(self._f_xlang, textvariable=self._var_xlang,
                     values=list(AUDIO_LANGS.keys()),
                     state="readonly", width=18, font=FONT).pack(side="left")

        # ── Output folder row (always visible) ──
        self._f_outdir = tk.Frame(self, bg=c["bg"])
        self._f_outdir.pack(fill="x", padx=24, pady=(8, 2))

        self._lbl_outdir = tk.Label(self._f_outdir, bg=c["bg"], font=FONT,
                                     width=12, anchor="w")
        self._lbl_outdir.pack(side="left")

        # Pack Browse first so entry can fill the remaining space
        self._btn_outdir = tk.Button(self._f_outdir, command=self._browse_outdir,
                                      font=FONT_TINY, relief="flat", cursor="hand2",
                                      bd=0, padx=8, pady=3)
        self._btn_outdir.pack(side="right")

        self._entry_outdir = tk.Entry(self._f_outdir, textvariable=self._var_outdir,
                                       state="readonly", font=FONT_TINY,
                                       relief="flat", bd=1, cursor="arrow")
        self._entry_outdir.pack(side="left", fill="x", expand=True, padx=(0, 4))

        # ── Settings toggle button ──
        self._btn_settings = tk.Button(self, command=self._toggle_settings,
                                        font=FONT, relief="flat", cursor="hand2",
                                        bd=0, anchor="w", padx=24, pady=8)
        self._btn_settings.pack(fill="x")

        # ── Settings panel (hidden by default) ──
        self._panel_settings = tk.Frame(self, bg=c["bg"])

        # Grid: model / device / speakers
        self._grid_settings = tk.Frame(self._panel_settings, bg=c["bg"])
        self._grid_settings.pack(fill="x", padx=24, pady=(0, 4))

        self._var_model    = tk.StringVar(value=config.WHISPER_MODEL)
        self._var_device   = tk.StringVar(value=config.DEVICE)
        self._var_speakers = tk.StringVar(value="auto")

        rows = [
            ("_lbl_model",   self._var_model,    MODELS),
            ("_lbl_device",  self._var_device,   DEVICES),
            ("_lbl_speaker", self._var_speakers, SPEAKERS),
        ]
        for i, (attr, var, vals) in enumerate(rows):
            lbl = tk.Label(self._grid_settings, bg=c["bg"], font=FONT, width=12, anchor="w")
            lbl.grid(row=i, column=0, sticky="w", pady=3)
            setattr(self, attr, lbl)
            ttk.Combobox(self._grid_settings, textvariable=var, values=vals,
                         state="readonly", width=14, font=FONT).grid(
                row=i, column=1, sticky="w", padx=10, pady=3)

        # HF Token row
        self._f_token = tk.Frame(self._panel_settings, bg=c["bg"])
        self._f_token.pack(fill="x", padx=24, pady=(4, 2))

        self._lbl_token = tk.Label(self._f_token, bg=c["bg"], font=FONT, width=12, anchor="w")
        self._lbl_token.pack(side="left")

        self._entry_token = tk.Entry(self._f_token, show="•", font=FONT, width=22,
                                      relief="flat", bd=1)
        self._entry_token.pack(side="left", padx=(10, 4))
        # Pre-fill with saved token
        _saved = config.HF_TOKEN or (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else "")
        if _saved:
            self._entry_token.insert(0, _saved)

        self._btn_token_eye = tk.Button(self._f_token, command=self._toggle_token_vis,
                                         font=FONT_TINY, relief="flat", cursor="hand2",
                                         bd=0, padx=4)
        self._btn_token_eye.pack(side="left")

        self._btn_token_save = tk.Button(self._f_token, command=self._save_token,
                                          font=FONT_TINY, relief="flat", cursor="hand2",
                                          bd=0, padx=8, pady=3)
        self._btn_token_save.pack(side="left", padx=(4, 0))

        # Diarize checkbox
        self._var_diarize = tk.BooleanVar(value=True)
        self._cb_diarize = tk.Checkbutton(self._panel_settings,
                                           variable=self._var_diarize,
                                           font=FONT, cursor="hand2",
                                           bg=c["bg"], activebackground=c["bg"])
        self._cb_diarize.pack(anchor="w", padx=24, pady=(2, 8))

        self._divs.append(self._mk_div())

        # ── Action row ──
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

        # ── Log toggle + panel ──
        self._btn_log = tk.Button(self, command=self._toggle_log,
                                   font=FONT_TINY, relief="flat", cursor="hand2",
                                   bd=0, anchor="w", padx=24, pady=6)
        self._btn_log.pack(fill="x")

        self._panel_log = tk.Frame(self, bg=c["bg"])
        self._f_log_inner = tk.Frame(self._panel_log, bg=c["bg2"])
        self._f_log_inner.pack(fill="x", padx=24, pady=(0, 8))
        self._log_text = tk.Text(self._f_log_inner, width=62, height=16,
                                  font=FONT_TINY, relief="flat", bd=0,
                                  state="disabled", wrap="word")
        sb = tk.Scrollbar(self._f_log_inner, command=self._log_text.yview, relief="flat")
        self._log_text.configure(yscrollcommand=sb.set)
        self._log_text.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        sb.pack(side="right", fill="y")
        self._panel_log.pack(fill="x")

        # ── Open button (shown after done) ──
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
                  self._f_xlang, self._f_action,
                  self._panel_settings, self._grid_settings,
                  self._f_token, self._f_outdir, self._panel_log):
            f.configure(bg=c["bg"])

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

        self._btn_settings.configure(bg=c["bg"], fg=c["fg"],
                                      activebackground=c["bg"], activeforeground=c["fg"])
        self._btn_log.configure(bg=c["bg"], fg=c["fg_dim"],
                                 activebackground=c["bg"], activeforeground=c["fg"])

        for attr in ("_lbl_model", "_lbl_device", "_lbl_speaker", "_lbl_token",
                     "_lbl_outdir"):
            getattr(self, attr).configure(bg=c["bg"], fg=c["fg"])

        self._entry_token.configure(bg=c["bg2"], fg=c["fg"],
                                     insertbackground=c["fg"],
                                     highlightbackground=c["bg3"],
                                     highlightthickness=1)

        self._entry_outdir.configure(bg=c["bg2"], fg=c["fg"],
                                      readonlybackground=c["bg2"],
                                      highlightbackground=c["bg3"],
                                      highlightthickness=1)

        for btn in (self._btn_token_eye, self._btn_token_save, self._btn_outdir):
            btn.configure(bg=c["bg2"], fg=c["accent"],
                          activebackground=c["bg3"], activeforeground=c["accent"])

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

        # Re-apply banner colors with new theme
        self._update_banner()

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
        self._lbl_token.configure(text=t("token_lbl"))
        self._btn_token_eye.configure(text=t("token_show"))
        self._btn_token_save.configure(text=t("token_save"))
        self._lbl_outdir.configure(text=t("outdir_lbl"))
        self._btn_outdir.configure(text=t("browse_btn"))
        self._cb_diarize.configure(text=t("diarize"))
        self._btn_start.configure(text=t("transcribe"))
        self._btn_log.configure(text=t("log_hide") if self._log_open else t("log_show"))
        self._btn_open.configure(text=t("open_btn"))

    def _toggle_ui_lang(self):
        self._ui_lang = "zh" if self._ui_lang == "en" else "en"
        self._retranslate()
        self._update_banner()

    # ── Startup dependency checks ──────────────────────────────────────────────

    def _run_startup_checks(self):
        results: dict = {}
        lines: list[str] = ["=== System Check ==="]

        # ffmpeg
        try:
            r = subprocess.run(["ffmpeg", "-version"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                ver_line = (r.stdout or r.stderr or "").splitlines()[0]
                ver = ver_line.split()[2] if len(ver_line.split()) > 2 else "?"
                results["ffmpeg"] = (True, ver)
                lines.append(f"✓ ffmpeg {ver}")
            else:
                results["ffmpeg"] = (False, "error")
                lines += ["✗ ffmpeg returned an error",
                          "  → Reinstall from https://ffmpeg.org/download.html"]
        except FileNotFoundError:
            results["ffmpeg"] = (False, "not found")
            lines += [
                "✗ ffmpeg not found — cannot process video/audio",
                "  → Windows:  winget install Gyan.FFmpeg",
                "              or https://ffmpeg.org/download.html (add to PATH)",
                "  → macOS:    brew install ffmpeg",
                "  → Linux:    sudo apt install ffmpeg",
            ]
        except Exception as e:
            results["ffmpeg"] = (False, str(e))
            lines.append(f"✗ ffmpeg check failed: {e}")

        # torch
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
            if cuda_ok:
                cuda_info = f"CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}"
            else:
                cuda_info = "CPU only (no CUDA detected)"
            results["torch"] = (True, cuda_ok, torch.__version__)
            lines.append(f"✓ torch {torch.__version__} — {cuda_info}")
            if not cuda_ok:
                lines += [
                    "  Note: transcription will be slower on CPU.",
                    "  For GPU support, reinstall torch with CUDA:",
                    "  → pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121",
                    "    (replace cu121 with your CUDA version)",
                ]
        except ImportError:
            results["torch"] = (False, False, None)
            lines += [
                "✗ torch not installed",
                "  → pip install torch torchaudio",
            ]

        # faster-whisper
        try:
            import faster_whisper
            ver = getattr(faster_whisper, "__version__", "installed")
            results["whisper"] = (True, ver)
            lines.append(f"✓ faster-whisper {ver}")
        except ImportError:
            results["whisper"] = (False, None)
            lines += [
                "✗ faster-whisper not installed",
                "  → pip install faster-whisper",
            ]

        # pyannote
        try:
            import pyannote.audio
            ver = getattr(pyannote.audio, "__version__", "installed")
            results["pyannote"] = (True, ver)
            lines.append(f"✓ pyannote.audio {ver}")
        except ImportError:
            results["pyannote"] = (False, None)
            lines += [
                "✗ pyannote.audio not installed",
                "  → pip install pyannote.audio",
            ]

        # HF token
        token = config.HF_TOKEN or (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else "")
        results["token"] = bool(token)
        if token:
            lines.append("✓ HF token configured")
        else:
            lines += [
                "✗ HF token not found — speaker diarization will be disabled",
                "  → Open Settings ▸ HF Token  to set your token",
                "  → Get a free token at: https://hf.co/settings/tokens",
                "  → Accept model terms at:",
                "      https://hf.co/pyannote/speaker-diarization-3.1",
                "      https://hf.co/pyannote/segmentation-3.0",
            ]

        lines += [
            "",
            "=== Ready ===",
            "Drop a video file above to get started.",
            "Note: first run downloads model weights (~2–3 GB). Please be patient.",
            "",
        ]

        self._checks = results
        # Batch into one callback to prevent interleaving with transcription log
        def _flush(captured_lines=lines):
            if not self._running:
                for line in captured_lines:
                    self._append_log(line)
            self._update_banner()
        self.after(0, _flush)

    def _update_banner(self):
        c = self._c
        # Always unpack first so we can repack cleanly
        self._banner.pack_forget()

        if not self._checks:
            return

        ffmpeg_ok  = self._checks.get("ffmpeg",  (True,))[0]
        torch_ok   = self._checks.get("torch",   (True,))[0]
        whisper_ok = self._checks.get("whisper", (True,))[0]
        token_ok   = self._checks.get("token",   True)

        if not ffmpeg_ok:
            bg  = c["danger"]
            fg  = "#ffffff"
            msg = "⚠  ffmpeg not found — transcription will fail."
            btxt = self._t("details_btn")
            bcmd = self._scroll_log_end
        elif not torch_ok or not whisper_ok:
            bg  = c["danger"]
            fg  = "#ffffff"
            msg = "⚠  Missing packages — see log for install commands."
            btxt = self._t("details_btn")
            bcmd = self._scroll_log_end
        elif not token_ok:
            bg  = c["bg3"]
            fg  = c["fg"]
            msg = "⚠  No HF token — speaker diarization disabled."
            btxt = self._t("set_token_btn")
            bcmd = self._focus_token
        else:
            self._autosize()
            return

        self._banner.configure(bg=bg)
        self._banner_lbl.configure(text=msg, bg=bg, fg=fg)
        self._banner_btn.configure(text=btxt, command=bcmd, bg=bg, fg=fg,
                                    activebackground=bg, activeforeground=fg)
        self._banner.pack(fill="x", padx=24, pady=(4, 0), after=self._divs[0])
        self._autosize()

    # ── HF Token ──────────────────────────────────────────────────────────────

    def _toggle_token_vis(self):
        self._token_visible = not self._token_visible
        self._entry_token.configure(show="" if self._token_visible else "•")

    def _save_token(self):
        token = self._entry_token.get().strip()
        if token:
            TOKEN_FILE.write_text(token)
        elif TOKEN_FILE.exists():
            TOKEN_FILE.unlink()
        config.HF_TOKEN = token
        if self._checks:
            self._checks["token"] = bool(token)
        self._btn_token_save.configure(text=self._t("token_saved"))
        self.after(1500, lambda: self._btn_token_save.configure(text=self._t("token_save")))
        self._update_banner()

    def _focus_token(self):
        if not self._settings_open:
            self._toggle_settings()
        self._entry_token.focus_set()
        self._entry_token.icursor("end")

    def _scroll_log_end(self):
        self._log_text.see("end")

    # ── Output directory ──────────────────────────────────────────────────────

    def _browse_outdir(self):
        current = self._var_outdir.get()
        path = filedialog.askdirectory(
            title="Select output folder for transcripts",
            initialdir=current if Path(current).exists() else str(Path.home()),
        )
        if path:
            self._var_outdir.set(path)
            self._save_user_settings()

    def _load_user_settings(self) -> dict:
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_user_settings(self):
        settings = self._load_user_settings()
        settings["transcript_dir"] = self._var_outdir.get()
        SETTINGS_FILE.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

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
        self._outdir_snapshot = Path(self._var_outdir.get())
        self._btn_open.pack_forget()
        self._btn_start.configure(
            text=self._t("cancel"), command=self._cancel,
            bg=self._c["danger"], state="normal")
        self._set_status(self._t("running"))
        self._clear_log()
        self._progress.pack(fill="x", padx=24, pady=(0, 4))
        self._progress.start(12)
        self._autosize()
        threading.Thread(target=self._run, daemon=True).start()

    def _cancel(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        # _run thread will see proc exit with non-zero and call _on_error

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

        cmd += ["--output-dir", str(self._outdir_snapshot)]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=env,
        )
        done_marker = "✓ Done → "
        emitted_path: Path | None = None
        for line in proc.stdout:
            line = line.rstrip()
            self.after(0, self._append_log, line)
            if done_marker in line:
                emitted_path = Path(line.split(done_marker, 1)[1].strip())
            m = re.search(r'(\d+) segments, up to (\S+)', line)
            if m:
                n, ts = m.group(1), m.group(2)
                self.after(0, self._set_status, f"Transcribing — {n} segs · {ts}")
                continue
            for marker, label in STEP_LABELS.items():
                if marker in line:
                    self.after(0, self._set_status, label)
                    break
        proc.wait()

        if proc.returncode == 0:
            # Prefer the path transcribe.py actually wrote; fall back to the
            # expected location if the marker line wasn't captured.
            self._output_path = emitted_path or (
                self._outdir_snapshot / f"{self._video_path.stem}.md")
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
        color = {"dim":     self._c["fg_dim"],
                 "success": self._c["success"],
                 "danger":  self._c["danger"],
                 "warn":    self._c["warn"]}.get(level, self._c["fg_dim"])
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

    def _center(self):
        self.update_idletasks()
        h  = self.winfo_reqheight()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x  = (sw - W) // 2
        y  = (sh - h) // 2
        self.geometry(f"{W}x{h}+{x}+{y}")


if __name__ == "__main__":
    App().mainloop()
