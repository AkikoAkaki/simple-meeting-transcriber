# Design Journal — Simple Video Transcriber

> Key decisions, rationale, and technical details.  
> Started: 2026-06-15.  Updated as decisions are made.

---

## 1. Project Identity

### Name: `simple-video-transcriber` / Simple Video Transcriber
- Originally `meeting-transcriber`. Changed because the tool works for any video/audio, not just meetings.
- `Simple Video Transcriber` — deliberately English-only title. Self-descriptive, zero ambiguity. Chinese users get the zh I18N UI; the English title communicates the tool's purpose to everyone.
- Alternatives considered: `whisperbox` (cute but opaque), `autocaption` (misleads toward subtitle-only), `localtalk` (vague). Chose clarity over cleverness.

### Target audience
- **Non-technical users** who don't edit config files, don't use terminals. The README no longer mentions config.py editing or CLI commands in the main flow.

---

## 2. Architecture

### Subprocess pattern (GUI/watcher → transcribe.py)
- **Decision**: `gui.py` and `watch.py` spawn `transcribe.py` as a child process and parse its stdout, rather than importing it as a library.
- **Why**: Loading `torch` + `faster-whisper` + `pyannote` takes 5-15 seconds and consumes 2-4 GB GPU memory. Running these in the GUI process would mean every launch is slow and VRAM stays allocated while idle. The subprocess pattern:
  - GUI startup <1s, ~50 MB RAM
  - Crash isolation: ML crash doesn't kill the GUI
  - CLI remains independently usable
- **Cost**: stdout parsing is fragile (string-matching on `[1/4]`, `ERROR:`, `✓ Done →`). Acceptable for now — a JSON-lines protocol was considered but rejected under YAGNI.

### File structure (4 modules, no sub-packages)
- `gui.py` (~1200 lines), `transcribe.py` (~500 lines), `watch.py` (~180 lines), `config.py` (~65 lines)
- **Decision**: keep single-file modules. At <500 lines each, splitting adds import complexity without meaningful benefit.
- `transcribe.py` was considered for splitting into `pipeline.py` + `formats.py`. Rejected: the formats are small, and the single-file layout makes end-to-end understanding trivial.

### Configuration flow
- **`config.py`**: module-level globals (`WHISPER_MODEL`, `DEVICE`, etc.). Mutated at runtime by CLI args in `transcribe.py:_main()`.
- **`hf_token.txt`**: user's HuggingFace token, saved by GUI Save button.
- **`user_settings.json`**: output directory preference, saved by GUI Browse button. Read by `watch.py` to sync output location.
- **Known smell**: global mutation of config. Accepted because `transcribe.py` always runs as `__main__` (never imported as a library). If that ever changes, refactor to parameter-passing.

---

## 3. UI / UX

### Onboarding panel
- **Problem**: new users had no idea what to do. They saw a blank GUI with a Transcribe button.
- **Solution**: first-launch detection (no `hf_token.txt`) triggers a 3-step guided panel:
  1. Get HF token (with link + inline paste field + step-by-step hint)
  2. Accept model licenses (two links + guidance)
  3. Output folder selection
- **Design**: replaces the old warning banner. When token is saved, collapses to a "✓ Ready" bar. The Transcribe button is hidden until setup completes (prevents confusing failure).
- **Panel shows immediately** (packed in `_build()`) — no waiting for background startup checks. Content updates asynchronously when checks finish.

### Startup dependency checks
- Run in background thread 200ms after window opens.
- Check ffmpeg, torch, faster-whisper, pyannote.audio, HF token.
- Results displayed in log pane + color-coded banner for missing items.
- **`except ImportError` → `except Exception`**: On Windows, `import torch` can throw `OSError` (missing VC++ runtime, DLL load failure), not `ImportError`. Changed all three import guards to `except Exception` so the thread never silently crashes.

### Cancel button
- While transcription runs, the Transcribe button becomes a red Cancel button.
- Stores `subprocess.Popen` as `self._proc`. `terminate()` kills the child process.
- **`_cancelled` flag**: distinguishes user-initiated cancel from genuine crash. Cancel shows neutral "Cancelled" status; crash shows `✗ exit {code}` error.
- Window close (`WM_DELETE_WINDOW`) also terminates the subprocess — prevents orphan GPU processes.

### Dark mode detection
- **Windows**: reads `AppsUseDarkTheme` registry key.
- **macOS**: `defaults read -g AppleInterfaceStyle`.
- **Linux**: tries `kreadconfig5` (KDE) then `gsettings` (GNOME). Falls back to light.
- KDE support added because it's the 2nd largest Linux desktop and costs 3 lines.

### CJK font fallback
- Priority: `Microsoft YaHei UI` → `PingFang SC` → `Source Han Sans SC` → platform sans-serif fallback.
- Platform fallback: macOS `Helvetica Neue`, Windows `Segoe UI`, Linux `DejaVu Sans`.

### i18n (EN / 中文)
- All UI strings in `I18N` dict, keyed by `"en"` / `"zh"`.
- `_t(key)` dispatches. `_retranslate()` reapplies all text. Theme toggle preserves language.
- Onboarding panel fully bilingual.

### Progress display
- Whisper outputs `"... N segments, up to HH:MM:SS"` every 20 segments.
- Regex capture shows live "Transcribing — 40 segs · 00:02:15" in status bar.
- `continue` skips the generic `STEP_LABELS` match for these lines.

### Output format selector
- Markdown (`.md`, default), SRT subtitles (`.srt`), plain text (`.txt`).
- GUI combobox + `--output-format` CLI arg.
- Output path fallback respects format suffix — if `✓ Done →` line fails to parse, constructs `{stem}.{fmt_suffix}`.

### Output directory
- Always-visible row above Settings. Saves to `user_settings.json`.
- `_outdir_snapshot` captured at transcription start — immune to user changing dir mid-run.
- `watch.py` reads `user_settings.json` and passes `--output-dir` to transcribe.py.

### Pipeline mode selector
- Replaced the binary "Speaker diarization" checkbox with a 3-way combobox:
  - Full pipeline (transcribe + diarize)
  - Transcribe only (no HF token needed)
  - Re-diarize only (uses cached Whisper output)
- Exposes `--diarize-only` in GUI for the first time.

### Filename display truncation
- `f"{name[:20]}…{name[-24:]}"` — preserves both prefix (date/team) and suffix (extension).
- Old behavior `f"…{name[-44:]}"` lost the beginning, which often contains the most distinguishing info.

---

## 4. Safety & Robustness

### Cache integrity
- **Corrupt JSON** (`JSONDecodeError` / `OSError`): delete cache, re-run. Applies to both `whisper_json` and `diarize_json`.
- **Empty `[]` is valid**: treated as "no speech detected" cache (return immediately, don't re-run). Previously empty was conflated with corrupt, causing silent files to re-run Whisper every launch.
- **WAV cache <1KB**: treated as corrupt (impossible for a valid WAV header). Deleted, re-converted.

### Subprocess safety
- `self._proc` assigned immediately after `Popen()`. Stored as instance variable so cancel/close can access it.
- Stdout read with `encoding="utf-8", errors="replace"` — prevents GBK decode crash on Windows.
- Both GUI and watcher use this encoding.

### Thread safety
- **gui.py**: UI updates via `self.after(0, callback)` — canonical tkinter thread safety. `_running` flag gates re-entry.
- **watch.py**: `_pending` dict shared between watchdog observer thread and main loop. Protected by `threading.Lock` — snapshot under lock before iterating, all mutations under lock.

### `torch.load` safety
- PyTorch ≥2.6 defaults `weights_only=True`. pyannote checkpoints need `False`.
- **Context manager** `_allow_unsafe_torch_load()` scoped to `Pipeline.from_pretrained()` only — not a global monkey-patch. Restores original `torch.load` on exit.

### `torchaudio.info()` protection
- Wrapped in try/except. Falls back to `segments[-1]["end"]` for duration. Prevents late-stage crash after successful transcription.

### Error propagation
- `FATAL:` / `ERROR:` markers in stdout parsed by GUI's `STEP_LABELS` to show status.
- `_on_error()` displays exit code. `_cancelled` flag distinguishes user cancel from crash.
- `run_diarization()` returns `[]` on failure (doesn't `sys.exit()`) — allows pipeline to continue without diarization.

### SRT timestamp edge case
- `min(round((sec - int(sec)) * 1000), 999)` — clamps milliseconds to valid range, preventing `1000` from appearing in SRT timestamp.

---

## 5. Performance

### `word_timestamps=False`
- faster-whisper's `word_timestamps=True` adds ~20% compute overhead.
- The merge/overlap logic only uses segment-level start/end — word-level data is discarded.
- Changed to `False` with no quality loss.

### `vad_filter=True` with `min_silence_duration_ms=2000`
- Voice Activity Detection skips silent segments, reducing Whisper work and improving segment boundaries.

---

## 6. Distribution

### Install scripts (non-technical users)
- **Windows**: `install.bat` — `winget install python3 ffmpeg` + `pip install -r requirements.txt`. `start.bat` — `python gui.py`.
- **macOS**: `install.command` — `brew install python@3.12 ffmpeg` + `pip3 install --break-system-packages -r requirements.txt`. `start.command` with `read -p` so errors are visible.
- **`--break-system-packages`**: macOS Homebrew Python is PEP 668 externally-managed. Without this flag, `pip3 install` fails with an opaque error.
- **`start.command` has `read -p`**: prevents Terminal from closing instantly on error (user can't see what went wrong).

### README
- English root + Chinese `README.zh.md` linked from top.
- Main flow: Download zip → double-click install → double-click start → drag file.
- CLI / config.py / watcher details collapsed in `<details>` sections.

### Dependencies
- `requirements.txt`: `faster-whisper`, `pyannote.audio`, `torch` (with GPU reinstall note), `torchaudio`, `watchdog`, `tkinterdnd2` (optional).
- `torch` listed explicitly even though `torchaudio` depends on it — GPU users need to see the CUDA reinstall note.

---

## 7. Explicit Non-choices (YAGNI / KISS)

| Idea | Why rejected |
|------|-------------|
| JSON-lines stdout protocol | Current string-matching works. No real breakage yet. ~60 line change across two files. |
| `sys.exit()` → custom exceptions | Only `transcribe.py` calls these functions. No one imports them. Internal refactor with zero user benefit. |
| Split transcribe.py into sub-modules | <500 lines. Single file is easier to read end-to-end. |
| MVC/MVP refactor of GUI | 1200-line tkinter app. Single-class pattern is standard. |
| Merge `hf_token.txt` into `user_settings.json` | Token is a secret — keeping it in a gitignored separate file adds a layer of safety against accidental commit. |
| Configuration GUI (in-app config editor) | Users set token once via onboarding. Everything else has sensible defaults. |
| Auto-detect GPU and install CUDA torch | Too fragile across platforms. README note is sufficient. |
| Multi-file drag-and-drop / batch processing | Watcher handles batch via inbox folder. GUI is single-file by design. |
