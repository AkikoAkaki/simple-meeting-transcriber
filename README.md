# Simple Video Transcriber

> Transcribe any video or audio with speaker labels — fully local, no data ever leaves your machine.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-lightgrey)]()

[中文说明](README.zh.md)

---

## What it does

Drop any video or audio file onto the app — meetings, lectures, interviews, podcasts. Get back a transcript with speaker labels, timestamps, and your choice of output format — all processed locally using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [pyannote.audio](https://github.com/pyannote/pyannote-audio).

```
### [00:00:00 – 00:00:16] SPEAKER_00
Let's first look at what you've been working on.

### [00:00:16 – 00:01:10] SPEAKER_01
Can you see my screen? So I ran the layout experiment...
```

Output formats: **Markdown** (`.md`), **SRT subtitles** (`.srt`), or **plain text** (`.txt`).

---

## Features

- **Speaker diarization** — labels each segment with a speaker ID
- **Three output formats** — Markdown for notes, SRT for video editing, plain text for LLMs
- **Flexible pipeline** — run full pipeline, transcription only, or re-diarize existing output
- **Offline** — all models run locally; no API key, no cloud service
- **GPU-accelerated** — uses CUDA if available, falls back to CPU automatically
- **Auto-watcher** — drop files into `inbox/` and transcriptions run unattended
- **Cross-platform** — Windows and macOS, with one-click install scripts

---

## Install

### Windows

1. [Download the zip](https://github.com/AkikoAkaki/simple-video-transcriber/releases) and extract it
2. Double-click `install.bat`
3. Double-click `start.bat`

### macOS

1. [Download the zip](https://github.com/AkikoAkaki/simple-video-transcriber/releases) and extract it
2. Double-click `install.command` — enter your password if prompted (for Homebrew)
3. Double-click `start.command`

> **GPU acceleration (optional):** The default install uses CPU-only PyTorch. After installing, replace it with your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Quick start

1. On first launch, an onboarding panel guides you through getting a free [HuggingFace token](https://huggingface.co/settings/tokens) — takes about 2 minutes, one time only
2. Drag a file onto the drop zone (or click **Browse…**)
3. Choose your output format and pipeline mode
4. Click **Transcribe**
5. Click **Open transcript →** when done

The HuggingFace token is required only for speaker diarization. If you skip it, transcription still works — you just won't get speaker labels.

---

## Performance

Measured on RTX 4060 (8 GB VRAM), 25-minute video, `large-v3` model:

| Step | Time |
|------|------|
| Audio extraction | ~10 s |
| Whisper transcription | ~8 min |
| Speaker diarization | ~12 min |

On CPU-only hardware, expect 5–10× longer. Switch to `medium` or `small` in Settings to trade accuracy for speed.

---

## FAQ

**Do I need a GPU?**
No. CPU works out of the box, just slower. The app shows a note in the status bar if no GPU is detected.

**Which Whisper model should I use?**
`large-v3` gives the best accuracy and is the default. For weaker hardware, `medium` is a good balance. `small` and `base` are faster but noticeably less accurate.

**The detected language is wrong.**
Pick your language from the dropdown in Settings (Auto / English / 中文 / 日本語 / …).

**I get no speaker labels in the output.**
You need a HuggingFace token with the pyannote model licenses accepted. The onboarding panel walks you through this step by step.

**Can I re-run just the diarization without re-transcribing?**
Yes — select **Re-diarize only** in the Pipeline dropdown. Whisper output is cached and reused.

---

## Requirements

- Windows 10+ or macOS 12+
- Python 3.10+ (installed automatically by the install script)
- ffmpeg (installed automatically by the install script)

---

<details>
<summary>Advanced: CLI usage</summary>

```bash
# Basic transcription
python transcribe.py path/to/meeting.mp4

# Force a specific language
python transcribe.py meeting.mp4 --language zh

# Skip diarization (no token needed)
python transcribe.py meeting.mp4 --mode transcribe-only

# Re-run diarization on existing cached Whisper output
python transcribe.py meeting.mp4 --mode diarize-only

# Choose output format
python transcribe.py meeting.mp4 --output-format srt
python transcribe.py meeting.mp4 --output-format txt

# Override model, device, speaker count, or output location
python transcribe.py meeting.mp4 --model medium --device cpu --max-speakers 3 --output-dir ./my-transcripts
```

</details>

<details>
<summary>Advanced: Auto-watcher</summary>

The watcher monitors `inbox/` and transcribes new files automatically.

```bash
python watch.py            # start watching
python watch.py --dry-run  # detect files but don't transcribe
```

Files are transcribed once their size has been stable for 10 seconds (configurable). After a successful transcription, videos can optionally be moved into a `YYYY/` subfolder (see `ORGANIZE_BY_YEAR` in `config.py`).

</details>

<details>
<summary>Advanced: Configuration reference</summary>

Edit `config.py` to change defaults:

| Setting | Default | Description |
|---------|---------|-------------|
| `WATCH_DIR` | `inbox/` | Folder the watcher monitors |
| `TRANSCRIPT_DIR` | `transcripts/` | Output folder |
| `CACHE_DIR` | `cache/` | Intermediate files — safe to delete any time |
| `WHISPER_MODEL` | `large-v3` | Model size: `tiny` / `base` / `small` / `medium` / `large-v3` |
| `LANGUAGE` | `None` | `"en"` / `"zh"` / `"ja"` / … — `None` = auto-detect |
| `DEVICE` | `"auto"` | `"cuda"` / `"cpu"` / `"auto"` |
| `MAX_SPEAKERS` | `None` | Set an integer if you know the speaker count |
| `ORGANIZE_BY_YEAR` | `True` | Move processed videos into `YYYY/` subfolders |
| `STABLE_SECONDS` | `10` | Seconds a file must be unchanged before transcription starts |
| `MIN_FILE_SIZE_KB` | `100` | Ignore files smaller than this |
| `WATCH_EXTENSIONS` | `{".mp4", …}` | File types the watcher picks up |

</details>

---

## License

MIT
