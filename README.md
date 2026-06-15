# meeting-transcriber

Automatically transcribe meeting recordings with speaker labels, powered by [Whisper](https://github.com/openai/whisper) and [pyannote](https://github.com/pyannote/pyannote-audio).

Drop a video into the `inbox/` folder — a Markdown transcript appears in `transcripts/` with timestamps and speaker turns identified.

Works with English, Chinese, or any mixed-language audio. No cloud API needed; everything runs locally.

---

## Output example

```markdown
## Transcript

### [00:00:00 – 00:00:16] SPEAKER_00
Let's first look at what you've been working on.

### [00:00:16 – 00:01:10] SPEAKER_01
Can you see my screen? So I ran the layout experiment...
```

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) in PATH
- NVIDIA GPU recommended (CUDA) — CPU works but is slow
- HuggingFace account (free) for speaker diarization

---

## Quick start

**1. Clone and install**

```bash
git clone https://github.com/AkikoAkaki/meeting-transcriber.git
cd meeting-transcriber
pip install -r requirements.txt
```

**2. Configure**

Open `config.py` and set at minimum:

```python
HF_TOKEN = "hf_xxxxxxxxxxxx"   # get at https://hf.co/settings/tokens
```

Accept pyannote model terms (one-time):
- https://hf.co/pyannote/speaker-diarization-3.1
- https://hf.co/pyannote/segmentation-3.0

**3. Transcribe**

```bash
# Option A — GUI (recommended)
python gui.py

# Option B — CLI
python transcribe.py inbox/my-meeting.mp4

# Option C — drop a file into inbox/ and start the watcher
python watch.py
```

Transcripts appear in `transcripts/` as `<original-filename>.md`.

---

## Configuration

All settings are in `config.py`. The most useful ones:

| Setting | Default | Description |
|---------|---------|-------------|
| `WATCH_DIR` | `inbox/` | Folder the watcher monitors |
| `TRANSCRIPT_DIR` | `transcripts/` | Where output Markdown files go |
| `CACHE_DIR` | `cache/` | Intermediate WAV and JSON files (safe to delete) |
| `WHISPER_MODEL` | `large-v3` | Model size: `tiny` / `base` / `small` / `medium` / `large-v3` |
| `LANGUAGE` | `None` | Language code (`"en"`, `"zh"`, `"ja"`, …) or `None` for auto-detect |
| `DEVICE` | `"auto"` | `"cuda"` / `"cpu"` / `"auto"` |
| `MAX_SPEAKERS` | `None` | Set to an integer if you know the speaker count (improves accuracy) |
| `ORGANIZE_BY_YEAR` | `True` | Move videos into `YYYY/` subfolders after transcription |
| `STABLE_SECONDS` | `10` | Seconds the file size must be stable before transcription starts |
| `WATCH_EXTENSIONS` | `{".mp4", …}` | File types the watcher picks up |

---

## Manual usage

```bash
# Transcribe a file
python transcribe.py path/to/meeting.mp4

# Force language (overrides config.py)
python transcribe.py meeting.mp4 --language en
python transcribe.py meeting.mp4 --language zh

# Skip diarization (no HF token needed, faster)
python transcribe.py meeting.mp4 --transcribe-only

# Re-run diarization on an already-transcribed file
python transcribe.py meeting.mp4 --diarize-only

# Override model, device, or speaker count (one-off, without editing config.py)
python transcribe.py meeting.mp4 --model medium --device cpu --max-speakers 3
```

---

## Auto-start on login (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File setup_autostart.ps1
```

This registers the watcher as a Task Scheduler task that starts automatically at login. After that you never need to think about it.

```powershell
# Manage the task
Start-ScheduledTask    -TaskName "MeetingTranscriber-Watcher"   # start now
Stop-ScheduledTask     -TaskName "MeetingTranscriber-Watcher"   # stop
Disable-ScheduledTask  -TaskName "MeetingTranscriber-Watcher"   # pause auto-start
Unregister-ScheduledTask -TaskName "MeetingTranscriber-Watcher" # remove entirely

# Check recent activity
Get-Content watch.log -Tail 30
```

---

## How it works

```
inbox/meeting.mp4
       │
       ▼ ffmpeg
  cache/meeting_16k.wav
       │
       ├─▶ faster-whisper ──▶ cache/_meeting_whisper.json
       │
       └─▶ pyannote ─────────▶ cache/_meeting_diarize.json
                                        │
                                        ▼ merge
                              transcripts/meeting.md
```

Intermediate files in `cache/` are kept so you can re-run either step independently. Delete them any time to free disk space.

---

## Performance

Tested on RTX 4060 (8 GB VRAM), 25-minute meeting:

| Step | Time |
|------|------|
| Audio conversion (ffmpeg) | ~10s |
| Whisper large-v3 | ~8 min |
| pyannote diarization | ~12 min |

On CPU, expect 5–10× longer. Use `WHISPER_MODEL = "medium"` for a faster / lower-VRAM option.

---

## FAQ

**Can I use this without speaker diarization?**
Yes. Pass `--transcribe-only` or leave `HF_TOKEN` empty to skip diarization.

**Which model should I use?**
`large-v3` for best quality. `medium` is a good balance for weaker hardware. `small` / `base` for speed.

**The detected language is wrong.**
Set `LANGUAGE = "zh"` (or your language) in `config.py`, or pass `--language zh` on the command line.

**Can I point `WATCH_DIR` at a different folder?**
Yes. Edit `WATCH_DIR` in `config.py` to any absolute path.

---

## License

MIT
