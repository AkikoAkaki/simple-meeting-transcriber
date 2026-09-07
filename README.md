# Meeting Transcriber

A personal Windows tool that turns OBS meeting recordings, iPhone Voice Memos, and other audio/video into Markdown with timestamps and speaker labels. Transcripts are kept for reference and agent context. Recognition runs locally; initial model downloads require internet access. The tool does not summarize, rewrite, or infer speaker names.

[中文说明](README.zh.md)

## Use

- **Manual files:** run `start.bat` or click the tray icon, select or drop a file in the dashboard, and click **Transcribe**.
- **OBS recordings:** select the recording folder and enable **Watch in background**. The watcher handles new files appearing while it is running, not an existing archive. Disable it when unnecessary.
- **Agents / CLI:** call `transcribe.py`. It uses the same pipeline as the dashboard. Success exits with code 0; failure exits with a nonzero code.

**Full pipeline** includes speaker diarization. Save a HuggingFace token in the dashboard and accept the terms for [speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) and [segmentation-3.0](https://hf.co/pyannote/segmentation-3.0). Without a token or valid diarization cache, or if loading or diarization fails or returns no speaker turns, the task fails and does not publish a new Markdown transcript.

Choose **Transcribe only** explicitly for recordings known to contain one speaker when labels are unnecessary. Use **Full pipeline** for lectures with student interaction, seminars, and meetings. Leave speaker counts automatic unless known; **Exact speakers** takes precedence over **Max speakers**. **Names / terms** provides recognition hints, not verified speaker identities.

## Install and start

Requires Windows, Python 3.10+, and ffmpeg. `install.bat` checks Python, attempts to install missing ffmpeg through winget, and installs `requirements.txt` into `.venv`. Then run `start.bat`.

To start the tray at login:

```powershell
powershell -ExecutionPolicy Bypass -File setup_autostart.ps1
```

Append `-Uninstall` to remove autostart. This script manages autostart; `config.py` does not.

## Command line

Use the Python environment containing the dependencies, such as `.venv\Scripts\python.exe` after running the installer:

```powershell
python transcribe.py "lecture.m4a" --language en
python transcribe.py "meeting.mp4" --num-speakers 3 --hotwords "vLLM,KV Cache"
python transcribe.py "solo-lecture.m4a" --transcribe-only
python transcribe.py "meeting.mp4" --diarize-only
```

`--diarize-only` reuses Whisper results matching the current recognition settings; matching diarization results are also reused. Keep recognition caches when retrying failures. Overrides include `--model large-v3`, `--device cpu`, `--max-speakers 3`, and `--output-dir ./transcripts`. CLI token sources include the dashboard store, the `HF_TOKEN` environment variable, and `config.py`.

## Files and settings

- `transcripts/`: Markdown containing the source filename, timestamps, and speaker IDs. Source fingerprints distinguish files with the same name and replaced source contents.
- `cache/`: intermediate audio and recognition, diarization, and merged results. **Clear Audio Cache** removes only audio associated with terminal jobs, preserving JSON and Markdown.
- `%LOCALAPPDATA%/SimpleVideoTranscriber/`: dashboard settings, the job database, and text logs. Existing historical event tables remain intact but receive no new events.

`config.py` supplies defaults. The tray uses settings saved by the dashboard; CLI arguments override the corresponding defaults. The default model is `large-v3-turbo`, with `large-v3` also available; the default device is `auto`. The watch folder defaults to the user's Videos directory and can be changed in the dashboard or defaulted through `MEETING_TRANSCRIBER_WATCH_DIR`.

The watcher waits for roughly 15 seconds of stable file size and checks whether the file can be opened. This is a recording-completion heuristic; paused recordings or transfers can trigger early. Source audio/video is neither moved nor deleted, and media links are not added automatically.
