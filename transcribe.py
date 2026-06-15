#!/usr/bin/env python3
"""
meeting-transcriber — transcribe.py
Transcribe a meeting recording with speaker labels.

Usage:
  python transcribe.py <video>                    # full pipeline
  python transcribe.py <video> --language en      # force language
  python transcribe.py <video> --transcribe-only  # skip diarization
  python transcribe.py <video> --diarize-only     # re-run diarization only

Output: transcripts/<filename>.md
"""

import json
import os
import subprocess
import sys
import argparse
from pathlib import Path

import config

# ── Compatibility patches ─────────────────────────────────────────────────────
# huggingface_hub ≥1.0 dropped use_auth_token
import huggingface_hub.file_download as _hf_dl
_orig_download = _hf_dl.hf_hub_download
def _patched_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _orig_download(*args, **kwargs)
_hf_dl.hf_hub_download = _patched_download

# PyTorch ≥2.6 tightened weights_only; pyannote checkpoints need False
import torch as _torch, functools as _functools
_orig_load = _torch.load
@_functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_device() -> str:
    import torch
    if config.DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config.DEVICE


def derive_paths(input_path: Path) -> dict[str, Path]:
    stem = input_path.stem
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "wav":          config.CACHE_DIR      / f"{stem}_16k.wav",
        "whisper_json": config.CACHE_DIR      / f"_{stem}_whisper.json",
        "diarize_json": config.CACHE_DIR      / f"_{stem}_diarize.json",
        "output_md":    config.TRANSCRIPT_DIR / f"{stem}.md",
    }


def format_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_hf_token() -> str:
    if config.HF_TOKEN:
        return config.HF_TOKEN
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    token_file = Path(__file__).parent / "hf_token.txt"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            print(f"[INFO] Using HF token from hf_token.txt", flush=True)
            return token
    print("[INFO] No HuggingFace token found — skipping speaker diarization.", flush=True)
    print("       To enable: set HF_TOKEN in config.py, or create hf_token.txt", flush=True)
    print("       Accept model terms at: https://hf.co/pyannote/speaker-diarization-3.1", flush=True)
    return ""


# ── Step 1: Audio conversion ──────────────────────────────────────────────────

def convert_to_wav(input_path: Path, output_path: Path):
    if output_path.exists():
        print(f"[1/4] WAV cache found: {output_path.name}", flush=True)
        return
    print(f"[1/4] Converting audio → 16kHz mono WAV...", flush=True)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), "-ac", "1", "-ar", "16000", str(output_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        print("ERROR: ffmpeg not found in PATH.", flush=True)
        print("       Install ffmpeg: https://ffmpeg.org/download.html", flush=True)
        sys.exit(1)
    if result.returncode != 0:
        print(f"ERROR: ffmpeg failed (exit {result.returncode}):", flush=True)
        for line in result.stderr.splitlines()[-20:]:
            print(f"       {line}", flush=True)
        sys.exit(1)
    print(f"[1/4] Done — {output_path.stat().st_size / 1024 / 1024:.1f} MB", flush=True)


# ── Step 2: Whisper transcription ─────────────────────────────────────────────

def run_whisper(wav_path: Path, whisper_json: Path, language: str | None) -> list[dict]:
    from faster_whisper import WhisperModel

    if whisper_json.exists():
        print("[2/4] Loading cached Whisper result...", flush=True)
        segs = json.loads(whisper_json.read_text(encoding="utf-8"))
        print(f"      {len(segs)} segments from cache", flush=True)
        return segs

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    lang_display = language or "auto-detect"
    print(f"[2/4] Loading Whisper {config.WHISPER_MODEL} on {device} ({compute_type})...", flush=True)
    print(f"      (First run: model weights will be downloaded to HuggingFace cache)", flush=True)

    try:
        model = WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute_type)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("ERROR: GPU out of memory loading Whisper model.", flush=True)
            print(f"       Try --model medium or --device cpu", flush=True)
        else:
            print(f"ERROR: Failed to load Whisper model: {e}", flush=True)
        sys.exit(1)

    print(f"      Model loaded. Transcribing [{lang_display}]...", flush=True)
    seg_iter, info = model.transcribe(
        str(wav_path),
        language=language,
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=2000),
    )

    segments = []
    for s in seg_iter:
        if not s.text.strip():
            continue
        segments.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
        if len(segments) % 20 == 0:
            print(f"      ... {len(segments)} segments, up to {format_time(s.end)}", flush=True)

    whisper_json.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      Done — {len(segments)} segments | detected: {info.language} ({info.language_probability:.0%})", flush=True)
    return segments


# ── Step 3: Speaker diarization (pyannote) ────────────────────────────────────

def run_diarization(wav_path: Path, diarize_json: Path) -> list[dict]:
    import torch
    from pyannote.audio import Pipeline

    if diarize_json.exists():
        print("[3/4] Loading cached diarization result...", flush=True)
        turns = json.loads(diarize_json.read_text(encoding="utf-8"))
        print(f"      {len(turns)} turns from cache", flush=True)
        return turns

    token = get_hf_token()
    if not token:
        print("[3/4] Skipping diarization (no HF token)", flush=True)
        return []

    os.environ["HF_TOKEN"] = token
    print("[3/4] Loading pyannote/speaker-diarization-3.1...", flush=True)
    print("      (First run: model weights will be downloaded)", flush=True)
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    except Exception as e:
        err = str(e)
        if any(k in err for k in ("401", "403", "gated", "unauthorized", "PermissionError")):
            print("ERROR: HuggingFace access denied. Check that:", flush=True)
            print("  1. HF_TOKEN is valid — https://hf.co/settings/tokens", flush=True)
            print("  2. Model terms accepted — https://hf.co/pyannote/speaker-diarization-3.1", flush=True)
            print("  3. Model terms accepted — https://hf.co/pyannote/segmentation-3.0", flush=True)
        else:
            print(f"ERROR: Failed to load diarization model: {e}", flush=True)
        return []

    device = _resolve_device()
    if device == "cuda":
        try:
            pipeline.to(torch.device("cuda"))
            print("      Diarization pipeline moved to GPU", flush=True)
        except RuntimeError as e:
            print(f"      GPU move failed ({e}), falling back to CPU", flush=True)
            device = "cpu"

    kwargs = {}
    if config.MAX_SPEAKERS:
        kwargs["max_speakers"] = config.MAX_SPEAKERS
        print(f"      max_speakers={config.MAX_SPEAKERS}", flush=True)

    print(f"      Running diarization on {device} — may take 10–25 min...", flush=True)
    try:
        diarization = pipeline(str(wav_path), **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("ERROR: GPU out of memory during diarization.", flush=True)
            print("       Retry with --device cpu", flush=True)
        else:
            print(f"ERROR: Diarization failed: {e}", flush=True)
        return []

    turns = [
        {"start": round(t.start, 2), "end": round(t.end, 2), "speaker": f"SPEAKER_{spk}"}
        for t, _, spk in diarization.itertracks(yield_label=True)
    ]
    diarize_json.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      Done — {len(turns)} speaker turns identified", flush=True)
    return turns


# ── Step 4: Merge & output ────────────────────────────────────────────────────

def _overlap_ratio(a_start, a_end, b_start, b_end) -> float:
    start, end = max(a_start, b_start), min(a_end, b_end)
    dur = a_end - a_start
    return max(0.0, (end - start) / dur) if dur > 0 else 0.0


def merge_results(whisper_segments: list[dict], speaker_turns: list[dict]) -> list[dict]:
    print("[4/4] Merging transcript and speaker labels...", flush=True)

    labeled = []
    for seg in whisper_segments:
        speaker = "[unknown]"
        best = 0.0
        for turn in speaker_turns:
            ov = _overlap_ratio(seg["start"], seg["end"], turn["start"], turn["end"])
            if ov > best:
                best, speaker = ov, turn["speaker"]
        labeled.append({**seg, "speaker": speaker})

    # Merge consecutive segments from the same speaker (gap ≤ 2s)
    merged = []
    for seg in labeled:
        if merged and merged[-1]["speaker"] == seg["speaker"] and seg["start"] - merged[-1]["end"] <= 2.0:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(dict(seg))

    print(f"      {len(merged)} segments after merge", flush=True)
    return merged


def generate_markdown(segments: list[dict], source_file: str, total_sec: float,
                      has_diarization: bool, language: str | None) -> str:
    lang_str = language or "auto-detect"
    lines = [
        "# Meeting Transcript", "",
        f"**Source**: {source_file}",
        f"**Duration**: {format_time(total_sec)} ({int(total_sec)}s)",
        f"**Model**: Whisper {config.WHISPER_MODEL}  |  Language: {lang_str}",
        f"**Diarization**: {'pyannote/speaker-diarization-3.1' if has_diarization else 'disabled'}",
        "", "---", "",
    ]

    if has_diarization:
        speakers = sorted({s["speaker"] for s in segments if s["speaker"] != "[unknown]"})
        if speakers:
            lines += ["## Speakers", ""]
            for spk in speakers:
                dur = sum(s["end"] - s["start"] for s in segments if s["speaker"] == spk)
                count = sum(1 for s in segments if s["speaker"] == spk)
                lines.append(f"- **{spk}**: {format_time(dur)} ({count} segments)")
            lines += ["", "---", ""]

    lines.append("## Transcript")
    lines.append("")
    for seg in segments:
        lines.append(f"### [{format_time(seg['start'])} – {format_time(seg['end'])}] {seg['speaker']}")
        lines.append("")
        lines.append(seg["text"])
        lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        _main()
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\nFATAL: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


def _main():
    parser = argparse.ArgumentParser(description="Transcribe a meeting recording with speaker labels")
    parser.add_argument("input", help="Path to video/audio file")
    parser.add_argument("--language", default=None,
                        help="Language code (en/zh/ja/...). Default: auto-detect")
    parser.add_argument("--transcribe-only", action="store_true",
                        help="Run Whisper only, skip diarization (no HF token needed)")
    parser.add_argument("--diarize-only", action="store_true",
                        help="Re-run diarization using cached Whisper result")
    parser.add_argument("--model", default=None,
                        help="Whisper model size (tiny/base/small/medium/large-v3). Overrides config.py")
    parser.add_argument("--device", default=None,
                        help="Compute device (auto/cuda/cpu). Overrides config.py")
    parser.add_argument("--max-speakers", default=None, type=int,
                        help="Maximum number of speakers. Overrides config.py")
    args = parser.parse_args()

    if args.model:
        config.WHISPER_MODEL = args.model
    if args.device:
        config.DEVICE = args.device
    if args.max_speakers is not None:
        config.MAX_SPEAKERS = args.max_speakers

    language = args.language or config.LANGUAGE
    input_path = Path(args.input).resolve()

    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}", flush=True)
        sys.exit(1)

    paths = derive_paths(input_path)

    convert_to_wav(input_path, paths["wav"])

    if args.diarize_only:
        if not paths["whisper_json"].exists():
            print(f"ERROR: no cached Whisper result for {input_path.name}", flush=True)
            print("       Run without --diarize-only first.", flush=True)
            sys.exit(1)
        whisper_segments = json.loads(paths["whisper_json"].read_text(encoding="utf-8"))
    else:
        whisper_segments = run_whisper(paths["wav"], paths["whisper_json"], language)

    if args.transcribe_only:
        speaker_turns = []
    else:
        speaker_turns = run_diarization(paths["wav"], paths["diarize_json"])

    segments = merge_results(whisper_segments, speaker_turns)

    import torchaudio
    info = torchaudio.info(str(paths["wav"]))
    total_sec = info.num_frames / info.sample_rate

    md = generate_markdown(segments, input_path.name, total_sec, bool(speaker_turns), language)
    paths["output_md"].write_text(md, encoding="utf-8")

    print(f"\n✓ Done → {paths['output_md']}", flush=True)
    print(f"  Cache files in {config.CACHE_DIR} can be deleted to free disk space.", flush=True)


if __name__ == "__main__":
    main()
