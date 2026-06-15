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
    # Priority: config → env var → hf_token.txt → interactive prompt
    if config.HF_TOKEN:
        return config.HF_TOKEN
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    token_file = Path(__file__).parent / "hf_token.txt"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            return token
    print("\n[INFO] HuggingFace token needed for speaker diarization.")
    print("  1. Get a free token at: https://hf.co/settings/tokens")
    print("  2. Accept terms at: https://hf.co/pyannote/speaker-diarization-3.1")
    print("  3. Accept terms at: https://hf.co/pyannote/segmentation-3.0")
    print("  Or set HF_TOKEN= in config.py to skip this prompt.\n")
    token = input("  Enter token (or press Enter to skip diarization): ").strip()
    if token:
        token_file.write_text(token)
        os.environ["HF_TOKEN"] = token
    return token


# ── Step 1: Audio conversion ──────────────────────────────────────────────────

def convert_to_wav(input_path: Path, output_path: Path):
    if output_path.exists():
        print(f"[1/4] WAV cache found: {output_path.name}")
        return
    print(f"[1/4] Converting audio → 16kHz mono WAV...")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-ac", "1", "-ar", "16000",
        "-progress", "pipe:1", "-nostats",
        str(output_path),
    ], check=True, capture_output=True)
    print(f"      Done ({output_path.stat().st_size / 1024 / 1024:.0f} MB)")


# ── Step 2: Whisper transcription ─────────────────────────────────────────────

def run_whisper(wav_path: Path, whisper_json: Path, language: str | None) -> list[dict]:
    from faster_whisper import WhisperModel

    if whisper_json.exists():
        print("[2/4] Loading cached Whisper result...")
        return json.loads(whisper_json.read_text(encoding="utf-8"))

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    lang_display = language or "auto-detect"
    print(f"[2/4] Loading Whisper {config.WHISPER_MODEL} on {device}...")
    print("      (First run downloads ~3 GB model weights)")

    model = WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute_type)

    print(f"      Transcribing [{lang_display}] — this may take 10–25 min...")
    seg_iter, info = model.transcribe(
        str(wav_path),
        language=language,
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=2000),
    )

    segments = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in seg_iter if s.text.strip()
    ]
    whisper_json.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      {len(segments)} segments (detected: {info.language}, p={info.language_probability:.0%})")
    return segments


# ── Step 3: Speaker diarization (pyannote) ────────────────────────────────────

def run_diarization(wav_path: Path, diarize_json: Path) -> list[dict]:
    import torch
    from pyannote.audio import Pipeline

    if diarize_json.exists():
        print("[3/4] Loading cached diarization result...")
        return json.loads(diarize_json.read_text(encoding="utf-8"))

    token = get_hf_token()
    if not token:
        print("[3/4] Skipping diarization (no token)")
        return []

    os.environ["HF_TOKEN"] = token
    print("[3/4] Loading pyannote diarization model...")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    except Exception as e:
        print(f"      Failed to load model: {e}")
        print("      Check token validity and that you accepted model terms.")
        return []

    device = _resolve_device()
    if device == "cuda":
        pipeline.to(torch.device("cuda"))

    kwargs = {}
    if config.MAX_SPEAKERS:
        kwargs["max_speakers"] = config.MAX_SPEAKERS

    print(f"      Diarizing on {device} — this may take 15–30 min...")
    diarization = pipeline(str(wav_path), **kwargs)

    turns = [
        {"start": round(t.start, 2), "end": round(t.end, 2), "speaker": f"SPEAKER_{spk}"}
        for t, _, spk in diarization.itertracks(yield_label=True)
    ]
    diarize_json.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      {len(turns)} speaker turns identified")
    return turns


# ── Step 4: Merge & output ────────────────────────────────────────────────────

def _overlap_ratio(a_start, a_end, b_start, b_end) -> float:
    start, end = max(a_start, b_start), min(a_end, b_end)
    dur = a_end - a_start
    return max(0.0, (end - start) / dur) if dur > 0 else 0.0


def merge_results(whisper_segments: list[dict], speaker_turns: list[dict]) -> list[dict]:
    print("[4/4] Merging transcript and speaker labels...")

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

    print(f"      {len(merged)} segments after merge")
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
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    paths = derive_paths(input_path)

    convert_to_wav(input_path, paths["wav"])

    if args.diarize_only:
        if not paths["whisper_json"].exists():
            print(f"Error: no cached Whisper result for {input_path.name}")
            print("Run without --diarize-only first.")
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

    print(f"\n✓ Done → {paths['output_md']}")
    print(f"  Cache files in {config.CACHE_DIR} can be deleted to free disk space.")


if __name__ == "__main__":
    main()
