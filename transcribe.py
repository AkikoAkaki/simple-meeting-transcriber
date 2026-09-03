#!/usr/bin/env python3
"""
simple-video-transcriber — transcribe.py
Transcribe a video or audio file with speaker labels.

Usage:
  python transcribe.py <video>                    # full pipeline
  python transcribe.py <video> --language en      # force language
  python transcribe.py <video> --transcribe-only  # skip diarization
  python transcribe.py <video> --diarize-only     # re-run diarization only

Output: transcripts/<filename>.md
"""

import json
import hashlib
import os
import subprocess
import sys
import argparse
from pathlib import Path

# Force UTF-8 stdout so Unicode characters (checkmarks, arrows, em-dashes)
# don't crash on Windows systems with GBK/CP936 console encoding.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import warnings
import logging
import threading
import time

# Suppress verbose/cosmetic warnings from pyannote and huggingface_hub
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=r".*std\(\).*degrees of freedom.*")
warnings.filterwarnings("ignore", message=".*resume_download.*")

import config
from paths import source_fingerprint, transcript_path


CACHE_SCHEMA_VERSION = 2
DIARIZATION_PIPELINE_ID = "pyannote/speaker-diarization-3.1"


def _cache_key(kind: str, **params) -> str:
    """Return a stable key for a stage's inputs and quality-affecting options."""
    payload = {"kind": kind, "schema_version": CACHE_SCHEMA_VERSION, **params}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _read_stage_payload(path: Path, kind: str, cache_key: str | None) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"[WARN] {kind} cache is corrupt — re-running...", flush=True)
        try:
            path.unlink()
        except OSError:
            pass
        return None

    # Keep direct callers and old tests compatible. Production calls always
    # supply a key, which intentionally invalidates legacy list-only caches.
    if cache_key is None and isinstance(payload, list):
        return {"result": payload}
    if not isinstance(payload, dict):
        return None
    if (payload.get("schema_version") != CACHE_SCHEMA_VERSION or
            payload.get("kind") != kind or
            cache_key is not None and payload.get("cache_key") != cache_key):
        return None
    result = payload.get("result")
    if kind == "whisper" and result == []:
        # An empty result usually means the audio was transiently silent or
        # corrupt; never let it poison the cache for this key.
        return None
    return payload if isinstance(result, list) else None


def _read_stage_cache(path: Path, kind: str, cache_key: str | None) -> list[dict] | None:
    payload = _read_stage_payload(path, kind, cache_key)
    return payload.get("result") if payload is not None else None


def _write_stage_cache(path: Path, kind: str, cache_key: str, result: list[dict],
                      metadata: dict | None = None) -> None:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": kind,
        "cache_key": cache_key,
        "result": result,
    }
    if metadata:
        payload.update(metadata)
    partial = path.with_name(f"{path.name}.part")
    partial.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    partial.replace(path)


def _whisper_cache_key(model_name: str, device: str, language: str | None,
                       hotwords: str | None, word_timestamps: bool) -> str:
    compute_type = "float16" if device == "cuda" else "int8"
    return _cache_key(
        "whisper", model=model_name, device=device, compute_type=compute_type,
        language=language or "auto", hotwords=hotwords or "",
        word_timestamps=bool(word_timestamps),
    )


def _diarization_cache_key(max_speakers: int | None, num_speakers: int | None) -> str:
    return _cache_key(
        "diarization", pipeline=DIARIZATION_PIPELINE_ID,
        max_speakers=max_speakers, num_speakers=num_speakers,
    )


def _normalize_speaker_count(value) -> int | None:
    if value is None or str(value).strip().lower() in {"", "auto", "none"}:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid speaker count: {value}")
    if parsed < 1:
        raise ValueError(f"Speaker count must be positive: {value}")
    return parsed


# Machine-readable events are prefixed so the background controller can parse
# them without breaking the existing human-readable CLI output.
_EVENT_LOCK = threading.Lock()


class TranscriptionError(RuntimeError):
    """A job-level failure that must not terminate the persistent worker."""


class DiarizationDeviceError(RuntimeError):
    """The cached diarization model could not be placed on a device."""


def _emit_event(event: str, **payload) -> None:
    job_id = os.environ.get("TRANSCRIBE_JOB_ID")
    if job_id and "job_id" not in payload:
        payload["job_id"] = job_id
    record = {"event": event, "timestamp": time.time(), **payload}
    with _EVENT_LOCK:
        print("@@EVENT " + json.dumps(record, ensure_ascii=False), flush=True)


def _heartbeat(stage: str, message: str, interval: float = 10.0):
    stop = threading.Event()
    started = time.monotonic()

    def _run():
        while not stop.wait(interval):
            _emit_event("heartbeat", stage=stage,
                        elapsed_sec=round(time.monotonic() - started, 1),
                        message=message)

    thread = threading.Thread(target=_run, name=f"{stage}-heartbeat", daemon=True)
    thread.start()
    return stop, thread

# ── Compatibility patches ─────────────────────────────────────────────────────
# huggingface_hub ≥1.0 dropped use_auth_token
import huggingface_hub.file_download as _hf_dl
_orig_download = _hf_dl.hf_hub_download
def _patched_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _orig_download(*args, **kwargs)
_hf_dl.hf_hub_download = _patched_download

import contextlib as _contextlib

@_contextlib.contextmanager
def _allow_unsafe_torch_load():
    """Temporarily allow weights_only=False — needed for pyannote checkpoints only."""
    import torch
    import functools
    orig = torch.load
    @functools.wraps(orig)
    def _unsafe(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig(*args, **kwargs)
    torch.load = _unsafe
    try:
        yield
    finally:
        torch.load = orig
# ─────────────────────────────────────────────────────────────────────────────


def get_wav_duration(wav_path: Path) -> float:
    """Read WAV file header directly to calculate duration without loading torchaudio."""
    import struct
    try:
        with open(wav_path, "rb") as f:
            riff_header = f.read(12)
            if riff_header[:4] != b"RIFF" or riff_header[8:12] != b"WAVE":
                return 0.0

            sample_rate = 16000
            channels = 1
            bits_per_sample = 16
            data_size = 0

            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break
                chunk_id = chunk_header[:4]
                chunk_len = struct.unpack("<I", chunk_header[4:8])[0]

                if chunk_id == b"fmt ":
                    fmt_data = f.read(chunk_len)
                    if len(fmt_data) >= 16:
                        channels = struct.unpack("<H", fmt_data[2:4])[0]
                        sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                        bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
                    if chunk_len & 1:
                        f.seek(1, 1)
                elif chunk_id == b"data":
                    data_size = chunk_len
                    break
                else:
                    f.seek(chunk_len + (chunk_len & 1), 1)

            bytes_per_second = sample_rate * channels * (bits_per_sample // 8)
            if bytes_per_second > 0:
                return data_size / bytes_per_second
    except Exception:
        pass
    return 0.0


_whisper_model = None
_whisper_model_params = None

def get_whisper_model(model_name: str, device: str, compute_type: str):
    global _whisper_model, _whisper_model_params
    params = (model_name, device, compute_type)
    if _whisper_model is None or _whisper_model_params != params:
        _whisper_model = None
        import gc
        gc.collect()
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
        _whisper_model_params = params
    return _whisper_model


def clear_whisper_model() -> None:
    global _whisper_model, _whisper_model_params
    had_model = _whisper_model is not None
    _whisper_model = None
    _whisper_model_params = None
    if had_model:
        import gc
        gc.collect()


_diarize_pipeline = None
_diarize_token = None
_diarize_device = None

def get_diarize_pipeline(token: str, device: str):
    global _diarize_pipeline, _diarize_token, _diarize_device
    if _diarize_pipeline is None or _diarize_token != token:
        _diarize_pipeline = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from pyannote.audio import Pipeline
        with _allow_unsafe_torch_load():
            try:
                _diarize_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", token=token)
            except TypeError:
                _diarize_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", use_auth_token=token)
        _diarize_token = token
        _diarize_device = None
    if _diarize_device != device:
        import torch
        try:
            _diarize_pipeline.to(torch.device(device))
        except RuntimeError as exc:
            _diarize_device = None
            raise DiarizationDeviceError(str(exc)) from exc
        _diarize_device = device
    return _diarize_pipeline


def clear_diarize_pipeline() -> None:
    global _diarize_pipeline, _diarize_token, _diarize_device
    had_pipeline = _diarize_pipeline is not None
    _diarize_pipeline = None
    _diarize_token = None
    _diarize_device = None
    if not had_pipeline:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass



def _resolve_device() -> str:
    if config.DEVICE == "auto":
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda"
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return config.DEVICE


def derive_paths(input_path: Path) -> dict[str, Path]:
    stem = input_path.stem
    path_hash = source_fingerprint(input_path)

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "wav":          config.CACHE_DIR      / f"{stem}_{path_hash}_16k.wav",
        "whisper_json": config.CACHE_DIR      / f"_{stem}_{path_hash}_whisper.json",
        "diarize_json": config.CACHE_DIR      / f"_{stem}_{path_hash}_diarize.json",
        "segments_json": config.CACHE_DIR    / f"_{stem}_{path_hash}_segments.json",
        "output_md":    transcript_path(input_path, config.TRANSCRIPT_DIR, "md"),
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
    print("       To enable: save a token in the tray dashboard (or set HF_TOKEN env var)", flush=True)
    print("       Accept model terms at: https://hf.co/pyannote/speaker-diarization-3.1", flush=True)
    return ""


# ── Step 1: Audio conversion ──────────────────────────────────────────────────

def convert_to_wav(input_path: Path, output_path: Path):
    if output_path.exists():
        if output_path.stat().st_size < 1024:
            output_path.unlink()
        else:
            _emit_event("stage", stage="converting", progress=1.0,
                        message="Using cached 16 kHz WAV")
            print(f"[1/4] WAV cache found: {output_path.name}", flush=True)
            return
    partial_path = output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")
    try:
        partial_path.unlink(missing_ok=True)
    except OSError:
        pass
    _emit_event("stage", stage="converting", progress=0.0,
                message="Converting audio to 16 kHz mono WAV")
    print(f"[1/4] Converting audio → 16kHz mono WAV...", flush=True)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), "-ac", "1", "-ar", "16000", str(partial_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        print("ERROR: ffmpeg not found in PATH.", flush=True)
        print("       Install ffmpeg: https://ffmpeg.org/download.html", flush=True)
        raise TranscriptionError("ffmpeg not found in PATH")
    if result.returncode != 0:
        partial_path.unlink(missing_ok=True)
        print(f"ERROR: ffmpeg failed (exit {result.returncode}):", flush=True)
        for line in result.stderr.splitlines()[-20:]:
            print(f"       {line}", flush=True)
        raise TranscriptionError(f"ffmpeg failed with exit code {result.returncode}")
    if not partial_path.is_file() or partial_path.stat().st_size < 1024:
        partial_path.unlink(missing_ok=True)
        print("ERROR: ffmpeg produced no usable WAV output.", flush=True)
        raise TranscriptionError("ffmpeg produced no usable WAV output")
    partial_path.replace(output_path)
    print(f"[1/4] Done — {output_path.stat().st_size / 1024 / 1024:.1f} MB", flush=True)
    _emit_event("stage", stage="converting", progress=1.0,
                message="Audio conversion completed")


# ── Step 2: Whisper transcription ─────────────────────────────────────────────

def run_whisper(wav_path: Path, whisper_json: Path, language: str | None,
                hotwords: str | None = None, cache_key: str | None = None) -> list[dict]:
    if whisper_json.exists():
        _emit_event("stage", stage="transcribing", progress=1.0,
                    message="Loading cached Whisper result")
        print("[2/4] Loading cached Whisper result...", flush=True)
        segs = _read_stage_cache(whisper_json, "whisper", cache_key)
        if segs is not None:
            print(f"      {len(segs)} segments from cache", flush=True)
            return segs
        print("[INFO] Whisper cache does not match current settings — re-transcribing...", flush=True)

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    lang_display = language or "auto-detect"
    MODEL_SIZES = {"tiny": "~75 MB", "base": "~145 MB", "small": "~466 MB",
                   "medium": "~1.5 GB", "large-v3": "~3.1 GB"}
    size_hint = MODEL_SIZES.get(config.WHISPER_MODEL, "")
    print(f"[2/4] Loading Whisper {config.WHISPER_MODEL} on {device} ({compute_type})...", flush=True)
    _emit_event("stage", stage="loading_whisper", progress=0.0,
                message=f"Loading Whisper {config.WHISPER_MODEL} on {device}")
    print(f"      (First run: downloading {config.WHISPER_MODEL} {size_hint} — please wait)", flush=True)

    try:
        model = get_whisper_model(config.WHISPER_MODEL, device, compute_type)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("ERROR: GPU out of memory loading Whisper model.", flush=True)
            print(f"       Try --model medium or --device cpu", flush=True)
        else:
            print(f"ERROR: Failed to load Whisper model: {e}", flush=True)
        raise TranscriptionError(f"Failed to load Whisper model: {e}") from e

    print(f"      Model loaded. Transcribing [{lang_display}]...", flush=True)
    _emit_event("stage", stage="transcribing", progress=0.0,
                message=f"Transcribing [{lang_display}]")
    seg_iter, info = model.transcribe(
        str(wav_path),
        language=language,
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=2000),
        hotwords=hotwords or None,
    )

    segments = []
    last_progress = -1.0
    last_progress_emit = 0.0
    for s in seg_iter:
        if not s.text.strip():
            continue
        segment = {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        words = []
        for word in getattr(s, "words", None) or []:
            word_text = str(getattr(word, "word", ""))
            word_start = getattr(word, "start", None)
            word_end = getattr(word, "end", None)
            if word_text.strip() and word_start is not None and word_end is not None:
                words.append({
                    "start": round(float(word_start), 2),
                    "end": round(float(word_end), 2),
                    "word": word_text,
                })
        if words:
            segment["words"] = words
        segments.append(segment)
        duration = getattr(info, "duration", None)
        progress = None
        if duration and duration > 0:
            progress = min(max(float(s.end) / float(duration), 0.0), 0.99)
        now = time.monotonic()
        if (progress is not None and
                (progress >= 1.0 or progress - last_progress >= 0.01)) or now - last_progress_emit >= 1.0:
            _emit_event("progress", stage="transcribing", progress=progress,
                        segments=len(segments), audio_position_sec=round(float(s.end), 2),
                        message=f"Transcribed through {format_time(s.end)}")
            last_progress = progress if progress is not None else last_progress
            last_progress_emit = now
        if len(segments) % 20 == 0:
            print(f"      ... {len(segments)} segments, up to {format_time(s.end)}", flush=True)

    if segments:
        _write_stage_cache(whisper_json, "whisper", cache_key or "legacy", segments)
    print(f"      Done — {len(segments)} segments | detected: {info.language} ({info.language_probability:.0%})", flush=True)
    _emit_event("stage", stage="transcribing", progress=1.0,
                segments=len(segments), message="Whisper transcription completed")
    clear_whisper_model()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return segments


# ── Step 3: Speaker diarization (pyannote) ────────────────────────────────────

def run_diarization(wav_path: Path, diarize_json: Path, token: str | None = None,
                    num_speakers: int | None = None,
                    max_speakers: int | None = None,
                    cache_key: str | None = None) -> list[dict]:
    import torch

    if cache_key is None:
        cache_key = _diarization_cache_key(max_speakers, num_speakers)

    if diarize_json.exists():
        _emit_event("stage", stage="diarizing", progress=1.0,
                    message="Loading cached diarization result")
        print("[3/4] Loading cached diarization result...", flush=True)
        turns = _read_stage_cache(diarize_json, "diarization", cache_key)
        if turns is not None:
            print(f"      {len(turns)} turns from cache", flush=True)
            return turns
        print("[INFO] Diarization cache does not match current settings — re-running...", flush=True)

    if token is None:
        token = get_hf_token()
    if not token:
        _emit_event("warning", stage="diarizing",
                    message="Speaker diarization disabled: no HuggingFace token")
        print("[3/4] Skipping diarization (no HF token)", flush=True)
        return []

    _emit_event("stage", stage="loading_diarization", progress=0.0,
                message="Loading speaker diarization model")
    print(f"[3/4] Loading {DIARIZATION_PIPELINE_ID}...", flush=True)
    print("      (First run: downloading pyannote models ~500 MB — please wait)", flush=True)
    device = _resolve_device()
    try:
        try:
            pipeline = get_diarize_pipeline(token, device)
        except DiarizationDeviceError as e:
            if device != "cuda":
                raise
            print(f"      GPU move failed ({e}), falling back to CPU", flush=True)
            device = "cpu"
            pipeline = get_diarize_pipeline(token, device)
    except Exception as e:
        err = str(e)
        if any(k in err for k in ("401", "403", "gated", "unauthorized", "PermissionError")):
            print("ERROR: HuggingFace access denied. Check that:", flush=True)
            print("  1. HF_TOKEN is valid — https://hf.co/settings/tokens", flush=True)
            print("  2. Model terms accepted — https://hf.co/pyannote/speaker-diarization-3.1", flush=True)
            print("  3. Model terms accepted — https://hf.co/pyannote/segmentation-3.0", flush=True)
        else:
            print(f"ERROR: Failed to load diarization model: {e}", flush=True)
        _emit_event("warning", stage="loading_diarization",
                    message="Speaker diarization model could not be loaded", error=err)
        return []

    if device == "cuda":
        print("      Diarization pipeline moved to GPU", flush=True)

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
        print(f"      num_speakers={num_speakers}", flush=True)
    elif max_speakers:
        kwargs["max_speakers"] = max_speakers
        print(f"      max_speakers={max_speakers}", flush=True)

    print(f"      Running diarization on {device} — may take 10–25 min...", flush=True)
    _emit_event("stage", stage="diarizing", progress=0.0,
                message=f"Running speaker diarization on {device}")
    heartbeat_stop, heartbeat_thread = _heartbeat(
        "diarizing", "Speaker diarization is still running")
    try:
        diarization = pipeline(str(wav_path), **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("ERROR: GPU out of memory during diarization.", flush=True)
            print("       Retry with --device cpu", flush=True)
        else:
            print(f"ERROR: Diarization failed: {e}", flush=True)
        _emit_event("warning", stage="diarizing", message="Speaker diarization failed", error=str(e))
        return []
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1)

    turns = []
    for t, _, spk in diarization.itertracks(yield_label=True):
        spk_str = str(spk)
        speaker_label = spk_str if spk_str.startswith("SPEAKER_") else f"SPEAKER_{spk_str}"
        turns.append({"start": round(t.start, 2), "end": round(t.end, 2), "speaker": speaker_label})
    _write_stage_cache(diarize_json, "diarization", cache_key or "legacy", turns)
    print(f"      Done — {len(turns)} speaker turns identified", flush=True)
    _emit_event("stage", stage="diarizing", progress=1.0,
                turns=len(turns), message="Speaker diarization completed")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return turns


# ── Step 4: Merge & output ────────────────────────────────────────────────────

def _overlap_ratio(a_start, a_end, b_start, b_end) -> float:
    start, end = max(a_start, b_start), min(a_end, b_end)
    dur = a_end - a_start
    return max(0.0, (end - start) / dur) if dur > 0 else 0.0


def _speaker_for_interval(start: float, end: float, speaker_turns: list[dict], turn_idx: int):
    """Choose the speaker with the greatest overlap for one word interval."""
    num_turns = len(speaker_turns)
    while turn_idx < num_turns and speaker_turns[turn_idx]["end"] <= start:
        turn_idx += 1

    best_speaker = "[unknown]"
    best_overlap = 0.0
    i = turn_idx
    while i < num_turns and speaker_turns[i]["start"] < end:
        turn = speaker_turns[i]
        overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]
        i += 1
    return best_speaker, turn_idx


def _append_word_text(current: str, word: str) -> str:
    """Append the token exactly as faster-whisper emitted it.

    Word strings carry meaningful leading whitespace for English and usually
    omit it for CJK text. Inventing a separator corrupts both cases and can
    even split one English word into two pieces (for example, ``integr`` +
    ``ation``).
    """
    if not current:
        return word.strip()
    return current + word


def _is_cjk_char(char: str) -> bool:
    return bool(char) and ("\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff")


def _join_display_text(current: str, addition: str) -> str:
    """Join adjacent display blocks without inventing CJK whitespace."""
    current = current.rstrip()
    addition = addition.lstrip()
    if not current:
        return addition
    if not addition:
        return current
    punctuation = ".,!?;:%，。！？；：、）)]}"
    if (addition[0] in punctuation or current[-1] in punctuation or
            _is_cjk_char(current[-1]) or _is_cjk_char(addition[0])):
        return current + addition
    return current + " " + addition


def _join_whisper_segment_text(current: str, addition: str) -> str:
    """Separate adjacent Whisper segments while keeping punctuation tight."""
    current = current.rstrip()
    addition = addition.lstrip()
    if not current:
        return addition
    if not addition:
        return current
    if (addition[0] in ".,!?;:%，。！？；：、）)]}" or
            current[-1] in "([{\"“‘"):
        return current + addition
    return current + " " + addition


def _coalesce_short_speaker_runs(segments: list[dict], max_duration: float = 1.0) -> list[dict]:
    """Suppress only high-confidence, short diarization flicker.

    A short unknown block is reassigned only when its known neighbours agree.
    A known speaker block is reassigned under the same condition only when its
    displayed text is at most four non-space characters. Boundary unknown
    blocks may borrow their single known neighbour. All other changes remain
    untouched so real speaker switches are preserved.
    """
    work = [dict(segment) for segment in segments]
    for index, segment in enumerate(work):
        duration = float(segment.get("end", 0.0)) - float(segment.get("start", 0.0))
        if duration > max_duration:
            continue
        previous = work[index - 1] if index > 0 else None
        following = work[index + 1] if index + 1 < len(work) else None
        previous_speaker = previous.get("speaker") if previous else None
        following_speaker = following.get("speaker") if following else None
        compact_chars = len("".join(str(segment.get("text", "")).split()))

        if segment.get("speaker") == "[unknown]":
            if previous_speaker and previous_speaker == following_speaker and previous_speaker != "[unknown]":
                segment["speaker"] = previous_speaker
            elif previous_speaker and not following and previous_speaker != "[unknown]":
                segment["speaker"] = previous_speaker
            elif following_speaker and not previous and following_speaker != "[unknown]":
                segment["speaker"] = following_speaker
        elif (compact_chars <= 4 and previous_speaker and
              previous_speaker == following_speaker and
              previous_speaker != segment.get("speaker") and
              previous_speaker != "[unknown]"):
            segment["speaker"] = previous_speaker

    coalesced = []
    for segment in work:
        if (coalesced and coalesced[-1].get("speaker") == segment.get("speaker") and
                float(segment["start"]) - float(coalesced[-1]["end"]) <= 2.0):
            coalesced[-1]["end"] = segment["end"]
            coalesced[-1]["text"] = _join_display_text(
                str(coalesced[-1].get("text", "")), str(segment.get("text", "")))
        else:
            coalesced.append(segment)
    return coalesced


def _merge_word_timestamps(whisper_segments: list[dict], speaker_turns: list[dict]) -> list[dict]:
    labeled_words = []
    turn_idx = 0
    for segment_index, segment in enumerate(whisper_segments):
        for word in segment.get("words", []):
            start = float(word["start"])
            end = float(word["end"])
            speaker, turn_idx = _speaker_for_interval(start, end, speaker_turns, turn_idx)
            labeled_words.append({
                "start": start,
                "end": end,
                "word": word["word"],
                "speaker": speaker,
                "segment_index": segment_index,
            })

    merged = []
    for word in labeled_words:
        if (merged and merged[-1]["speaker"] == word["speaker"] and
                word["start"] - merged[-1]["end"] <= 2.0):
            merged[-1]["end"] = word["end"]
            if merged[-1]["_source_segment"] == word["segment_index"]:
                merged[-1]["text"] = _append_word_text(merged[-1]["text"], word["word"])
            else:
                merged[-1]["text"] = _join_whisper_segment_text(
                    merged[-1]["text"], word["word"])
                merged[-1]["_source_segment"] = word["segment_index"]
        else:
            merged.append({
                "start": word["start"],
                "end": word["end"],
                "text": word["word"].strip(),
                "speaker": word["speaker"],
                "_source_segment": word["segment_index"],
            })
    for segment in merged:
        segment.pop("_source_segment", None)
    return merged


def merge_results(whisper_segments: list[dict], speaker_turns: list[dict]) -> list[dict]:
    _emit_event("stage", stage="merging", progress=0.0,
                message="Merging transcript and speaker labels")
    print("[4/4] Merging transcript and speaker labels...", flush=True)

    # Word-level timestamps let a single Whisper segment contain multiple
    # speakers without assigning the entire sentence to one person.
    # A mixed cache (some segments with word timestamps, some without) cannot
    # be safely reconstructed word-by-word without dropping the latter. Use
    # the segment-level path for that case; it preserves all source text.
    if (speaker_turns and whisper_segments and
            all(segment.get("words") for segment in whisper_segments)):
        merged = _merge_word_timestamps(whisper_segments, speaker_turns)
        merged = _coalesce_short_speaker_runs(merged)
        print(f"      {len(merged)} segments after word-level merge", flush=True)
        _emit_event("stage", stage="merging", progress=1.0,
                    segments=len(merged), message="Transcript merge completed")
        return merged

    labeled = []
    turn_idx = 0
    num_turns = len(speaker_turns)
    for seg in whisper_segments:
        speaker = "[unknown]"
        best = 0.0

        # Advance turn_idx to skip turns that end before this segment starts
        while turn_idx < num_turns and speaker_turns[turn_idx]["end"] <= seg["start"]:
            turn_idx += 1

        # Check all candidate turns that start before the segment ends
        i = turn_idx
        while i < num_turns and speaker_turns[i]["start"] < seg["end"]:
            turn = speaker_turns[i]
            ov = _overlap_ratio(seg["start"], seg["end"], turn["start"], turn["end"])
            if ov > best:
                best, speaker = ov, turn["speaker"]
            i += 1

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
    _emit_event("stage", stage="merging", progress=1.0,
                segments=len(merged), message="Transcript merge completed")
    return merged


def generate_markdown(segments: list[dict], source_file: str, total_sec: float,
                      has_diarization: bool, language: str | None) -> str:
    lang_str = language or "auto-detect"
    lines = [
        "# Transcript", "",
        f"**Source**: {source_file}",
        f"**Duration**: {format_time(total_sec)} ({int(total_sec)}s)",
        f"**Model**: Whisper {config.WHISPER_MODEL}  |  Language: {lang_str}",
        f"**Diarization**: {'pyannote/speaker-diarization-3.1' if has_diarization else 'disabled'}",
        "", "---", "",
    ]

    if has_diarization:
        from collections import defaultdict
        spk_stats = defaultdict(lambda: {"dur": 0.0, "count": 0})
        for s in segments:
            spk = s.get("speaker", "[unknown]")
            if spk != "[unknown]":
                spk_stats[spk]["dur"] += s["end"] - s["start"]
                spk_stats[spk]["count"] += 1
        speakers = sorted(spk_stats.keys())
        if speakers:
            lines += ["## Speakers", ""]
            for spk in speakers:
                dur = spk_stats[spk]["dur"]
                count = spk_stats[spk]["count"]
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


def generate_srt(segments: list[dict]) -> str:
    if not segments:
        return ""

    def _srt_ts(sec: float) -> str:
        h, rem = divmod(int(sec), 3600)
        m, s = divmod(rem, 60)
        ms = min(round((sec - int(sec)) * 1000), 999)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(
            f"{i}\n"
            f"{_srt_ts(seg['start'])} --> {_srt_ts(seg['end'])}\n"
            f"{seg['text'].strip()}"
        )
    return "\n\n".join(blocks)


def _clean_speaker_mapping(mapping: dict[str, str] | None) -> dict[str, str]:
    return {
        str(old): str(new).strip()
        for old, new in (mapping or {}).items()
        if str(new).strip()
    }


def rename_speakers_in_segments(segments: list[dict], mapping: dict[str, str]) -> list[dict]:
    """Return a renamed copy without changing the immutable stage caches."""
    clean_mapping = _clean_speaker_mapping(mapping)
    return [
        {**segment, "speaker": clean_mapping.get(segment.get("speaker"), segment.get("speaker"))}
        for segment in segments
    ]


def _speaker_aliases_for_segments(payload: dict | None,
                                  segments: list[dict]) -> dict[str, str]:
    """Keep only remembered names that still match this exact merged result."""
    aliases = _clean_speaker_mapping((payload or {}).get("speaker_aliases"))
    labels = {
        str(segment.get("speaker"))
        for segment in segments
        if segment.get("speaker") and segment.get("speaker") != "[unknown]"
    }
    return {label: name for label, name in aliases.items() if label in labels}


def _merged_cache_key(whisper_key: str, max_speakers: int | None,
                      num_speakers: int | None, has_diarization: bool) -> str:
    return _cache_key(
        "merged", whisper=whisper_key,
        diarization=_diarization_cache_key(max_speakers, num_speakers),
        has_diarization=bool(has_diarization),
    )


def _render_output(segments: list[dict], output_format: str, source_file: str,
                   total_sec: float, has_diarization: bool,
                   language: str | None) -> str:
    if output_format == "srt":
        return generate_srt(segments)
    if output_format == "txt":
        return "\n\n".join(segment["text"].strip() for segment in segments)
    return generate_markdown(
        segments, source_file, total_sec, has_diarization, language)


def rename_output(source_path: Path, output_path: Path, mapping: dict[str, str]) -> Path:
    """Rename speakers in a completed result and regenerate only its output file."""
    paths = derive_paths(Path(source_path).resolve())
    segments_path = paths.get("segments_json")
    if not segments_path or not segments_path.exists():
        raise FileNotFoundError(
            "No merged segment cache is available; run the transcription again first."
        )
    payload = _read_stage_payload(segments_path, "merged", None)
    if payload is None:
        raise ValueError("Merged segment cache is corrupt or from an unsupported version")

    aliases = _speaker_aliases_for_segments(payload, payload["result"])
    aliases.update(_clean_speaker_mapping(mapping))
    aliases = _speaker_aliases_for_segments(
        {"speaker_aliases": aliases}, payload["result"])
    segments = rename_speakers_in_segments(payload["result"], aliases)

    # Keep the canonical diarization labels in ``result`` and store display
    # names as metadata. This makes a later SRT/TXT/Markdown regeneration
    # reuse the user's names without changing the underlying model output.
    metadata = {
        key: value for key, value in payload.items()
        if key not in {"schema_version", "kind", "cache_key", "result"}
    }
    if aliases:
        metadata["speaker_aliases"] = aliases
    else:
        metadata.pop("speaker_aliases", None)
    _write_stage_cache(
        segments_path,
        "merged",
        str(payload.get("cache_key") or "legacy"),
        payload["result"],
        metadata=metadata,
    )
    output_format = output_path.suffix.lower().lstrip(".") or "md"
    if output_format not in {"md", "srt", "txt"}:
        output_format = "md"
    total_sec = float(payload.get("total_sec") or 0.0)
    if total_sec <= 0:
        total_sec = get_wav_duration(paths["wav"])
    if total_sec <= 0 and segments:
        total_sec = float(segments[-1]["end"])
    content = _render_output(
        segments,
        output_format,
        str(payload.get("source_file") or Path(source_path).name),
        total_sec,
        bool(payload.get("has_diarization")),
        payload.get("language"),
    )
    partial = output_path.with_name(f"{output_path.name}.part")
    partial.write_text(content, encoding="utf-8")
    partial.replace(output_path)
    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────

def run_server():
    print("SERVER_READY", flush=True)

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            data = json.loads(line.strip())
            command = data.get("command")
            if command == "transcribe":
                job_id = data.get("job_id", "")
                os.environ["TRANSCRIBE_JOB_ID"] = job_id
                try:
                    run_job_from_json(data)
                except Exception as e:
                    import traceback
                    traceback.print_exc(file=sys.stdout)
                    _emit_event("failed", stage="worker", message=str(e), error=str(e))
                finally:
                    os.environ.pop("TRANSCRIBE_JOB_ID", None)
            elif command == "ping":
                print("PONG", flush=True)
        except Exception as e:
            print(f"ERROR: Server loop error: {e}", file=sys.stderr, flush=True)


def run_job_from_json(data: dict):
    input_path = Path(data["input_path"]).resolve()
    output_dir = Path(data["output_dir"]).resolve()

    model_name = data.get("model") or config.WHISPER_MODEL
    device = data.get("device") or config.DEVICE
    max_speakers = _normalize_speaker_count(data.get("max_speakers"))
    num_speakers = _normalize_speaker_count(data.get("num_speakers"))
    language = data.get("language", config.LANGUAGE)
    if language == "auto":
        language = None

    transcribe_only = data.get("transcribe_only", False)
    diarize_only = data.get("diarize_only", False)
    output_format = data.get("output_format", "md")
    hotwords = data.get("hotwords", config.HOTWORDS) or ""
    token = data.get("token", "")

    # Apply settings
    config.WHISPER_MODEL = model_name
    config.DEVICE = device
    config.MAX_SPEAKERS = max_speakers
    config.NUM_SPEAKERS = num_speakers
    config.HOTWORDS = hotwords
    config.TRANSCRIPT_DIR = output_dir

    if not token:
        clear_diarize_pipeline()

    paths = derive_paths(input_path)
    convert_to_wav(input_path, paths["wav"])
    resolved_device = _resolve_device()
    whisper_key = _whisper_cache_key(
        model_name, resolved_device, language, hotwords, word_timestamps=True)

    if diarize_only:
        whisper_segments = _read_stage_cache(paths["whisper_json"], "whisper", whisper_key)
        if whisper_segments is None:
            raise FileNotFoundError(f"No cached Whisper result for {input_path.name}")
    else:
        whisper_segments = run_whisper(
            paths["wav"], paths["whisper_json"], language, hotwords, whisper_key)

    if not whisper_segments:
        raise ValueError("No speech detected in audio")

    if transcribe_only:
        speaker_turns = []
    else:
        if num_speakers is None:
            speaker_turns = run_diarization(
                paths["wav"], paths["diarize_json"], token, max_speakers=max_speakers)
        else:
            speaker_turns = run_diarization(
                paths["wav"], paths["diarize_json"], token, num_speakers, max_speakers)

    segments = merge_results(whisper_segments, speaker_turns)

    total_sec = get_wav_duration(paths["wav"])
    if total_sec <= 0 and segments:
        total_sec = segments[-1]["end"]

    segments_path = paths.get("segments_json")
    speaker_aliases = {}
    merged_key = _merged_cache_key(
        whisper_key, max_speakers, num_speakers, bool(speaker_turns))
    if segments_path:
        previous = (
            _read_stage_payload(segments_path, "merged", merged_key)
            if segments_path.exists() else None
        )
        speaker_aliases = _speaker_aliases_for_segments(previous, segments)
        metadata = {
            "source_file": input_path.name,
            "total_sec": total_sec,
            "language": language,
            "has_diarization": bool(speaker_turns),
        }
        if speaker_aliases:
            metadata["speaker_aliases"] = speaker_aliases
        _write_stage_cache(
            segments_path, "merged", merged_key, segments, metadata=metadata)

    fmt = output_format
    _emit_event("stage", stage="writing", progress=0.0, message=f"Writing {fmt} output")
    display_segments = rename_speakers_in_segments(segments, speaker_aliases)
    if fmt == "srt":
        content = generate_srt(display_segments)
        out_path = paths["output_md"].with_suffix(".srt")
    elif fmt == "txt":
        content = "\n\n".join(seg["text"].strip() for seg in display_segments)
        out_path = paths["output_md"].with_suffix(".txt")
    else:
        content = generate_markdown(display_segments, input_path.name, total_sec,
                                    bool(speaker_turns), language)
        out_path = paths["output_md"]

    out_path.write_text(content, encoding="utf-8")
    print(f"\n✓ Done → {out_path}", flush=True)
    _emit_event("completed", stage="completed", progress=1.0,
                output_path=str(out_path), diarization=bool(speaker_turns),
                message="Transcription completed")


def main():
    try:
        _main()
    except KeyboardInterrupt:
        _emit_event("cancelled", stage="worker", message="Interrupted")
        print("\nInterrupted.", flush=True)
        sys.exit(1)
    except Exception as e:
        _emit_event("failed", stage="worker", message=str(e), error=str(e))
        import traceback
        print(f"\nFATAL: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


def _main():
    parser = argparse.ArgumentParser(description="Transcribe a video or audio file with speaker labels")
    parser.add_argument("input", nargs="?", help="Path to video/audio file")
    parser.add_argument("--server", action="store_true", help="Run in persistent server mode")
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
    parser.add_argument("--num-speakers", default=None, type=int,
                        help="Exact number of speakers, when known")
    parser.add_argument("--hotwords", default=None,
                        help="Comma-separated names or technical terms to bias Whisper")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output .md file. Overrides config.py TRANSCRIPT_DIR")
    parser.add_argument("--output-format", choices=["md", "srt", "txt"], default="md",
                        help="Output format: md (Markdown), srt (subtitles), txt (plain text)")
    args = parser.parse_args()

    if args.server:
        run_server()
        return

    if not args.input:
        parser.error("the following arguments are required: input (unless running with --server)")

    _emit_event("started", stage="worker", job_id=os.environ.get("TRANSCRIBE_JOB_ID", ""),
                source=str(Path(args.input).resolve()), message="Transcription worker started")

    if args.model:
        config.WHISPER_MODEL = args.model
    if args.device:
        config.DEVICE = args.device
    if args.max_speakers is not None:
        config.MAX_SPEAKERS = args.max_speakers
    if args.num_speakers is not None:
        config.NUM_SPEAKERS = args.num_speakers
    if args.hotwords is not None:
        config.HOTWORDS = args.hotwords
    if args.output_dir:
        config.TRANSCRIPT_DIR = Path(args.output_dir)

    import platform
    print(f"Python {sys.version.split()[0]} | {platform.system()} {platform.release()}", flush=True)
    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count > 0:
            cuda_info = f"CUDA (device count: {cuda_count})"
        else:
            cuda_info = "CPU only (no CUDA)"
        print(f"ctranslate2 | {cuda_info}", flush=True)
    except Exception:
        pass

    language = args.language or config.LANGUAGE
    hotwords = config.HOTWORDS or ""
    input_path = Path(args.input).resolve()

    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}", flush=True)
        sys.exit(1)

    paths = derive_paths(input_path)

    convert_to_wav(input_path, paths["wav"])
    resolved_device = _resolve_device()
    whisper_key = _whisper_cache_key(
        config.WHISPER_MODEL, resolved_device, language, hotwords, word_timestamps=True)

    if args.diarize_only:
        whisper_segments = _read_stage_cache(paths["whisper_json"], "whisper", whisper_key)
        if whisper_segments is None:
            print(f"ERROR: no cached Whisper result for {input_path.name}", flush=True)
            print("       Run without --diarize-only first.", flush=True)
            sys.exit(1)
    else:
        whisper_segments = run_whisper(
            paths["wav"], paths["whisper_json"], language, hotwords, whisper_key)

    if not whisper_segments:
        _emit_event("failed", stage="transcribing", message="No speech detected in audio")
        print("ERROR: No speech detected in audio.", flush=True)
        print("  Possible causes:", flush=True)
        print("  1. Audio is silent or contains only music/noise (no speech)", flush=True)
        print("  2. WAV cache may be corrupted from a previous failed run.", flush=True)
        print(f"     Delete it and retry: {paths['wav']}", flush=True)
        print("  3. Wrong --language setting (try without it for auto-detect)", flush=True)
        print("  4. Source file is corrupted or has no audio track", flush=True)
        sys.exit(1)

    if args.transcribe_only:
        speaker_turns = []
    else:
        if config.NUM_SPEAKERS is None:
            speaker_turns = run_diarization(
                paths["wav"], paths["diarize_json"], max_speakers=config.MAX_SPEAKERS)
        else:
            speaker_turns = run_diarization(
                paths["wav"], paths["diarize_json"], None, config.NUM_SPEAKERS,
                config.MAX_SPEAKERS)

    segments = merge_results(whisper_segments, speaker_turns)

    total_sec = get_wav_duration(paths["wav"])
    if total_sec <= 0 and segments:
        total_sec = segments[-1]["end"]

    segments_path = paths.get("segments_json")
    speaker_aliases = {}
    merged_key = _merged_cache_key(
        whisper_key, config.MAX_SPEAKERS, config.NUM_SPEAKERS, bool(speaker_turns))
    if segments_path:
        previous = (
            _read_stage_payload(segments_path, "merged", merged_key)
            if segments_path.exists() else None
        )
        speaker_aliases = _speaker_aliases_for_segments(previous, segments)
        metadata = {
            "source_file": input_path.name,
            "total_sec": total_sec,
            "language": language,
            "has_diarization": bool(speaker_turns),
        }
        if speaker_aliases:
            metadata["speaker_aliases"] = speaker_aliases
        _write_stage_cache(
            segments_path, "merged", merged_key, segments, metadata=metadata)

    fmt = args.output_format
    _emit_event("stage", stage="writing", progress=0.0,
                message=f"Writing {fmt} output")
    display_segments = rename_speakers_in_segments(segments, speaker_aliases)
    if fmt == "srt":
        content = generate_srt(display_segments)
        out_path = paths["output_md"].with_suffix(".srt")
    elif fmt == "txt":
        content = "\n\n".join(seg["text"].strip() for seg in display_segments)
        out_path = paths["output_md"].with_suffix(".txt")
    else:
        content = generate_markdown(display_segments, input_path.name, total_sec,
                                    bool(speaker_turns), language)
        out_path = paths["output_md"]

    out_path.write_text(content, encoding="utf-8")
    _emit_event("completed", stage="completed", progress=1.0,
                output_path=str(out_path), diarization=bool(speaker_turns),
                message="Transcription completed")
    print(f"\n✓ Done → {out_path}", flush=True)
    print(f"  Cache files in {config.CACHE_DIR} can be deleted to free disk space.", flush=True)


if __name__ == "__main__":
    main()
