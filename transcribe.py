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

# On Windows, ensure CTranslate2 can load cuBLAS / cuDNN from torch/lib to prevent silent CPU fallback
_TORCH_DLL_HANDLE = None
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    try:
        import importlib.util
        _spec = importlib.util.find_spec("torch")
        if _spec and _spec.submodule_search_locations:
            for _loc in _spec.submodule_search_locations:
                _cand = Path(_loc) / "lib"
                if _cand.is_dir():
                    _TORCH_DLL_HANDLE = os.add_dll_directory(str(_cand))
                    cand_str = str(_cand)
                    if cand_str not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = cand_str + os.pathsep + os.environ.get("PATH", "")
                    break
    except Exception:
        pass

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


_INITIAL_PROMPTS = {
    "zh": "以下是普通话的会议记录，包含完整的标点符号。",
    "en": "Here is a transcript of the meeting with complete punctuation.",
    "ja": "これは会議の書き起こしです。句読点を含めます。",
    "ko": "다음은 회의 녹취록이며 완전한 구두점이 포함되어 있습니다.",
}


def select_initial_prompt(language: str | None) -> str | None:
    """Return the Whisper initial_prompt for a language, or None for auto/other.

    Single selection point: None/auto -> None (no bias), zh/en/ja/ko ->
    the matching prompt, any other explicit language -> None (never inject
    a wrong-language prompt).
    """
    if language is None:
        return None
    normalized = str(language).strip().lower()
    if normalized in ("", "auto"):
        return None
    return _INITIAL_PROMPTS.get(normalized)


def _whisper_cache_key(model_name: str, device: str, language: str | None,
                       hotwords: str | None, word_timestamps: bool,
                       initial_prompt: str | None = None) -> str:
    compute_type = "float16" if device == "cuda" else "int8"
    if initial_prompt is None:
        try:
            initial_prompt = select_initial_prompt(language)
        except Exception:
            initial_prompt = None
    return _cache_key(
        "whisper", model=model_name, device=device, compute_type=compute_type,
        language=language or "auto", hotwords=hotwords or "",
        word_timestamps=bool(word_timestamps),
        initial_prompt=initial_prompt or "",
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


def _release_whisper_before_diarization() -> None:
    """Free Whisper (and its CUDA blocks) before loading pyannote.

    Owned by the pipeline orchestration layer so full pipelines never hold
    previous-task pyannote + current Whisper at the same time, while
    transcribe-only jobs can keep Whisper for reuse.
    """
    clear_whisper_model()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
        "output_md":    transcript_path(input_path, config.TRANSCRIPT_DIR),
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
    # Tray dashboard store (OS keyring with AppData fallback) is the
    # canonical source. Read it here so CLI runs see the same token.
    try:
        from service import TokenStore
        stored = TokenStore().get()
        if stored:
            return stored
    except Exception:
        pass
    print("[INFO] No HuggingFace token found.", flush=True)
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
    heartbeat_stop, heartbeat_thread = _heartbeat(
        "converting", "Audio conversion is still running")
    try:
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-y", "-i", str(input_path),
                    "-vn", "-sn", "-dn",
                    "-ac", "1", "-ar", "16000", str(partial_path),
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                encoding="utf-8", errors="replace",
            )
        except FileNotFoundError:
            print("ERROR: ffmpeg not found in PATH.", flush=True)
            print("       Install ffmpeg: https://ffmpeg.org/download.html", flush=True)
            raise TranscriptionError("ffmpeg not found in PATH")
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1)
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
    if language is None or str(language).strip().lower() in ("", "auto"):
        transcribe_language = None
        lang_display = "auto-detect"
    else:
        transcribe_language = language
        lang_display = language
    MODEL_SIZES = {"large-v3-turbo": "~1.6 GB", "large-v3": "~3.1 GB"}
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
            print(f"       Try --model large-v3-turbo or --device cpu", flush=True)
        else:
            print(f"ERROR: Failed to load Whisper model: {e}", flush=True)
        raise TranscriptionError(f"Failed to load Whisper model: {e}") from e

    print(f"      Model loaded. Transcribing [{lang_display}]...", flush=True)
    _emit_event("stage", stage="transcribing", progress=0.0,
                message=f"Transcribing [{lang_display}]")
    initial_prompt = select_initial_prompt(language)

    seg_iter = None
    info = None
    segments = []
    if True:
        seg_iter, info = model.transcribe(
            str(wav_path),
            language=transcribe_language,
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=2000),
            hotwords=hotwords or None,
            initial_prompt=initial_prompt,
        )

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
            preview_text = s.text.strip()
            if (progress is not None and
                    (progress >= 1.0 or progress - last_progress >= 0.01)) or now - last_progress_emit >= 1.0 or last_progress < 0:
                _emit_event("progress", stage="transcribing", progress=progress,
                            segments=len(segments), audio_position_sec=round(float(s.end), 2),
                            message=f"Transcribed through {format_time(s.end)}",
                            preview=preview_text)
                last_progress = progress if progress is not None else last_progress
                last_progress_emit = now
            if len(segments) % 20 == 0:
                print(f"      ... {len(segments)} segments, up to {format_time(s.end)}", flush=True)

        if segments:
            _write_stage_cache(whisper_json, "whisper", cache_key or "legacy", segments)
        print(f"      Done — {len(segments)} segments | detected: {info.language} ({info.language_probability:.0%})", flush=True)
        _emit_event("stage", stage="transcribing", progress=1.0,
                    segments=len(segments), message="Whisper transcription completed")
    # NOTE: model release is owned by the pipeline orchestration layer
    # (_run_job_impl) so transcribe-only jobs can reuse Whisper.
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
        if turns:
            print(f"      {len(turns)} turns from cache", flush=True)
            return turns
        print("[INFO] Diarization cache does not match current settings — re-running...", flush=True)

    if token is None:
        token = get_hf_token()
    if not token:
        raise TranscriptionError("Speaker diarization requires a HuggingFace token. "
                                 "Save one in the dashboard, or explicitly choose Transcribe only.")

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
        raise TranscriptionError(f"Speaker diarization model could not be loaded: {err}") from e

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
        raise TranscriptionError(f"Speaker diarization failed: {e}") from e
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1)

    turns = []
    for t, _, spk in diarization.itertracks(yield_label=True):
        spk_str = str(spk)
        speaker_label = spk_str if spk_str.startswith("SPEAKER_") else f"SPEAKER_{spk_str}"
        turns.append({"start": round(t.start, 2), "end": round(t.end, 2), "speaker": speaker_label})
    if not turns:
        raise TranscriptionError("Speaker diarization returned no speaker turns")
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
    """Return True if character is CJK ideograph, kana, hangul, or fullwidth/CJK punctuation."""
    if not char or len(char) != 1:
        return False
    cp = ord(char)
    return (
        # CJK Unified Ideographs & Extensions A-J
        (0x4E00 <= cp <= 0x9FFF) or
        (0x3400 <= cp <= 0x4DBF) or
        (0x20000 <= cp <= 0x2EBEF) or
        (0x30000 <= cp <= 0x323AF) or
        # CJK Compatibility Ideographs & Supplement
        (0xF900 <= cp <= 0xFAFF) or
        (0x2F800 <= cp <= 0x2FA1F) or
        # Japanese Hiragana, Katakana, and Extensions
        (0x3040 <= cp <= 0x309F) or
        (0x30A0 <= cp <= 0x30FF) or
        (0x31F0 <= cp <= 0x31FF) or
        (0x1B000 <= cp <= 0x1B12F) or
        (0x1B130 <= cp <= 0x1B16F) or
        (0x1AFF0 <= cp <= 0x1AFFF) or
        # Korean Hangul Syllables, Jamo, Compatibility Jamo, Extended A/B
        (0xAC00 <= cp <= 0xD7AF) or
        (0x1100 <= cp <= 0x11FF) or
        (0x3130 <= cp <= 0x318F) or
        (0xA960 <= cp <= 0xA97F) or
        (0xD7B0 <= cp <= 0xD7FF) or
        # Bopomofo (注音符号) & Bopomofo Extended
        (0x3100 <= cp <= 0x312F) or
        (0x31A0 <= cp <= 0x31BF) or
        # CJK Strokes & Kanbun
        (0x3190 <= cp <= 0x31EF) or
        # Enclosed CJK Letters and Months & CJK Compatibility
        (0x3200 <= cp <= 0x32FF) or
        (0x3300 <= cp <= 0x33FF) or
        # Yijing Hexagram Symbols
        (0x4DC0 <= cp <= 0x4DFF) or
        # CJK Symbols and Punctuation (e.g., 、。〈〉《》「」『』【】〔〕〖〗〜)
        (0x3000 <= cp <= 0x303F) or
        # Halfwidth and Fullwidth Forms
        (0xFF00 <= cp <= 0xFFEF) or
        # General Punctuation commonly used in CJK (e.g. — … ‘ ’ “ ”)
        (0x2010 <= cp <= 0x2027) or
        (0x2030 <= cp <= 0x205E)
    )


_ASCII_PUNCT = set(".,!?;:%'\"()[]{}-/\\_`~@#$^&*+=|<>")
_OPENING_BRACKETS = set("([{<“‘「『【《〈〔〖（")
_CLOSING_PUNCT = set(")]}>”’」』】》〉〕〗）.,!?;:%，。！？；：、")
_ASCII_QUOTES = set("\"'")


def _is_cjk_punct(char: str) -> bool:
    """Return True if character is a fullwidth or CJK punctuation mark."""
    if not char or len(char) != 1:
        return False
    cp = ord(char)
    return (
        (0x3000 <= cp <= 0x303F) or
        (0xFF01 <= cp <= 0xFF0F) or
        (0xFF1A <= cp <= 0xFF20) or
        (0xFF3B <= cp <= 0xFF40) or
        (0xFF5B <= cp <= 0xFF65) or
        (0xFFE0 <= cp <= 0xFFEE) or
        (0x2010 <= cp <= 0x2027) or
        (0x2030 <= cp <= 0x205E) or
        char in "，。！？；：、“”‘’《》（）【】〔〕…—～·"
    )


def _is_cjk_text(char: str) -> bool:
    """Return True if character is a CJK textual character (Hanzi, Kana, Hangul)."""
    return _is_cjk_char(char) and not _is_cjk_punct(char)


def _is_western_char(char: str) -> bool:
    """Return True if character is a Western/Latin alphanumeric or general non-CJK text character."""
    if not char or len(char) != 1:
        return False
    if _is_cjk_char(char):
        return False
    return char.isalnum() or (char.isascii() and char not in _ASCII_PUNCT and not char.isspace())


def _needs_space_between(
    left_char: str,
    right_char: str,
    left_is_closing: bool = True,
    right_is_opening: bool = True,
) -> bool:
    """Determine whether a space should be inserted between two characters.

    Rules:
    - CJK + CJK: No space (中+中不加空格)
    - CJK + Punctuation / Punctuation + CJK: No space (中+标点不加空格)
    - Punctuation + Punctuation: No space
    - CJK punct adjacent to any character: No space (fullwidth punctuation carries its own spacing)
    - Western + Western: Single space (英+英保留单空格)
    - CJK text + Western alnum / Western alnum + CJK text: Single space (中+英/英+中保留单空格)
    - Western alnum + Closing punct: No space (e.g. word.)
    - Opening punct / quote + Western alnum: No space (e.g. (word, "word)
    - Western alnum + Opening bracket / quote: Single space (e.g. word (, word ")
    - Western closing punct / quote + Western alnum: Single space (e.g. word, next, "word" next)
    - Western closing punct / quote + Opening bracket / quote: Single space (e.g. word. (Next))
    """
    if not left_char or not right_char:
        return False

    # Any CJK/fullwidth punctuation has built-in spacing; never add space adjacent to it
    if _is_cjk_punct(left_char) or _is_cjk_punct(right_char):
        return False

    # CJK text + CJK text: no space (中+中不加空格)
    if _is_cjk_text(left_char) and _is_cjk_text(right_char):
        return False

    # CJK text + punctuation / punctuation + CJK text: no space (中+标点不加空格)
    if _is_cjk_text(left_char) and (
        right_char in _CLOSING_PUNCT or right_char in _ASCII_PUNCT
        or right_char in _OPENING_BRACKETS or right_char in _ASCII_QUOTES
    ):
        return False
    if (
        _is_cjk_punct(left_char) or left_char in _OPENING_BRACKETS
        or left_char in _CLOSING_PUNCT or left_char in _ASCII_PUNCT
        or left_char in _ASCII_QUOTES
    ) and _is_cjk_text(right_char):
        return False

    # Western alnum + Closing punctuation (e.g. "word,") -> no space
    if _is_western_char(left_char) and (
        right_char in _CLOSING_PUNCT or (right_char in _ASCII_QUOTES and not right_is_opening)
    ):
        return False

    # Opening punctuation / quote + Western alnum (e.g. "(word", '"word') -> no space
    if (left_char in _OPENING_BRACKETS or (left_char in _ASCII_QUOTES and not left_is_closing)) and _is_western_char(right_char):
        return False

    # Western alnum + Opening bracket / quote (e.g. "word (", 'word "') -> single space
    if _is_western_char(left_char) and (
        right_char in _OPENING_BRACKETS or (right_char in _ASCII_QUOTES and right_is_opening)
    ):
        return True

    # Western closing punct / quote + Opening bracket / quote (e.g. "word. (Next)", '"Hello!" (whispered)') -> single space
    if (
        left_char in _CLOSING_PUNCT or (left_char in _ASCII_QUOTES and left_is_closing)
    ) and (
        right_char in _OPENING_BRACKETS or (right_char in _ASCII_QUOTES and right_is_opening)
    ):
        return True

    # Western closing punct / quote + Western alnum (e.g. "Hello," + "world", '"Hello,"' + "she") -> single space
    if (
        left_char in _CLOSING_PUNCT or (left_char in _ASCII_QUOTES and left_is_closing)
    ) and _is_western_char(right_char):
        return True

    # CJK text + Western alnum or Western alnum + CJK text -> single space (中+英/英+中保留单空格)
    if _is_cjk_text(left_char) and _is_western_char(right_char):
        return True
    if _is_western_char(left_char) and _is_cjk_text(right_char):
        return True

    # Western + Western -> single space (英+英保留单空格)
    if _is_western_char(left_char) and _is_western_char(right_char):
        return True

    return False


def _join_display_text(current: str, addition: str) -> str:
    """Join adjacent display blocks following Chinese/CJK and Western typography rules."""
    current = current.rstrip()
    addition = addition.lstrip()
    if not current:
        return addition
    if not addition:
        return current
    left_is_closing = len(current) > 1 or current not in _ASCII_QUOTES
    right_is_opening = len(addition) > 1 or addition not in _ASCII_QUOTES
    if _needs_space_between(current[-1], addition[0], left_is_closing, right_is_opening):
        return current + " " + addition
    return current + addition


def _join_whisper_segment_text(current: str, addition: str) -> str:
    """Separate adjacent Whisper segments following CJK and Western typography rules."""
    return _join_display_text(current, addition)


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
            merged[-1]["text"] = _join_display_text(merged[-1]["text"], seg["text"])
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


def _merged_cache_key(whisper_key: str, max_speakers: int | None,
                      num_speakers: int | None, has_diarization: bool) -> str:
    return _cache_key(
        "merged", whisper=whisper_key,
        diarization=_diarization_cache_key(max_speakers, num_speakers),
        has_diarization=bool(has_diarization),
    )


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


_SUPPORTED_DEVICES = frozenset({"auto", "cuda", "cpu"})


def run_job_from_json(data: dict):
    raw_input = data.get("input_path")
    if not raw_input:
        raise ValueError("Job is missing required field: input_path")
    raw_output = data.get("output_dir")
    if not raw_output:
        raise ValueError("Job is missing required field: output_dir")
    input_path = Path(raw_input).resolve()
    output_dir = Path(raw_output).resolve()

    model_name = data.get("model") or config.WHISPER_MODEL
    device = data.get("device") or config.DEVICE
    max_speakers = _normalize_speaker_count(data.get("max_speakers"))
    num_speakers = _normalize_speaker_count(data.get("num_speakers"))
    language = data.get("language", config.LANGUAGE)
    if language == "auto":
        language = None

    transcribe_only = data.get("transcribe_only", False)
    diarize_only = data.get("diarize_only", False)
    hotwords = data.get("hotwords", config.HOTWORDS) or ""
    token = data.get("token", "")

    if model_name not in config.SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name} "
            f"(expected one of: {', '.join(config.SUPPORTED_MODELS)})"
        )
    if device not in _SUPPORTED_DEVICES:
        raise ValueError(f"Unsupported device: {device} (expected one of: auto, cuda, cpu)")
    if transcribe_only and diarize_only:
        raise ValueError("Transcribe only and Re-diarize only are mutually exclusive")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Apply settings for this job only. Snapshot the globals first so the
    # next job falls back to clean defaults instead of this job's values.
    _saved_settings = (
        config.WHISPER_MODEL, config.DEVICE, config.MAX_SPEAKERS,
        config.NUM_SPEAKERS, config.HOTWORDS, config.TRANSCRIPT_DIR,
    )
    config.WHISPER_MODEL = model_name
    config.DEVICE = device
    config.MAX_SPEAKERS = max_speakers
    config.NUM_SPEAKERS = num_speakers
    config.HOTWORDS = hotwords
    config.TRANSCRIPT_DIR = output_dir
    try:
        _run_job_impl(
            input_path, model_name,
            max_speakers, num_speakers, language,
            transcribe_only, diarize_only, hotwords, token,
        )
    finally:
        (config.WHISPER_MODEL, config.DEVICE, config.MAX_SPEAKERS,
         config.NUM_SPEAKERS, config.HOTWORDS, config.TRANSCRIPT_DIR) = _saved_settings


def _run_job_impl(input_path, model_name,
                  max_speakers, num_speakers, language,
                  transcribe_only, diarize_only, hotwords, token):
    if not token:
        clear_diarize_pipeline()

    paths = derive_paths(input_path)
    convert_to_wav(input_path, paths["wav"])
    resolved_device = _resolve_device()
    initial_prompt = select_initial_prompt(language)
    whisper_key = _whisper_cache_key(
        model_name, resolved_device, language, hotwords, word_timestamps=True,
        initial_prompt=initial_prompt)

    if diarize_only:
        # Diarize-only never loads Whisper, only reuses its cache.
        whisper_segments = _read_stage_cache(paths["whisper_json"], "whisper", whisper_key)
        if whisper_segments is None:
            raise FileNotFoundError(f"No cached Whisper result for {input_path.name}")
    else:
        # Free the previous task's diarization pipeline before loading Whisper
        # so both models are never held on GPU at the same time.
        clear_diarize_pipeline()
        try:
            whisper_segments = run_whisper(
                paths["wav"], paths["whisper_json"], language, hotwords, whisper_key)
        except Exception:
            _release_whisper_before_diarization()
            raise

    if not whisper_segments:
        if not diarize_only and not transcribe_only:
            _release_whisper_before_diarization()
        raise ValueError("No speech detected in audio")

    if transcribe_only:
        # Keep Whisper loaded for compatible follow-up tasks.
        speaker_turns = []
    else:
        # Full and diarize-only need pyannote: ensure Whisper is off GPU first.
        _release_whisper_before_diarization()
        try:
            if num_speakers is None:
                speaker_turns = run_diarization(
                    paths["wav"], paths["diarize_json"], token, max_speakers=max_speakers)
            else:
                speaker_turns = run_diarization(
                    paths["wav"], paths["diarize_json"], token, num_speakers, max_speakers)
        except Exception:
            clear_diarize_pipeline()
            raise

    segments = merge_results(whisper_segments, speaker_turns)

    total_sec = get_wav_duration(paths["wav"])
    if total_sec <= 0 and segments:
        total_sec = segments[-1]["end"]

    segments_path = paths.get("segments_json")
    merged_key = _merged_cache_key(
        whisper_key, max_speakers, num_speakers, bool(speaker_turns))
    if segments_path:
        metadata = {
            "source_file": input_path.name,
            "total_sec": total_sec,
            "language": language,
            "has_diarization": bool(speaker_turns),
        }
        _write_stage_cache(
            segments_path, "merged", merged_key, segments, metadata=metadata)

    _emit_event("stage", stage="writing", progress=0.0, message="Writing md output")
    content = generate_markdown(segments, input_path.name, total_sec,
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


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe a video or audio file with speaker labels")
    parser.add_argument("input", nargs="?", help="Path to video/audio file")
    parser.add_argument("--server", action="store_true", help="Run in persistent server mode")
    parser.add_argument("--language", default=None,
                        help="Language code (en/zh/ja/...). Default: auto-detect")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--transcribe-only", action="store_true",
                        help="Run Whisper only, skip diarization (no HF token needed)")
    mode.add_argument("--diarize-only", action="store_true",
                        help="Re-run diarization using cached Whisper result")
    parser.add_argument("--model", default=None,
                        choices=list(config.SUPPORTED_MODELS),
                        help="Whisper model size (large-v3-turbo/large-v3). Overrides config.py")
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
    return parser


def _main():
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.server:
        run_server()
        return

    if not args.input:
        parser.error("the following arguments are required: input (unless running with --server)")

    _emit_event("started", stage="worker", job_id=os.environ.get("TRANSCRIBE_JOB_ID", ""),
                source=str(Path(args.input).resolve()), message="Transcription worker started")

    run_job_from_json({
        "input_path": args.input,
        "output_dir": args.output_dir or str(config.TRANSCRIPT_DIR),
        "model": args.model or config.WHISPER_MODEL,
        "device": args.device or config.DEVICE,
        "language": args.language if args.language is not None else config.LANGUAGE,
        "max_speakers": args.max_speakers if args.max_speakers is not None else config.MAX_SPEAKERS,
        "num_speakers": args.num_speakers if args.num_speakers is not None else config.NUM_SPEAKERS,
        "hotwords": args.hotwords if args.hotwords is not None else config.HOTWORDS,
        "transcribe_only": args.transcribe_only,
        "diarize_only": args.diarize_only,
        "token": "" if args.transcribe_only else get_hf_token(),
    })


if __name__ == "__main__":
    main()
