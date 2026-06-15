"""
meeting-transcriber configuration
All user-facing settings live here. Edit this file to customize behavior.
"""
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
# Root of this repo (don't change)
ROOT_DIR = Path(__file__).parent.resolve()

# Drop .mp4 / .m4a / .mov files here — watcher picks them up automatically
WATCH_DIR = ROOT_DIR / "inbox"

# Transcripts are written here as Markdown files
TRANSCRIPT_DIR = ROOT_DIR / "transcripts"

# Intermediate files (16k WAV, JSON caches) — safe to delete any time
CACHE_DIR = ROOT_DIR / "cache"

# After transcription, move video into a YYYY/ subfolder inside WATCH_DIR
ORGANIZE_BY_YEAR = True

# ── Whisper ───────────────────────────────────────────────────────────────────
# Model size vs. quality trade-off:
#   tiny / base / small  → fast, weaker accuracy
#   medium               → good balance for English
#   large-v2 / large-v3  → best quality, needs ~6 GB VRAM (recommended)
WHISPER_MODEL = "large-v3"

# Transcription language:
#   None  → auto-detect (recommended for mixed-language audio)
#   "en"  → force English
#   "zh"  → force Chinese
#   "ja"  → force Japanese  (see https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes)
LANGUAGE = None

# Compute device:
#   "auto"  → use CUDA if available, else CPU
#   "cuda"  → force GPU (faster, requires NVIDIA GPU)
#   "cpu"   → force CPU (slow but always works)
DEVICE = "auto"

# ── Speaker Diarization (pyannote) ───────────────────────────────────────────
# HuggingFace token — needed to download pyannote models.
# Leave empty to be prompted at runtime, or set env var HF_TOKEN.
# Get a free token at: https://hf.co/settings/tokens
# Then accept model terms at:
#   https://hf.co/pyannote/speaker-diarization-3.1
#   https://hf.co/pyannote/segmentation-3.0
HF_TOKEN = ""

# Expected number of speakers. Set to an integer for better accuracy.
# None = auto-detect.
MAX_SPEAKERS = None

# ── Watcher ──────────────────────────────────────────────────────────────────
# Seconds a file's size must be unchanged before transcription starts.
# Increase if you're copying large files over a slow network.
STABLE_SECONDS = 10

# Files smaller than this are ignored (filters out accidental tiny files).
MIN_FILE_SIZE_KB = 100

# Supported video/audio extensions the watcher will pick up.
WATCH_EXTENSIONS = {".mp4", ".m4a", ".mov", ".mkv", ".webm", ".mp3", ".wav"}
