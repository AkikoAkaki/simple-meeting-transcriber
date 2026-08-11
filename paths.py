"""Lightweight, dependency-free paths for source-specific artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def source_fingerprint(input_path: Path) -> str:
    """Return a stable fingerprint for this version of a source file.

    The path alone is not enough: OBS or a manual workflow can replace a file
    at the same path. Including size and mtime prevents stale ML caches from
    being reused for the new contents while preserving cache hits for an
    unchanged file.
    """
    input_path = Path(input_path).resolve()
    stat = input_path.stat()
    identity = f"{input_path}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]


def transcript_path(input_path: Path, output_dir: Path, output_format: str = "md", fingerprint: str | None = None) -> Path:
    """Return the output path used by both the worker and its controller."""
    suffix = {"md": ".md", "srt": ".srt", "txt": ".txt"}.get(output_format, ".md")
    input_path = Path(input_path)
    fp = fingerprint or source_fingerprint(input_path)
    return Path(output_dir) / f"{input_path.stem}_{fp}{suffix}"

