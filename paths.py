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


def transcript_path(input_path: Path, output_dir: Path, fingerprint: str | None = None) -> Path:
    """Return the output path used by both the worker and its controller (Markdown only)."""
    input_path = Path(input_path)
    fp = fingerprint or source_fingerprint(input_path)
    return Path(output_dir) / f"{input_path.stem}_{fp}.md"


def format_size(num_bytes: int | float | None) -> str:
    """Format bytes into a human-readable string (e.g., '128.5 MB')."""
    if num_bytes is None:
        return "0 B"
    try:
        size = float(num_bytes)
    except (TypeError, ValueError):
        return "0 B"
    if size <= 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"



def get_cache_size_bytes(cache_dir: Path | str) -> int:
    """Calculate the total size in bytes of all files in the cache directory."""
    path = Path(cache_dir)
    if not path.is_dir():
        return 0
    total = 0
    try:
        for entry in path.iterdir():
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total
