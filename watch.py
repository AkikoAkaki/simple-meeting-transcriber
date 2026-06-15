#!/usr/bin/env python3
"""
meeting-transcriber — watch.py
Watches WATCH_DIR for new video files and triggers transcription automatically.

Usage:
  python watch.py            # start watcher
  python watch.py --dry-run  # detect files but don't transcribe
"""

import argparse
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
import json

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    print("Missing dependency: pip install watchdog")
    sys.exit(1)

import config

TRANSCRIBE = Path(__file__).parent / "transcribe.py"
SETTINGS_FILE = Path(__file__).parent / "user_settings.json"


def _load_user_output_dir(settings_path: Path = SETTINGS_FILE) -> Path | None:
    """Return the transcript dir saved by the GUI, or None if unset."""
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        d = data.get("transcript_dir")
        if d:
            return Path(d)
    except Exception:
        pass
    return None


class VideoHandler(FileSystemEventHandler):
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self._pending: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def on_created(self, event):
        self._track(event.src_path)

    def on_modified(self, event):
        self._track(event.src_path)

    def _track(self, src_path: str):
        p = Path(src_path)
        if p.is_dir() or p.suffix.lower() not in config.WATCH_EXTENSIONS:
            return
        # Only watch files dropped directly into WATCH_DIR (not subfolders)
        if p.parent.resolve() != config.WATCH_DIR.resolve():
            return
        try:
            size = p.stat().st_size
        except FileNotFoundError:
            return
        with self._lock:
            self._pending[str(p)] = (size, time.time())
        logging.info(f"Detected: {p.name} ({size / 1024 / 1024:.1f} MB)")

    def flush_ready(self):
        """Check pending files; trigger transcription for stable ones."""
        now = time.time()
        ready = []

        with self._lock:
            snapshot = list(self._pending.items())
        for path_str, (last_size, last_changed) in snapshot:
            p = Path(path_str)
            try:
                current_size = p.stat().st_size
            except FileNotFoundError:
                with self._lock:
                    self._pending.pop(path_str, None)
                continue

            if current_size != last_size:
                with self._lock:
                    self._pending[path_str] = (current_size, now)
            elif now - last_changed >= config.STABLE_SECONDS:
                if current_size >= config.MIN_FILE_SIZE_KB * 1024:
                    ready.append(p)
                else:
                    logging.warning(f"Skipping too-small file: {p.name} ({current_size} bytes)")
                with self._lock:
                    self._pending.pop(path_str, None)

        for p in ready:
            self._transcribe(p)

    def _transcribe(self, video_path: Path):
        logging.info(f"Transcribing: {video_path.name}")
        if self.dry_run:
            logging.info("[dry-run] skipping actual transcription")
            return

        cmd = [sys.executable, str(TRANSCRIBE), str(video_path)]
        out_dir = _load_user_output_dir()
        if out_dir:
            cmd += ["--output-dir", str(out_dir)]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            logging.info(f"  {line.rstrip()}")
        proc.wait()

        if proc.returncode == 0:
            logging.info(f"Transcription complete: {video_path.name}")
            if config.ORGANIZE_BY_YEAR:
                year = video_path.stem[:4]
                if year.isdigit():
                    dest_dir = config.WATCH_DIR / year
                    dest_dir.mkdir(exist_ok=True)
                    new_path = dest_dir / video_path.name
                    try:
                        video_path.rename(new_path)
                        logging.info(f"Moved: {video_path.name} → {year}/")
                    except OSError as e:
                        logging.error(f"Could not move file: {e}")
        else:
            logging.error(f"Transcription failed (exit {proc.returncode}): {video_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Watch for new videos and auto-transcribe")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect files but don't transcribe")
    args = parser.parse_args()

    log_file = config.ROOT_DIR / "watch.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    config.WATCH_DIR.mkdir(parents=True, exist_ok=True)

    handler = VideoHandler(dry_run=args.dry_run)
    observer = Observer()
    observer.schedule(handler, str(config.WATCH_DIR), recursive=False)
    observer.start()

    logging.info(f"Watching: {config.WATCH_DIR}")
    if args.dry_run:
        logging.info("[dry-run mode]")

    try:
        while True:
            handler.flush_ready()
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        logging.info("Watcher stopped")


if __name__ == "__main__":
    main()
