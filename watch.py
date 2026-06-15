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
import time
from pathlib import Path

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    print("Missing dependency: pip install watchdog")
    sys.exit(1)

import config

TRANSCRIBE = Path(__file__).parent / "transcribe.py"


class VideoHandler(FileSystemEventHandler):
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self._pending: dict[str, tuple[int, float]] = {}

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
        self._pending[str(p)] = (size, time.time())
        logging.info(f"Detected: {p.name} ({size / 1024 / 1024:.1f} MB)")

    def flush_ready(self):
        """Check pending files; trigger transcription for stable ones."""
        now = time.time()
        ready = []

        for path_str, (last_size, last_changed) in list(self._pending.items()):
            p = Path(path_str)
            try:
                current_size = p.stat().st_size
            except FileNotFoundError:
                del self._pending[path_str]
                continue

            if current_size != last_size:
                self._pending[path_str] = (current_size, now)
            elif now - last_changed >= config.STABLE_SECONDS:
                if current_size >= config.MIN_FILE_SIZE_KB * 1024:
                    ready.append(p)
                else:
                    logging.warning(f"Skipping too-small file: {p.name} ({current_size} bytes)")
                del self._pending[path_str]

        for p in ready:
            self._transcribe(p)

    def _transcribe(self, video_path: Path):
        if config.ORGANIZE_BY_YEAR:
            year = video_path.stem[:4]
            if year.isdigit():
                dest_dir = config.WATCH_DIR / year
                dest_dir.mkdir(exist_ok=True)
                new_path = dest_dir / video_path.name
                try:
                    video_path.rename(new_path)
                    logging.info(f"Moved: {video_path.name} → {year}/")
                    video_path = new_path
                except OSError as e:
                    logging.error(f"Could not move file: {e}")
                    return

        logging.info(f"Transcribing: {video_path.name}")
        if self.dry_run:
            logging.info("[dry-run] skipping actual transcription")
            return

        proc = subprocess.Popen(
            [sys.executable, str(TRANSCRIBE), str(video_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            logging.info(f"  {line.rstrip()}")
        proc.wait()

        if proc.returncode == 0:
            logging.info(f"Transcription complete: {video_path.name}")
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
