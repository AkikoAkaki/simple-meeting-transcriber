"""Background service for Simple Video Transcriber.

This module intentionally contains no Qt imports.  It is the small, always-on
part of the application: it watches the OBS directory, persists job state, and
starts the heavy transcription process only when a file is ready.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
from collections import deque

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import config
from paths import transcript_path, format_size, get_cache_size_bytes


APP_DATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local")) / "SimpleVideoTranscriber"
SETTINGS_FILE = APP_DATA_DIR / "settings.json"
JOBS_DB = APP_DATA_DIR / "jobs.sqlite3"
LOG_DIR = APP_DATA_DIR / "logs"
TOKEN_FILE = APP_DATA_DIR / "hf_token.txt"
TRANSCRIBE_SCRIPT = Path(__file__).parent / "transcribe.py"
EVENT_PREFIX = "@@EVENT "
WORKER_START_TIMEOUT_SECONDS = 30.0
ACTIVE_STATUSES = {"queued", "running", "converting", "transcribing", "diarizing"}
TERMINAL_STATUSES = {"completed", "completed_with_warning", "failed", "cancelled"}


def parse_worker_line(line: str) -> dict | None:
    """Parse one worker stdout line without corrupting the JSON payload.

    ``EVENT_PREFIX`` is eight characters long; using ``removeprefix`` avoids
    the off-by-one bug that previously removed the opening ``{`` and turned
    every progress event into a raw log line.
    """
    if not line.startswith(EVENT_PREFIX):
        return None
    raw = line.removeprefix(EVENT_PREFIX)
    try:
        event = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"event": "parse_error", "message": "Malformed worker event", "raw": line}
    if not isinstance(event, dict) or not event.get("event"):
        return {"event": "parse_error", "message": "Worker event is not an object", "raw": line}
    return event


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Terminate the worker and any child processes it started."""
    try:
        if proc.poll() is not None:
            return
    except Exception:
        pass

    pid = getattr(proc, "pid", None)
    if os.name == "nt" and pid:
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return
        except (OSError, subprocess.SubprocessError):
            pass
    elif pid:
        try:
            import signal
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            return
        except (OSError, ProcessLookupError):
            pass

    try:
        proc.terminate()
    except (OSError, ProcessLookupError):
        pass


def clear_audio_cache(
    cache_dir: Path | str | None = None,
    store: JobStore | None = None,
) -> dict:
    """Delete only WAVs uniquely linked to terminal jobs via source fingerprint.

    Strictly preserves .json caches (whisper, diarization, segments) and never touches
    the transcripts directory. Never deletes WAVs belonging to active jobs
    (queued/running/converting/transcribing/diarizing/cancel_requested).
    Untracked WAVs (e.g. CLI runs) are always preserved: mtime alone cannot
    prove a file is unused because a CLI task may still be transcribing.
    Legacy `<stem>_16k.wav` names without a fingerprint are never trusted and
    are always preserved: they cannot distinguish a terminal job's old WAV
    from an untracked CLI WAV with the same stem.
    """
    if cache_dir is None:
        cache_dir = config.CACHE_DIR
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        return {"deleted_count": 0, "reclaimed_bytes": 0, "reclaimed_size": "0 B"}

    active_wav_stems: set[str] = set()
    terminal_wav_stems: set[str] = set()

    if store is None:
        try:
            store = JobStore()
        except Exception:
            store = None

    if store is not None:
        try:
            with store._connect() as conn:
                rows = conn.execute("SELECT source_path, source_key, status FROM jobs").fetchall()
            for r in rows:
                status = r["status"]
                src_path = Path(r["source_path"])
                try:
                    key = r["source_key"] if "source_key" in r.keys() else ""
                except Exception:
                    key = ""
                if not key:
                    continue
                fp = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
                stem = f"{src_path.stem}_{fp}_16k"
                if status in ACTIVE_STATUSES or status == "cancel_requested":
                    active_wav_stems.add(stem)
                elif status in TERMINAL_STATUSES:
                    terminal_wav_stems.add(stem)
        except Exception:
            pass

    def _extract_base_stem(filename: str) -> str:
        b = filename
        for _ in range(2):
            if b.endswith(".part"):
                b = b[:-5]
            if b.endswith(".wav"):
                b = b[:-4]
        return b

    deleted_count = 0
    reclaimed_bytes = 0

    try:
        entries = list(cache_path.iterdir())
    except OSError:
        entries = []

    for item in entries:
        if not item.is_file():
            continue
        name = item.name
        # Never touch non-audio files; strictly preserve all json caches
        if name.endswith(".json") or ".json.part" in name:
            continue
        is_wav = name.endswith(".wav") or ".wav.part" in name or name.endswith("_16k.part")
        if not is_wav:
            continue

        base = _extract_base_stem(name)

        # Active jobs must NEVER have their audio deleted
        if base in active_wav_stems:
            continue

        # Only WAVs explicitly associated with terminal jobs may be deleted.
        # Untracked files (including old CLI WAVs) are always preserved.
        if base not in terminal_wav_stems:
            continue

        try:
            size = item.stat().st_size
            item.unlink()
            deleted_count += 1
            reclaimed_bytes += size
        except OSError:
            pass

    return {
        "deleted_count": deleted_count,
        "reclaimed_bytes": reclaimed_bytes,
        "reclaimed_size": format_size(reclaimed_bytes),
    }



@dataclass
class AppSettings:
    """User settings shared by the tray process and transcription workers."""

    watch_dir: str = str(config.WATCH_DIR)
    transcript_dir: str = str(config.TRANSCRIPT_DIR)
    model: str = config.WHISPER_MODEL
    device: str = config.DEVICE
    language: str | None = config.LANGUAGE
    max_speakers: int | None = config.MAX_SPEAKERS
    num_speakers: int | None = config.NUM_SPEAKERS
    hotwords: str = config.HOTWORDS
    stable_seconds: int = 15
    min_file_size_kb: int = config.MIN_FILE_SIZE_KB
    watcher_enabled: bool = True

    def __post_init__(self) -> None:
        self.model = config.normalize_whisper_model(self.model)

    @classmethod
    def load(cls, path: Path | None = None) -> "AppSettings":
        # Resolve the module-level path at call time. Tests and alternate
        # installations may replace SETTINGS_FILE after import.
        path = path or SETTINGS_FILE
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            allowed = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
            return cls(**allowed)
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path | None = None) -> None:
        # Do not bind SETTINGS_FILE in a default argument: that allowed a
        # pytest tmp_path to be written into the real user settings file.
        path = path or SETTINGS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


class TokenStore:
    """One canonical token source, with OS keyring when available.

    The file fallback keeps the personal installation usable if keyring's
    backend is not installed.  It is stored outside the repository so the
    token cannot accidentally be committed with project files.
    """

    SERVICE = "simple-video-transcriber"
    USER = "huggingface"

    def __init__(self, file_path: Path = TOKEN_FILE):
        self.file_path = file_path
        self._cached_token = None

    def get(self) -> str:
        if self._cached_token is not None:
            return self._cached_token
        value = ""
        try:
            import keyring
            value = keyring.get_password(self.SERVICE, self.USER)
            if value:
                value = value.strip()
        except Exception:
            pass
        if not value:
            try:
                value = self.file_path.read_text(encoding="utf-8").strip()
            except OSError:
                value = ""
        self._cached_token = value
        return value

    def set(self, token: str) -> None:
        token = token.strip()
        self._cached_token = token
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import keyring
            if token:
                keyring.set_password(self.SERVICE, self.USER, token)
            else:
                keyring.delete_password(self.SERVICE, self.USER)
            if self.file_path.exists():
                self.file_path.unlink()
            return
        except Exception:
            pass
        if token:
            self.file_path.write_text(token, encoding="utf-8")
        elif self.file_path.exists():
            self.file_path.unlink()


class JobStore:
    """Small SQLite-backed job store.

    Each thread uses its own connection, so callbacks from watchdog and
    worker threads do not share a SQLite connection.
    """

    def __init__(self, path: Path = JOBS_DB):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_key TEXT NOT NULL UNIQUE,
                    source_size INTEGER NOT NULL,
                    source_mtime_ns INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL DEFAULT '',
                    progress REAL,
                    message TEXT NOT NULL DEFAULT '',
                    output_path TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    options_json TEXT DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_updated ON jobs(updated_at DESC);
                """
            )
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN options_json TEXT DEFAULT '{}'")
                conn.commit()
            except sqlite3.OperationalError:
                pass

    @staticmethod
    def source_key(path: Path) -> tuple[str, int, int, str]:
        stat = path.stat()
        resolved = str(path.resolve())
        return resolved, stat.st_size, stat.st_mtime_ns, f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"

    def create_if_new(self, path: Path, options: dict = None) -> dict | None:
        try:
            resolved, size, mtime, key = self.source_key(path)
        except OSError:
            return None
        now = _now()
        options_json = json.dumps(options or {}, ensure_ascii=False)
        with self._connect() as conn:
            existing = conn.execute("SELECT * FROM jobs WHERE source_key = ?", (key,)).fetchone()
            if existing:
                existing = dict(existing)
                if existing["status"] in {"queued", "running", "converting", "transcribing", "diarizing", "cancel_requested"}:
                    return None
                job_id = existing["job_id"]
                conn.execute(
                    "UPDATE jobs SET status='queued', stage='queued', progress=0.0, message='Queued', "
                    "output_path='', error='', started_at='', finished_at='', updated_at=?, options_json=?, "
                    "retry_count=0 WHERE job_id = ?",
                    (now, options_json, job_id)
                )
                # fetch and return updated row
                row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                return dict(row) if row else None

            job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            row = {
                "job_id": job_id,
                "source_path": resolved,
                "source_key": key,
                "source_size": size,
                "source_mtime_ns": mtime,
                "status": "queued",
                "stage": "queued",
                "progress": 0.0,
                "message": "Queued",
                "output_path": "",
                "error": "",
                "created_at": now,
                "updated_at": now,
                "started_at": "",
                "finished_at": "",
                "retry_count": 0,
                "options_json": options_json,
            }
            try:
                conn.execute(
                    "INSERT INTO jobs ({}) VALUES ({})".format(
                        ",".join(row), ",".join("?" for _ in row)
                    ),
                    tuple(row.values()),
                )
            except sqlite3.IntegrityError:
                return None
        return row

    def update(self, job_id: str, **changes) -> dict | None:
        changes["updated_at"] = _now()
        assignments = ", ".join(f"{key} = ?" for key in changes)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id = ?", (*changes.values(), job_id))
        return self.get(job_id)

    def update_if_status(self, job_id: str, expected_statuses: Iterable[str], **changes) -> dict | None:
        """Apply a state transition only if the current status still matches."""
        statuses = tuple(expected_statuses)
        if not statuses:
            return None
        changes["updated_at"] = _now()
        assignments = ", ".join(f"{key} = ?" for key in changes)
        placeholders = ",".join("?" for _ in statuses)
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE jobs SET {assignments} WHERE job_id = ? AND status IN ({placeholders})",
                (*changes.values(), job_id, *statuses),
            )
        return self.get(job_id) if cur.rowcount == 1 else None

    def get(self, job_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def recent(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def queued(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at, job_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def recover_interrupted(self) -> int:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='cancelled', stage='cancelled', "
                "message='Cancelled during previous shutdown', updated_at=?, finished_at=? "
                "WHERE status='cancel_requested'",
                (now, now),
            )
            # Mark jobs exceeding limit as failed
            conn.execute(
                "UPDATE jobs SET status='failed', stage='failed', message='Failed after 5 recovery retries', "
                "updated_at=?, finished_at=? WHERE status IN ('running','converting','transcribing','diarizing') AND retry_count >= 5",
                (now, now),
            )
            # Recover jobs under limit
            cur = conn.execute(
                "UPDATE jobs SET status='queued', stage='queued', message='Recovered after restart', "
                "updated_at=?, retry_count=retry_count+1 WHERE status IN ('running','converting','transcribing','diarizing') AND retry_count < 5",
                (now,),
            )
        return cur.rowcount


def _is_supported(path: Path, settings: AppSettings) -> bool:
    return path.is_file() and path.suffix.lower() in config.WATCH_EXTENSIONS


class _WatchHandler(FileSystemEventHandler):
    def __init__(self, owner: "FileWatcher"):
        self.owner = owner

    def on_created(self, event):
        if not event.is_directory:
            self.owner.track(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            self.owner.track(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            self.owner.track(Path(event.dest_path))


class FileWatcher:
    """Event-driven watcher with a stability debounce for files OBS is writing."""

    # Watchdog can drop events (buffer overflow, sleep/wake, network drives).
    # Reconcile the directory this often so a missed file is still picked up.
    RESCAN_INTERVAL_SECONDS = 60.0

    def __init__(self, settings: AppSettings, on_event: Callable[[str, dict], None]):
        self.settings = settings
        self.on_event = on_event
        self._pending: dict[str, tuple[int, float]] = {}
        self._known: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_rescan = time.monotonic()
        self._observer: Observer | None = None
        self._thread: threading.Thread | None = None
        try:
            self._resolved_watch_dir = _safe_path(self.settings.watch_dir).resolve()
        except Exception:
            self._resolved_watch_dir = None

    def start(self) -> None:
        root = _safe_path(self.settings.watch_dir)
        self._resolved_watch_dir = root.resolve()
        if not root.exists():
            self.on_event("watch_error", {"message": f"Watch folder unavailable: {root}"})
            self._stop.clear()
            self._thread = threading.Thread(target=self._retry_until_available,
                                             args=(root,), name="watch-reconnect", daemon=True)
            self._thread.start()
            return
        root.mkdir(parents=True, exist_ok=True)
        self._start_existing(root)

    def _start_existing(self, root: Path) -> None:
        self._resolved_watch_dir = root.resolve()
        # Baseline existing files.  A first launch must not process an archive.
        self._known = {str(p.resolve()) for p in root.iterdir() if _is_supported(p, self.settings)}
        self._observer = Observer()
        self._observer.schedule(_WatchHandler(self), str(root), recursive=False)
        self._observer.start()
        self._stop.clear()
        self._last_rescan = time.monotonic()
        self._thread = threading.Thread(target=self._tick_loop, name="file-stability", daemon=True)
        self._thread.start()
        self.on_event("watch_started", {"path": str(root), "baseline_count": len(self._known)})

    def _retry_until_available(self, root: Path) -> None:
        while not self._stop.wait(30.0):
            if root.exists():
                self._start_existing(root)
                return

    def stop(self) -> None:
        self._stop.set()
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def track(self, path: Path) -> None:
        try:
            path = path.resolve()
            if path.parent != self._resolved_watch_dir:
                return
            if not _is_supported(path, self.settings):
                return
            stat = path.stat()
        except OSError:
            return
        key = str(path)
        with self._lock:
            old = self._pending.get(key)
            if old is None:
                self._pending[key] = (stat.st_size, time.time())
                self.on_event("detected", {"path": str(path), "size": stat.st_size})
            elif old[0] != stat.st_size:
                self._pending[key] = (stat.st_size, time.time())

    def rescan(self) -> None:
        """Reconcile the watch directory against known/pending state.

        Picks up files whose watchdog events were lost. Already-seen files
        are skipped, and undersized files are intentionally left silent for
        a later pass (they may still be growing, and emitting `ignored`
        here would repeat on every pass). A rescan never emits duplicate
        events.
        """
        root = self._resolved_watch_dir
        if root is None:
            return
        try:
            entries = list(root.iterdir())
        except OSError:
            return
        with self._lock:
            known = set(self._known)
            pending = set(self._pending)
        try:
            min_bytes = self.settings.min_file_size_kb * 1024
        except (AttributeError, TypeError):
            min_bytes = 0
        for entry in entries:
            try:
                if not _is_supported(entry, self.settings):
                    continue
                key = str(entry.resolve())
            except OSError:
                continue
            if key in known or key in pending:
                continue
            try:
                if entry.stat().st_size < min_bytes:
                    continue
            except OSError:
                continue
            self.track(entry)

    def tick_once(self, now: float | None = None) -> list[Path]:
        now = now or time.time()
        ready: list[Path] = []
        with self._lock:
            snapshot = list(self._pending.items())
        for key, (last_size, last_changed) in snapshot:
            path = Path(key)
            try:
                current = path.stat().st_size
            except OSError:
                with self._lock:
                    self._pending.pop(key, None)
                continue
            if current != last_size:
                with self._lock:
                    self._pending[key] = (current, now)
                continue
            if now - last_changed < self.settings.stable_seconds:
                continue
            try:
                with open(path, "r+b"):
                    pass
            except FileNotFoundError:
                with self._lock:
                    self._pending.pop(key, None)
                continue
            except (PermissionError, OSError) as exc:
                winerror = getattr(exc, "winerror", None)
                is_sharing_violation = (
                    winerror in {32, 33}
                    or "being used by another process" in str(exc).lower()
                    or "sharing violation" in str(exc).lower()
                )
                if is_sharing_violation or winerror is None:
                    # File is still locked/being written (e.g. OBS recording, sharing violation error 32).
                    # Keep pending and wait for next tick.
                    with self._lock:
                        self._pending[key] = (current, now)
                    continue
                # If it's a different permission error (e.g. read-only file winerror 5),
                # verify it can at least be read for transcription.
                try:
                    with open(path, "rb"):
                        pass
                except OSError:
                    with self._lock:
                        self._pending[key] = (current, now)
                    continue
            with self._lock:
                self._pending.pop(key, None)
            if current >= self.settings.min_file_size_kb * 1024:
                self._known.add(key)
                ready.append(path)
                self.on_event("ready", {"path": str(path), "size": current})
            else:
                self.on_event("ignored", {"path": str(path), "message": "File is below minimum size"})
        return ready

    def _tick_loop(self) -> None:
        while not self._stop.wait(2.0):
            self.tick_once()
            if time.monotonic() - self._last_rescan >= self.RESCAN_INTERVAL_SECONDS:
                self._last_rescan = time.monotonic()
                try:
                    self.rescan()
                except Exception as exc:
                    self.on_event("log", {"message": f"Directory rescan failed: {exc}"})


class WorkerController:
    """Serial worker queue. Heavy ML dependencies live in a persistent child server."""

    def __init__(self, settings: AppSettings, store: JobStore, token_store: TokenStore,
                 on_event: Callable[[str, dict], None]):
        self.settings = settings
        self.store = store
        self.token_store = token_store
        self.on_event = on_event
        self._queue: deque[dict] = deque()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._active: dict | None = None
        self._last_progress: dict[str, tuple[float, float]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="transcription-worker", daemon=True)
        self._thread.start()

    @property
    def active(self) -> dict | None:
        with self._lock:
            return dict(self._active) if self._active else None

    def enqueue(self, row: dict) -> None:
        with self._lock:
            self._queue.append(row)
            self._last_progress.pop(row["job_id"], None)
        self._emit(row["job_id"], "queued", {"message": "Queued"})

    def stop(self) -> None:
        self._stop.set()
        row = self._request_cancel("Cancellation requested on exit")
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc:
            _terminate_process_tree(proc)
        if row:
            current = self.store.get(row["job_id"])
            if current and current.get("status") == "cancel_requested":
                self._finalize_cancelled(row["job_id"], "Cancelled on exit")

    def cancel_active(self) -> None:
        self._request_cancel("Cancellation requested")

    def _request_cancel(self, message: str) -> dict | None:
        with self._lock:
            row = self._active
        if not row:
            return None
        changed = self.store.update_if_status(
            row["job_id"], ACTIVE_STATUSES,
            status="cancel_requested", stage="cancelling", message=message,
        )
        if changed:
            self._emit(row["job_id"], "cancel_requested", {"message": message})
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc:
            _terminate_process_tree(proc)
        return row

    def _finalize_cancelled(self, job_id: str, message: str = "Cancelled") -> None:
        current = self.store.get(job_id)
        if not current or current.get("status") == "cancelled":
            return
        self.store.update(job_id, status="cancelled", stage="cancelled", message=message, finished_at=_now())
        self._emit(job_id, "cancelled", {"stage": "cancelled", "message": message})

    def _emit(self, job_id: str, event: str, payload: dict) -> None:
        payload = {"job_id": job_id, "event": event, **payload}
        self.on_event(event, payload)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            row = None
            with self._lock:
                if self._queue:
                    row = self._queue.popleft()
                    self._active = row
            if row:
                self._run_one_server(row)
                with self._lock:
                    self._active = None
                continue
            self._stop.wait(0.5)

    def _ensure_server_running(self) -> subprocess.Popen:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self._proc

            cmd = [sys.executable, str(TRANSCRIBE_SCRIPT), "--server"]
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            popen_kwargs = {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "env": env,
                "cwd": str(Path(__file__).parent),
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_kwargs["start_new_session"] = True

            proc = subprocess.Popen(cmd, **popen_kwargs)
            self._proc = proc

        assert proc.stdout is not None
        startup = {"ready": False, "logs": [], "error": None}

        def read_startup() -> None:
            try:
                for raw in proc.stdout:
                    line = raw.rstrip()
                    if line == "SERVER_READY":
                        startup["ready"] = True
                        return
                    startup["logs"].append(line)
            except Exception as exc:
                startup["error"] = exc

        reader = threading.Thread(target=read_startup, name="worker-startup", daemon=True)
        reader.start()
        reader.join(timeout=WORKER_START_TIMEOUT_SECONDS)

        for line in startup["logs"]:
            self.on_event("log", {"message": f"[Worker Startup] {line}"})

        if reader.is_alive() or not startup["ready"]:
            with self._lock:
                if self._proc is proc:
                    self._proc = None
            _terminate_process_tree(proc)
            if reader.is_alive():
                raise TimeoutError(
                    f"Worker did not become ready within {WORKER_START_TIMEOUT_SECONDS:g} seconds"
                )
            detail = str(startup["error"]) if startup["error"] else f"exit code {proc.poll()}"
            raise RuntimeError(f"Worker exited before readiness confirmation ({detail})")

        return proc

    def _run_one_server(self, row: dict) -> None:
        job_id = row["job_id"]
        source = Path(row["source_path"])
        out_dir = _safe_path(self.settings.transcript_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        job_db = self.store.get(job_id) or row
        options = {}
        if job_db.get("options_json"):
            try:
                options = json.loads(job_db["options_json"])
            except Exception:
                pass

        lang = options.get("language") if "language" in options else self.settings.language
        pipeline = options.get("pipeline")
        transcribe_only = pipeline in {"仅转录", "Transcribe only"}
        diarize_only = pipeline in {"仅重新分离", "Re-diarize only"}
        spk = options.get("max_speakers") if "max_speakers" in options else self.settings.max_speakers
        exact_spk = options.get("num_speakers") if "num_speakers" in options else self.settings.num_speakers
        hotwords = options.get("hotwords") if "hotwords" in options else self.settings.hotwords

        started = self.store.update_if_status(
            job_id, {"queued"}, status="running", stage="starting",
            message="Starting worker", started_at=_now(),
        )
        if not started:
            current = self.store.get(job_id) or row
            if current.get("status") in {"cancel_requested", "cancelled"}:
                self._finalize_cancelled(job_id)
            return

        self._emit(job_id, "started", {"stage": "starting", "message": "Starting worker"})
        worker_failed = False
        return_code = 0
        proc = None
        terminal_kind = None

        try:
            proc = self._ensure_server_running()

            task = {
                "command": "transcribe",
                "job_id": job_id,
                "input_path": str(source),
                "output_dir": str(out_dir),
                "model": self.settings.model,
                "device": self.settings.device,
                "max_speakers": spk if spk and str(spk).lower() != "auto" else None,
                "num_speakers": exact_spk if exact_spk and str(exact_spk).lower() != "auto" else None,
                "language": lang,
                "hotwords": hotwords or "",
                "transcribe_only": transcribe_only,
                "diarize_only": diarize_only,
                "token": self.token_store.get()
            }

            assert proc.stdin is not None
            proc.stdin.write(json.dumps(task) + "\n")
            proc.stdin.flush()

            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip()
                if line.startswith(EVENT_PREFIX):
                    event = parse_worker_line(line)
                    if event and event.get("event") != "parse_error":
                        kind = event.get("event")
                        event_job_id = event.get("job_id")
                        if event_job_id and event_job_id != job_id:
                            self.on_event(
                                "log",
                                {"job_id": job_id,
                                 "message": f"Ignored stale worker event for job {event_job_id}"},
                            )
                            continue
                        worker_failed = worker_failed or kind == "failed"
                        self._apply_worker_event(job_id, event)
                        if kind in {"completed", "completed_with_warning", "failed", "cancelled"}:
                            terminal_kind = kind
                            break
                    elif event:
                        self.on_event("log", {"job_id": job_id, "message": event["message"]})
                else:
                    self.on_event("log", {"job_id": job_id, "message": line})

            # A terminal worker event is authoritative: a completed event
            # followed by a non-zero exit (e.g. killed while flushing) must not
            # flip a done job to failed. Fall back to the exit code only when
            # the worker ended without sending any terminal event.
            if terminal_kind is None:
                ret = proc.poll()
                if ret is not None and ret != 0:
                    return_code = ret
                worker_failed = True
                self.store.update_if_status(
                    job_id,
                    {"running"},
                    message="Worker connection closed before a terminal event",
                    error="Worker connection closed before a terminal event",
                )
        except Exception as exc:
            return_code = -1
            worker_failed = True
            self.store.update_if_status(
                job_id, {"running"}, stage="starting", message=str(exc), error=str(exc),
            )
            with self._lock:
                if self._proc is proc:
                    self._proc = None
            if proc is not None:
                _terminate_process_tree(proc)

        latest = self.store.get(job_id) or row
        if terminal_kind == "cancelled" or latest.get("status") in {"cancelled", "cancel_requested"}:
            if latest.get("status") != "cancelled":
                self._finalize_cancelled(job_id)
        elif terminal_kind in {"completed", "completed_with_warning"} and return_code == 0 and not worker_failed:
            try:
                output = latest.get("output_path") or str(transcript_path(source, out_dir))
            except OSError as exc:
                message = f"Could not resolve expected output path: {exc}"
                self.store.update(job_id, status="failed", stage="failed", message=message,
                                  error=message, finished_at=_now())
                self._emit(job_id, "failed", {"stage": "failed", "message": message, "error": message})
                return
            output_file = Path(output)
            try:
                output_ok = output_file.is_file() and output_file.stat().st_size > 0
            except OSError:
                output_ok = False
            if not output_ok:
                message = f"Worker exited successfully but output file is missing or empty: {output}"
                self.store.update(job_id, status="failed", stage="failed", message=message,
                                  error=message, finished_at=_now())
                self._emit(job_id, "failed", {"stage": "failed", "message": message, "error": message})
                return
            had_warning = terminal_kind == "completed_with_warning" or latest.get("status") == "completed_with_warning" or bool(latest.get("error"))
            final_status = "completed_with_warning" if had_warning else "completed"
            final_message = "Completed with warnings" if had_warning else "Completed"
            self.store.update(job_id, status=final_status, stage="completed", progress=1.0,
                              message=final_message, output_path=output, finished_at=_now())
            self._emit(job_id, final_status,
                       {"stage": "completed", "progress": 1.0,
                        "message": final_message, "output_path": output})
        else:
            message = latest.get("error") or f"Worker exited with code {return_code}"
            self.store.update(job_id, status="failed", stage="failed", message=message,
                              error=message, finished_at=_now())
            self._emit(job_id, "failed", {"stage": "failed", "message": message, "error": message})

    def _apply_worker_event(self, job_id: str, event: dict) -> None:
        event = dict(event)
        event["job_id"] = job_id
        kind = event.get("event", "log")

        if kind == "progress" and not self._should_forward_progress(job_id, event):
            return
        stage = event.get("stage", "")
        changes = {"stage": stage or kind, "message": event.get("message", kind)}
        if isinstance(event.get("progress"), (int, float)):
            changes["progress"] = max(0.0, min(1.0, float(event["progress"])))

        if "preview" in event:
            with self._lock:
                if self._active and self._active.get("job_id") == job_id:
                    self._active["preview"] = event["preview"]

        if kind == "failed":
            changes["error"] = event.get("error", event.get("message", "Error"))
        elif kind == "cancelled":
            changes["stage"] = "cancelled"
        elif kind in {"completed", "completed_with_warning"}:
            if "output_path" in event:
                changes["output_path"] = event["output_path"]
        elif kind == "warning":
            changes["error"] = event.get("message", "Warning")
        updated = self.store.update_if_status(job_id, {"running"}, **changes)
        if not updated:
            return
        if kind in {"completed", "completed_with_warning", "failed", "cancelled"}:
            with self._lock:
                if self._active and self._active.get("job_id") == job_id:
                    self._active.pop("preview", None)
            return
        self.on_event(kind, event)

    def _should_forward_progress(self, job_id: str, event: dict) -> bool:
        """Keep live UI updates responsive without persisting every segment."""
        progress = event.get("progress")
        now = time.monotonic()
        if "preview" in event:
            self._last_progress[job_id] = (
                float(progress) if isinstance(progress, (int, float)) else -1.0,
                now,
            )
            return True
        previous = self._last_progress.get(job_id)
        if previous is None:
            self._last_progress[job_id] = (float(progress) if isinstance(progress, (int, float)) else -1.0, now)
            return True
        last_value, last_time = previous
        value = float(progress) if isinstance(progress, (int, float)) else last_value
        if value >= 1.0 or value - last_value >= 0.01 or now - last_time >= 1.0:
            self._last_progress[job_id] = (value, now)
            return True
        return False


class BackgroundService:
    """Coordinates watcher, durable state, and the serial worker queue."""

    def __init__(self, settings: AppSettings | None = None,
                 on_event: Callable[[str, dict], None] | None = None):
        self.settings = settings or AppSettings.load()
        try:
            normalize = getattr(config, "normalize_whisper_model", None)
            if callable(normalize):
                self.settings.model = normalize(self.settings.model)
        except Exception:
            pass
        self.settings.save()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.store = JobStore(JOBS_DB)
        self.token_store = TokenStore(TOKEN_FILE)
        self.on_event = on_event or (lambda _event, _payload: None)
        self.worker = WorkerController(self.settings, self.store, self.token_store, self._event)
        self.watcher = FileWatcher(self.settings, self._event)
        self._started = False
        self._log_lock = threading.Lock()
        self._log_count = 0

    def start(self) -> None:
        if self._started:
            return
        self.store.recover_interrupted()
        if self.settings.watcher_enabled:
            self.watcher.start()
        self._started = True
        for row in self.store.queued():
            self.worker.enqueue(row)
        self._event("service_started", {"message": "Background service started"})

    def stop(self) -> None:
        self.watcher.stop()
        self.worker.stop()
        self._started = False

    def set_watcher_enabled(self, enabled: bool) -> None:
        self.settings.watcher_enabled = enabled
        self.settings.save()
        if enabled:
            self.watcher.start()
        else:
            self.watcher.stop()
            self._event("watch_stopped", {"message": "Watcher paused"})

    def set_watch_dir(self, path: Path) -> None:
        was_enabled = self.settings.watcher_enabled
        self.watcher.stop()
        self.settings.watch_dir = str(path)
        self.settings.save()
        if was_enabled:
            self.watcher.start()

    def rescan_watch_dir(self) -> None:
        """Manually reconcile the watch directory (picks up missed files now)."""
        self.watcher.rescan()

    def add_file(self, path: Path, options: dict = None) -> dict | None:
        row = self.store.create_if_new(path, options)
        if row:
            self.worker.enqueue(row)
        return row

    def get_cache_size(self) -> str:
        """Return human-readable cache size."""
        return format_size(get_cache_size_bytes(config.CACHE_DIR))

    def clear_audio_cache(self) -> dict:
        """Delete temporary .wav files from completed/failed tasks."""
        return clear_audio_cache(config.CACHE_DIR, self.store)

    def _event(self, event: str, payload: dict) -> None:
        if event == "ready":
            path = Path(payload["path"])
            row = self.add_file(path)
            if row:
                payload = {**payload, "job_id": row["job_id"], "message": "Added to queue"}
            else:
                return
        self._write_log(event, payload)
        self.on_event(event, payload)

    def _write_log(self, event: str, payload: dict) -> None:
        """Persist a concise readable activity log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = str(payload.get("message") or event.replace("_", " "))
        token = self.token_store.get()
        if token:
            message = message.replace(token, "[redacted]")
        job_id = payload.get("job_id")
        line = f"{timestamp} [{event}] {message}\n"
        self._log_count += 1
        with self._log_lock:
            app_log = LOG_DIR / "app.log"
            if self._log_count % 100 == 0:
                try:
                    if app_log.exists() and app_log.stat().st_size > 5 * 1024 * 1024:
                        app_log.replace(LOG_DIR / "app.log.1")
                except OSError:
                    pass
            with app_log.open("a", encoding="utf-8") as handle:
                handle.write(line)
            if job_id:
                job_log = LOG_DIR / f"{job_id}.log"
                with job_log.open("a", encoding="utf-8") as handle:
                    handle.write(line)
