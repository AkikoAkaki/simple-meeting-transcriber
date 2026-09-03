"""Tests for the light-weight watcher and durable job layer."""

import time
from pathlib import Path

from service import AppSettings, FileWatcher, JobStore, TokenStore, parse_worker_line


def _settings(tmp_path: Path) -> AppSettings:
    watch = tmp_path / "watch"
    watch.mkdir()
    return AppSettings(
        watch_dir=str(watch),
        transcript_dir=str(tmp_path / "transcripts"),
        stable_seconds=1,
        min_file_size_kb=0,
    )


def test_watcher_baselines_existing_files_and_emits_new_file(tmp_path):
    settings = _settings(tmp_path)
    existing = Path(settings.watch_dir) / "old.mp4"
    existing.write_bytes(b"old")
    events = []
    watcher = FileWatcher(settings, lambda event, payload: events.append((event, payload)))
    watcher.start()
    try:
        assert not any(event == "detected" for event, _ in events)
        new_file = Path(settings.watch_dir) / "new.mp4"
        new_file.write_bytes(b"new" * 100)
        watcher.track(new_file)
        assert any(event == "detected" for event, _ in events)
        watcher._pending[str(new_file.resolve())] = (new_file.stat().st_size, time.time() - 2)
        ready = watcher.tick_once()
        assert ready == [new_file.resolve()]
        assert not any(payload.get("path", "").endswith("old.mp4") for _, payload in events)
    finally:
        watcher.stop()


def test_watcher_resets_stability_when_file_grows(tmp_path):
    settings = _settings(tmp_path)
    events = []
    watcher = FileWatcher(settings, lambda event, payload: events.append((event, payload)))
    path = Path(settings.watch_dir) / "growing.mkv"
    path.write_bytes(b"a" * 100)
    watcher.track(path)
    path.write_bytes(b"b" * 200)
    watcher.track(path)
    ready = watcher.tick_once(time.time() + 0.1)
    assert ready == []


def test_job_store_deduplicates_same_file(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    source = tmp_path / "meeting.mkv"
    source.write_bytes(b"x" * 100)
    first = store.create_if_new(source)
    second = store.create_if_new(source)
    assert first is not None
    assert second is None
    assert store.get(first["job_id"])["status"] == "queued"


def test_token_store_uses_private_app_file_fallback(tmp_path):
    path = tmp_path / "hf_token.txt"
    store = TokenStore(path)
    store.set("hf_test_token")
    assert store.get() == "hf_test_token"
    store.set("")
    assert store.get() == ""


def test_parse_worker_line_preserves_json_opening_brace():
    event = parse_worker_line(
        '@@EVENT {"event":"progress","stage":"transcribing","progress":0.48}'
    )
    assert event == {"event": "progress", "stage": "transcribing", "progress": 0.48}


def test_parse_worker_line_does_not_turn_malformed_event_into_progress():
    event = parse_worker_line("@@EVENT {not-json")
    assert event["event"] == "parse_error"
    assert "raw" in event


def test_token_store_caches_token(tmp_path, monkeypatch):
    path = tmp_path / "hf_token.txt"
    store = TokenStore(path)

    calls = 0
    def mock_get_password():
        nonlocal calls
        calls += 1
        return "my_mocked_token"

    import sys
    class FakeKeyring:
        def get_password(self, service, user):
            return mock_get_password()

    monkeypatch.setitem(sys.modules, "keyring", FakeKeyring())

    t1 = store.get()
    assert t1 == "my_mocked_token"
    assert calls == 1

    t2 = store.get()
    assert t2 == "my_mocked_token"
    assert calls == 1


def test_job_store_recover_interrupted_limits_retries(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    source = tmp_path / "meeting.mkv"
    source.write_bytes(b"x" * 100)

    job = store.create_if_new(source)
    job_id = job["job_id"]

    store.update(job_id, status="running", retry_count=4)
    recovered = store.recover_interrupted()
    assert recovered == 1
    assert store.get(job_id)["status"] == "queued"
    assert store.get(job_id)["retry_count"] == 5

    store.update(job_id, status="running")
    recovered = store.recover_interrupted()
    assert recovered == 0
    assert store.get(job_id)["status"] == "failed"
    assert "Failed after 5 recovery retries" in store.get(job_id)["message"]


def test_worker_controller_cancellation(tmp_path, monkeypatch):
    from service import WorkerController, JobStore, TokenStore, AppSettings
    store = JobStore(tmp_path / "jobs.sqlite3")
    token_store = TokenStore(tmp_path / "token.txt")
    settings = AppSettings(transcript_dir=str(tmp_path / "transcripts"))

    events = []
    controller = WorkerController(
        settings, store, token_store,
        lambda event, payload: on_event(event, payload)
    )

    def on_event(event, payload):
        events.append((event, payload))
        if event == "progress" and payload.get("progress") == 0.5:
            controller.cancel_active()

    source = tmp_path / "test.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)
    job_id = job["job_id"]

    stdout_lines = [
        '@@EVENT {"event":"stage","stage":"converting","message":"FFmpeg conversion"}',
        '@@EVENT {"event":"progress","stage":"converting","progress":0.5}',
        '@@EVENT {"event":"progress","stage":"converting","progress":0.8}',
    ]

    class Input:
        def __init__(self):
            self.data = ""

        def write(self, value):
            self.data += value

        def flush(self):
            pass

    class MockProc:
        def __init__(self):
            self.stdin = Input()
            self.stdout = iter(stdout_lines)

        def poll(self):
            return None

    monkeypatch.setattr(controller, "_ensure_server_running", lambda: MockProc())

    # Set active job manually because we are calling _run_one_server directly
    controller._active = job
    controller._run_one_server(job)

    final_job = store.get(job_id)
    assert final_job["status"] == "cancelled"
    # Ensure progress 0.8 was ignored (progress remains 0.5 or matches the last allowed one)
    assert final_job["progress"] == 0.5


def test_cancel_ignores_late_terminal_worker_events(tmp_path):
    from service import WorkerController, JobStore, TokenStore, AppSettings
    store = JobStore(tmp_path / "jobs.sqlite3")
    controller = WorkerController(AppSettings(transcript_dir=str(tmp_path / "transcripts")), store,
                                  TokenStore(tmp_path / "token.txt"), lambda *_: None)
    source = tmp_path / "late.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)
    store.update(job["job_id"], status="cancel_requested")
    controller._apply_worker_event(job["job_id"], {"event": "completed", "output_path": "bad.md"})
    controller._apply_worker_event(job["job_id"], {"event": "failed", "error": "late"})
    current = store.get(job["job_id"])
    assert current["status"] == "cancel_requested"
    assert current["output_path"] == ""


def test_conditional_status_update_cannot_overwrite_cancellation(tmp_path):
    from service import JobStore
    store = JobStore(tmp_path / "jobs.sqlite3")
    source = tmp_path / "atomic.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)
    store.update(job["job_id"], status="running")
    store.update_if_status(job["job_id"], {"running"}, status="cancel_requested")
    overwritten = store.update_if_status(job["job_id"], {"running"}, status="running", progress=0.5)
    assert overwritten is None
    assert store.get(job["job_id"])["status"] == "cancel_requested"


def test_cancel_before_worker_launch_does_not_spawn_process(tmp_path, monkeypatch):
    from service import WorkerController, JobStore, TokenStore, AppSettings
    store = JobStore(tmp_path / "jobs.sqlite3")
    controller = WorkerController(AppSettings(transcript_dir=str(tmp_path / "transcripts")), store,
                                  TokenStore(tmp_path / "token.txt"), lambda *_: None)
    source = tmp_path / "before-launch.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)
    store.update(job["job_id"], status="cancel_requested")

    def should_not_spawn(*args, **kwargs):
        raise AssertionError("cancelled job must not spawn a worker")

    monkeypatch.setattr(controller, "_ensure_server_running", should_not_spawn)
    controller._run_one_server(job)
    assert store.get(job["job_id"])["status"] == "cancelled"


def test_worker_controller_ffmpeg_failure(tmp_path, monkeypatch):
    from service import WorkerController, JobStore, TokenStore, AppSettings
    store = JobStore(tmp_path / "jobs.sqlite3")
    token_store = TokenStore(tmp_path / "token.txt")
    settings = AppSettings(transcript_dir=str(tmp_path / "transcripts"))

    controller = WorkerController(
        settings, store, token_store, lambda event, payload: None
    )

    source = tmp_path / "test.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)
    job_id = job["job_id"]

    stdout_lines = [
        '[1/4] Converting audio 鈫?16kHz mono WAV...',
        'ERROR: ffmpeg failed (exit 1):',
        '       ffmpeg: error while loading shared libraries',
        '@@EVENT {"event":"failed","stage":"converting","message":"ffmpeg failed","error":"ffmpeg: error while loading shared libraries"}'
    ]

    class Input:
        def __init__(self):
            self.data = ""

        def write(self, value):
            self.data += value

        def flush(self):
            pass

    class MockProc:
        def __init__(self):
            self.stdin = Input()
            self.stdout = iter(stdout_lines)

        def poll(self):
            return 1

    monkeypatch.setattr(controller, "_ensure_server_running", lambda: MockProc())

    controller._run_one_server(job)

    final_job = store.get(job_id)
    assert final_job["status"] == "failed"
    assert "ffmpeg: error while loading shared libraries" in final_job["error"]


def test_worker_controller_stop_cancels_active(tmp_path, monkeypatch):
    from service import WorkerController, JobStore, TokenStore, AppSettings
    store = JobStore(tmp_path / "jobs.sqlite3")
    token_store = TokenStore(tmp_path / "token.txt")
    settings = AppSettings(transcript_dir=str(tmp_path / "transcripts"))

    controller = WorkerController(
        settings, store, token_store, lambda event, payload: on_event(event, payload)
    )

    def on_event(event, payload):
        if event == "started":
            controller.stop()

    source = tmp_path / "test.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)
    job_id = job["job_id"]

    class Input:
        def __init__(self):
            self.data = ""

        def write(self, value):
            self.data += value

        def flush(self):
            pass

    class MockProc:
        def __init__(self):
            self.stdin = Input()
            self.stdout = iter([])

        def poll(self):
            return 0

    monkeypatch.setattr(controller, "_ensure_server_running", lambda: MockProc())

    controller._active = job
    controller._run_one_server(job)

    final_job = store.get(job_id)
    assert final_job["status"] == "cancelled"
    assert final_job["message"] == "Cancelled on exit"


def test_run_one_uses_job_options(tmp_path, monkeypatch):
    import json
    from service import WorkerController, JobStore, TokenStore, AppSettings
    store = JobStore(tmp_path / "jobs.sqlite3")
    token_store = TokenStore(tmp_path / "token.txt")
    settings = AppSettings(transcript_dir=str(tmp_path / "transcripts"), language="en", max_speakers=5)

    controller = WorkerController(
        settings, store, token_store, lambda event, payload: None
    )

    source = tmp_path / "test.mp4"
    source.write_bytes(b"data")

    options = {
        "language": "ja",
        "pipeline": "Transcribe only",
        "output_format": "txt",
        "max_speakers": "3"
    }
    job = store.create_if_new(source, options)
    from paths import transcript_path
    expected_output = transcript_path(source, Path(settings.transcript_dir), "txt")
    expected_output.parent.mkdir(parents=True, exist_ok=True)
    expected_output.write_text("plain text transcript", encoding="utf-8")

    captured = {}

    class Input:
        def __init__(self):
            self.data = ""

        def write(self, value):
            self.data += value

        def flush(self):
            pass

    class MockProc:
        def __init__(self):
            self.stdin = Input()
            captured["stdin"] = self.stdin
            self.stdout = iter([
                '@@EVENT ' + json.dumps({"event": "completed", "job_id": job["job_id"],
                                         "output_path": str(expected_output)}) + "\n",
            ])

        def poll(self):
            return 0

    monkeypatch.setattr(controller, "_ensure_server_running", lambda: MockProc())

    controller._active = job
    controller._run_one_server(job)

    task = json.loads(captured["stdin"].data)
    assert task["language"] == "ja"
    assert task["transcribe_only"] is True
    assert task["diarize_only"] is False
    assert task["max_speakers"] == "3"
    assert task["output_format"] == "txt"
    assert store.get(job["job_id"])["status"] == "completed"


def test_run_one_defaults_for_auto_watch(tmp_path, monkeypatch):
    import json
    from service import WorkerController, JobStore, TokenStore, AppSettings
    store = JobStore(tmp_path / "jobs.sqlite3")
    token_store = TokenStore(tmp_path / "token.txt")
    settings = AppSettings(transcript_dir=str(tmp_path / "transcripts"), language="en", max_speakers=5)

    controller = WorkerController(
        settings, store, token_store, lambda event, payload: None
    )

    source = tmp_path / "test.mp4"
    source.write_bytes(b"data")

    job = store.create_if_new(source)
    from paths import transcript_path
    expected_output = transcript_path(source, Path(settings.transcript_dir), "md")
    expected_output.parent.mkdir(parents=True, exist_ok=True)
    expected_output.write_text("transcript", encoding="utf-8")

    captured = {}

    class Input:
        def __init__(self):
            self.data = ""

        def write(self, value):
            self.data += value

        def flush(self):
            pass

    class MockProc:
        def __init__(self):
            self.stdin = Input()
            captured["stdin"] = self.stdin
            self.stdout = iter([
                '@@EVENT ' + json.dumps({"event": "completed", "job_id": job["job_id"],
                                         "output_path": str(expected_output)}) + "\n",
            ])

        def poll(self):
            return 0

    monkeypatch.setattr(controller, "_ensure_server_running", lambda: MockProc())

    controller._active = job
    controller._run_one_server(job)

    task = json.loads(captured["stdin"].data)
    assert task["language"] == "en"
    assert task["transcribe_only"] is False
    assert task["diarize_only"] is False
    assert task["max_speakers"] == 5
    assert task["output_format"] == "md"
    assert store.get(job["job_id"])["status"] == "completed"


def test_success_exit_without_output_is_failed(tmp_path, monkeypatch):
    import json
    from service import WorkerController, JobStore, TokenStore, AppSettings
    store = JobStore(tmp_path / "jobs.sqlite3")
    settings = AppSettings(transcript_dir=str(tmp_path / "transcripts"))
    controller = WorkerController(settings, store, TokenStore(tmp_path / "token.txt"), lambda *_: None)
    source = tmp_path / "missing-output.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)

    class Input:
        def __init__(self):
            self.data = ""

        def write(self, value):
            self.data += value

        def flush(self):
            pass

    class MockProc:
        def __init__(self):
            self.stdin = Input()
            self.stdout = iter([
                '@@EVENT ' + json.dumps({"event": "completed", "job_id": job["job_id"]}) + "\n",
            ])

        def poll(self):
            return 0

    monkeypatch.setattr(controller, "_ensure_server_running", lambda: MockProc())
    controller._run_one_server(job)
    current = store.get(job["job_id"])
    assert current["status"] == "failed"
    assert "output file is missing or empty" in current["error"]


def test_worker_spawn_failure_preserves_original_error(tmp_path, monkeypatch):
    from service import WorkerController, JobStore, TokenStore, AppSettings
    store = JobStore(tmp_path / "jobs.sqlite3")
    controller = WorkerController(
        AppSettings(transcript_dir=str(tmp_path / "transcripts")),
        store,
        TokenStore(tmp_path / "token.txt"),
        lambda *_: None,
    )
    source = tmp_path / "spawn-failure.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)

    def fail_to_spawn(*args, **kwargs):
        raise OSError("worker launch denied")

    monkeypatch.setattr(controller, "_ensure_server_running", fail_to_spawn)
    controller._run_one_server(job)
    current = store.get(job["job_id"])
    assert current["status"] == "failed"
    assert current["error"] == "worker launch denied"


def test_missing_source_during_output_fallback_is_failed(tmp_path, monkeypatch):
    import json
    from service import WorkerController, JobStore, TokenStore, AppSettings
    store = JobStore(tmp_path / "jobs.sqlite3")
    controller = WorkerController(
        AppSettings(transcript_dir=str(tmp_path / "transcripts")),
        store,
        TokenStore(tmp_path / "token.txt"),
        lambda *_: None,
    )
    source = tmp_path / "removed-source.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)

    def stdout_lines():
        source.unlink()
        yield '@@EVENT ' + json.dumps({"event": "completed", "job_id": job["job_id"]}) + "\n"

    class Input:
        def __init__(self):
            self.data = ""

        def write(self, value):
            self.data += value

        def flush(self):
            pass

    class MockProc:
        def __init__(self):
            self.stdin = Input()
            self.stdout = iter(stdout_lines())

        def poll(self):
            return 0

    monkeypatch.setattr(controller, "_ensure_server_running", lambda: MockProc())
    controller._run_one_server(job)
    current = store.get(job["job_id"])
    assert current["status"] == "failed"
    assert "Could not resolve expected output path" in current["error"]


def test_create_if_new_reruns_completed_job(tmp_path):
    import json
    from service import JobStore
    store = JobStore(tmp_path / "jobs.sqlite3")
    source = tmp_path / "test.mp4"
    source.write_bytes(b"data")

    job = store.create_if_new(source, {"language": "en"})
    job_id = job["job_id"]
    store.update(job_id, status="completed", progress=1.0, output_path="out.md", retry_count=3)

    rerun = store.create_if_new(source, {"language": "ja", "pipeline": "Transcribe only"})
    assert rerun is not None
    assert rerun["job_id"] == job_id
    assert rerun["status"] == "queued"
    assert rerun["progress"] == 0.0
    assert rerun["output_path"] == ""
    assert rerun["retry_count"] == 0

    options = json.loads(rerun["options_json"])
    assert options["language"] == "ja"
    assert options["pipeline"] == "Transcribe only"


def test_terminal_state_lock(tmp_path):
    from service import WorkerController, JobStore, TokenStore, AppSettings
    store = JobStore(tmp_path / "jobs.sqlite3")
    token_store = TokenStore(tmp_path / "token.txt")
    settings = AppSettings(transcript_dir=str(tmp_path / "transcripts"))

    controller = WorkerController(
        settings, store, token_store, lambda event, payload: None
    )

    source = tmp_path / "test.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)
    job_id = job["job_id"]

    store.update(job_id, status="completed", progress=1.0)
    controller._apply_worker_event(job_id, {"event": "progress", "progress": 0.5})

    updated_job = store.get(job_id)
    assert updated_job["status"] == "completed"
    assert updated_job["progress"] == 1.0


def test_warning_does_not_stop_progress(tmp_path):
    from service import WorkerController, JobStore, TokenStore, AppSettings
    store = JobStore(tmp_path / "jobs.sqlite3")
    controller = WorkerController(AppSettings(transcript_dir=str(tmp_path / "transcripts")), store,
                                  TokenStore(tmp_path / "token.txt"), lambda *_: None)
    source = tmp_path / "warning.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)
    store.update(job["job_id"], status="running")
    controller._apply_worker_event(job["job_id"], {"event": "warning", "message": "No token"})
    controller._apply_worker_event(job["job_id"], {"event": "progress", "progress": 0.5})
    current = store.get(job["job_id"])
    assert current["status"] == "running"
    assert current["progress"] == 0.5
    assert current["error"] == "No token"


def test_completed_saves_worker_output_path(tmp_path):
    from service import WorkerController, JobStore, TokenStore, AppSettings
    store = JobStore(tmp_path / "jobs.sqlite3")
    token_store = TokenStore(tmp_path / "token.txt")
    settings = AppSettings(transcript_dir=str(tmp_path / "transcripts"))

    controller = WorkerController(
        settings, store, token_store, lambda event, payload: None
    )

    source = tmp_path / "test.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)
    job_id = job["job_id"]
    store.update(job_id, status="running")

    controller._apply_worker_event(job_id, {
        "event": "completed",
        "stage": "completed",
        "output_path": "C:\\My\\Output.md"
    })

    updated_job = store.get(job_id)
    assert updated_job["status"] == "running"
    assert updated_job["output_path"] == "C:\\My\\Output.md"


def test_watcher_lifecycle_controls(tmp_path, monkeypatch):
    from service import BackgroundService, AppSettings
    watcher_calls = {"started": 0, "stopped": 0}

    from service import FileWatcher
    monkeypatch.setattr(FileWatcher, "start", lambda self: watcher_calls.update({"started": watcher_calls["started"] + 1}))
    monkeypatch.setattr(FileWatcher, "stop", lambda self: watcher_calls.update({"stopped": watcher_calls["stopped"] + 1}))

    monkeypatch.setattr("service.SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr("service.JOBS_DB", tmp_path / "jobs.sqlite3")
    monkeypatch.setattr("service.TOKEN_FILE", tmp_path / "token.txt")

    settings = AppSettings(watch_dir=str(tmp_path / "watch"), transcript_dir=str(tmp_path / "transcripts"))
    service = BackgroundService(settings=settings)

    service.set_watcher_enabled(False)
    assert service.settings.watcher_enabled is False
    assert watcher_calls["stopped"] == 1

    service.set_watcher_enabled(True)
    assert service.settings.watcher_enabled is True
    assert watcher_calls["started"] == 1

    new_watch_path = tmp_path / "watch_new"
    new_watch_path.mkdir(exist_ok=True)
    service.set_watch_dir(new_watch_path)
    assert service.settings.watch_dir == str(new_watch_path)
    assert watcher_calls["stopped"] == 2
    assert watcher_calls["started"] == 2


def test_background_service_uses_runtime_settings_path(tmp_path, monkeypatch):
    from service import BackgroundService, AppSettings

    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr("service.SETTINGS_FILE", settings_path)
    monkeypatch.setattr("service.JOBS_DB", tmp_path / "jobs.sqlite3")
    monkeypatch.setattr("service.TOKEN_FILE", tmp_path / "token.txt")

    settings = AppSettings(
        watch_dir=str(tmp_path / "watch"),
        transcript_dir=str(tmp_path / "transcripts"),
    )
    BackgroundService(settings=settings)

    assert settings_path.exists()
    import json
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved["transcript_dir"] == str(tmp_path / "transcripts")


def test_service_restart_recovery(tmp_path, monkeypatch):
    from service import BackgroundService, JobStore, AppSettings
    from service import FileWatcher
    monkeypatch.setattr(FileWatcher, "start", lambda self: None)
    monkeypatch.setattr(FileWatcher, "stop", lambda self: None)

    db_path = tmp_path / "jobs.sqlite3"
    monkeypatch.setattr("service.SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr("service.JOBS_DB", db_path)
    monkeypatch.setattr("service.TOKEN_FILE", tmp_path / "token.txt")

    store = JobStore(db_path)
    source1 = tmp_path / "s1.mp4"
    source1.write_bytes(b"data1")
    source2 = tmp_path / "s2.mp4"
    source2.write_bytes(b"data2")

    j1 = store.create_if_new(source1)
    j2 = store.create_if_new(source2)
    store.update(j2["job_id"], status="running", retry_count=1)

    settings = AppSettings(watch_dir=str(tmp_path / "watch"), transcript_dir=str(tmp_path / "transcripts"))
    service = BackgroundService(settings=settings)

    assert len(service.worker._queue) == 0
    service.start()

    assert store.get(j2["job_id"])["status"] == "queued"
    assert store.get(j2["job_id"])["retry_count"] == 2

    assert len(service.worker._queue) == 2
    queue_ids = {j["job_id"] for j in service.worker._queue}
    assert queue_ids == {j1["job_id"], j2["job_id"]}

    service.stop()


def test_filename_hash_deduplication(tmp_path):
    from transcribe import derive_paths

    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    dir2 = tmp_path / "dir2"
    dir2.mkdir()

    f1 = dir1 / "meeting.mp4"
    f1.write_bytes(b"data")
    f2 = dir2 / "meeting.mp4"
    f2.write_bytes(b"data")

    paths1 = derive_paths(f1)
    paths2 = derive_paths(f2)

    assert paths1["wav"] != paths2["wav"]
    assert paths1["output_md"] != paths2["output_md"]
    assert "meeting_" in paths1["wav"].name
    assert "meeting_" in paths2["wav"].name


def test_filename_hash_changes_when_source_changes(tmp_path):
    from transcribe import derive_paths
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"first")
    first = derive_paths(source)
    source.write_bytes(b"second-content")
    second = derive_paths(source)
    assert first["wav"] != second["wav"]


def test_cancel_requested_is_recovered_as_cancelled(tmp_path):
    from service import JobStore
    store = JobStore(tmp_path / "jobs.sqlite3")
    source = tmp_path / "cancelled.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)
    store.update(job["job_id"], status="cancel_requested")
    assert store.recover_interrupted() == 0
    current = store.get(job["job_id"])
    assert current["status"] == "cancelled"


def test_queued_query_has_no_hundred_job_cap(tmp_path):
    from service import JobStore
    store = JobStore(tmp_path / "jobs.sqlite3")
    for index in range(105):
        source = tmp_path / f"queued-{index}.mp4"
        source.write_bytes(str(index).encode())
        assert store.create_if_new(source) is not None
    assert len(store.queued()) == 105


def test_windows_process_tree_uses_taskkill(monkeypatch):
    import service
    calls = []

    class Proc:
        pid = 4321

        def poll(self):
            return None

        def terminate(self):
            raise AssertionError("taskkill success should not fall back to terminate")

    class Result:
        returncode = 0

    monkeypatch.setattr(service.os, "name", "nt")
    monkeypatch.setattr(service.subprocess, "run", lambda command, **kwargs: calls.append(command) or Result())
    service._terminate_process_tree(Proc())
    assert calls == [["taskkill", "/PID", "4321", "/T", "/F"]]


def test_run_loop_always_uses_persistent_server(tmp_path, monkeypatch):
    from service import WorkerController

    store = JobStore(tmp_path / "jobs.sqlite3")
    controller = WorkerController(
        AppSettings(transcript_dir=str(tmp_path / "transcripts")),
        store,
        TokenStore(tmp_path / "token.txt"),
        lambda *_: None,
    )
    source = tmp_path / "production-path.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)
    calls = []
    monkeypatch.setattr(controller, "_run_one_server", lambda row: calls.append(row["job_id"]))
    controller._queue.append(job)

    deadline = time.time() + 5
    while not calls and time.time() < deadline:
        time.sleep(0.05)
    controller.stop()
    assert calls == [job["job_id"]]


def test_persistent_server_job_protocol_and_stale_event_filter(tmp_path, monkeypatch):
    import json
    from paths import transcript_path
    from service import WorkerController

    store = JobStore(tmp_path / "jobs.sqlite3")
    settings = AppSettings(transcript_dir=str(tmp_path / "transcripts"))
    logs = []
    controller = WorkerController(
        settings,
        store,
        TokenStore(tmp_path / "token.txt"),
        lambda event, payload: logs.append((event, payload)),
    )
    source = tmp_path / "server-job.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source, {"num_speakers": "3", "hotwords": "Alice,vLLM"})
    output = transcript_path(source, Path(settings.transcript_dir), "md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("transcript", encoding="utf-8")

    class Input:
        def __init__(self):
            self.data = ""

        def write(self, value):
            self.data += value

        def flush(self):
            pass

    class Proc:
        def __init__(self):
            self.stdin = Input()
            self.stdout = iter([
                '@@EVENT {"event":"failed","job_id":"old-job","error":"stale"}\n',
                '@@EVENT ' + json.dumps({
                    "event": "completed", "job_id": job["job_id"],
                    "output_path": str(output),
                }) + "\n",
            ])

        def poll(self):
            return None

    proc = Proc()
    monkeypatch.setattr(controller, "_ensure_server_running", lambda: proc)

    controller._run_one_server(job)

    task = json.loads(proc.stdin.data)
    assert task["command"] == "transcribe"
    assert task["job_id"] == job["job_id"]
    assert task["num_speakers"] == "3"
    assert task["hotwords"] == "Alice,vLLM"
    assert store.get(job["job_id"])["status"] == "completed"
    assert any("Ignored stale worker event" in payload.get("message", "") for _, payload in logs)
    assert sum(event == "completed" for event, _ in logs) == 1
    controller.stop()


def test_persistent_server_warning_becomes_completed_with_warning(tmp_path, monkeypatch):
    import json
    from paths import transcript_path
    from service import WorkerController

    store = JobStore(tmp_path / "jobs.sqlite3")
    settings = AppSettings(transcript_dir=str(tmp_path / "transcripts"))
    logs = []
    controller = WorkerController(
        settings, store, TokenStore(tmp_path / "token.txt"),
        lambda event, payload: logs.append((event, payload)),
    )
    source = tmp_path / "warned-job.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)
    output = transcript_path(source, Path(settings.transcript_dir), "md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("transcript", encoding="utf-8")

    class Input:
        def __init__(self):
            self.data = ""

        def write(self, value):
            self.data += value

        def flush(self):
            pass

    class Proc:
        def __init__(self):
            self.stdin = Input()
            self.stdout = iter([
                '@@EVENT ' + json.dumps({
                    "event": "warning", "job_id": job["job_id"],
                    "message": "model download was slow",
                }) + "\n",
                '@@EVENT ' + json.dumps({
                    "event": "completed", "job_id": job["job_id"],
                    "output_path": str(output),
                }) + "\n",
            ])

        def poll(self):
            return 0

    monkeypatch.setattr(controller, "_ensure_server_running", lambda: Proc())

    controller._run_one_server(job)

    current = store.get(job["job_id"])
    assert current["status"] == "completed_with_warning"
    assert current["error"] == "model download was slow"
    assert any(event == "completed_with_warning" for event, _ in logs)
    controller.stop()


def test_terminal_event_not_overridden_by_exit_code(tmp_path, monkeypatch):
    import json
    from paths import transcript_path
    from service import WorkerController

    store = JobStore(tmp_path / "jobs.sqlite3")
    settings = AppSettings(transcript_dir=str(tmp_path / "transcripts"))
    controller = WorkerController(
        settings, store, TokenStore(tmp_path / "token.txt"), lambda *_: None,
    )
    source = tmp_path / "killed-after-complete.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)
    output = transcript_path(source, Path(settings.transcript_dir), "md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("transcript", encoding="utf-8")

    class Input:
        def __init__(self):
            self.data = ""

        def write(self, value):
            self.data += value

        def flush(self):
            pass

    class Proc:
        def __init__(self):
            self.stdin = Input()
            self.stdout = iter([
                '@@EVENT ' + json.dumps({
                    "event": "completed", "job_id": job["job_id"],
                    "output_path": str(output),
                }) + "\n",
            ])

        def poll(self):
            return -9

    monkeypatch.setattr(controller, "_ensure_server_running", lambda: Proc())

    controller._run_one_server(job)

    current = store.get(job["job_id"])
    assert current["status"] == "completed"
    controller.stop()


def test_server_startup_requires_ready_confirmation(tmp_path, monkeypatch):
    import pytest
    import service
    from service import WorkerController

    controller = WorkerController(
        AppSettings(transcript_dir=str(tmp_path / "transcripts")),
        JobStore(tmp_path / "jobs.sqlite3"),
        TokenStore(tmp_path / "token.txt"),
        lambda *_: None,
    )

    class Proc:
        pid = None

        def __init__(self, *args, **kwargs):
            self.stdout = iter(["startup failed\n"])

        def poll(self):
            return 2

        def terminate(self):
            pass

    monkeypatch.setattr(service.subprocess, "Popen", Proc)

    with pytest.raises(RuntimeError, match="before readiness confirmation"):
        controller._ensure_server_running()
    assert controller._proc is None
    controller.stop()


def test_server_eof_cannot_reuse_stale_output_as_success(tmp_path, monkeypatch):
    from paths import transcript_path
    from service import WorkerController

    store = JobStore(tmp_path / "jobs.sqlite3")
    settings = AppSettings(transcript_dir=str(tmp_path / "transcripts"))
    controller = WorkerController(
        settings,
        store,
        TokenStore(tmp_path / "token.txt"),
        lambda *_: None,
    )
    source = tmp_path / "stale-output.mp4"
    source.write_bytes(b"data")
    job = store.create_if_new(source)
    old_output = transcript_path(source, Path(settings.transcript_dir), "md")
    old_output.parent.mkdir(parents=True, exist_ok=True)
    old_output.write_text("old transcript", encoding="utf-8")

    class Input:
        def write(self, value):
            pass

        def flush(self):
            pass

    class Proc:
        stdin = Input()
        stdout = iter([])

        def poll(self):
            return 0

    monkeypatch.setattr(controller, "_ensure_server_running", lambda: Proc())

    controller._run_one_server(job)

    current = store.get(job["job_id"])
    assert current["status"] == "failed"
    assert current["error"] == "Worker connection closed before a terminal event"
    controller.stop()


def test_server_startup_timeout_terminates_process(tmp_path, monkeypatch):
    import threading
    import pytest
    import service
    from service import WorkerController

    controller = WorkerController(
        AppSettings(transcript_dir=str(tmp_path / "transcripts")),
        JobStore(tmp_path / "jobs.sqlite3"),
        TokenStore(tmp_path / "token.txt"),
        lambda *_: None,
    )
    released = threading.Event()

    class BlockingOutput:
        def __iter__(self):
            released.wait(timeout=1)
            return iter(())

    class Proc:
        pid = None

        def __init__(self, *args, **kwargs):
            self.stdout = BlockingOutput()
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True
            released.set()

    monkeypatch.setattr(service, "WORKER_START_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(service.subprocess, "Popen", Proc)

    with pytest.raises(TimeoutError, match="did not become ready"):
        controller._ensure_server_running()
    assert controller._proc is None
    controller.stop()


def test_watcher_sharing_violation_keeps_pending_without_premature_completion(tmp_path, monkeypatch):
    """FileWatcher must not mark a file ready if open('r+b') raises sharing violation (OBS writing)."""
    settings = _settings(tmp_path)
    events = []
    watcher = FileWatcher(settings, lambda event, payload: events.append((event, payload)))
    video = Path(settings.watch_dir) / "recording.mp4"
    video.write_bytes(b"v" * 200 * 1024)
    key = str(video.resolve())

    # Put file into pending as if it reached stable duration
    watcher._pending[key] = (video.stat().st_size, time.time() - 10)

    # Simulate OBS holding exclusive write lock (PermissionError / sharing conflict)
    orig_open = open
    lock_active = True

    def mock_open(file, mode="r", *args, **kwargs):
        if lock_active and str(Path(file).resolve()) == key and "r+b" in mode:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process.")
        return orig_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open)

    # While locked by OBS, tick_once must return empty and keep the file pending
    ready = watcher.tick_once()
    assert ready == []
    assert key in watcher._pending
    assert key not in watcher._known
    assert not any(event == "ready" for event, _ in events)

    # When OBS finishes and releases the lock, tick_once must recognize it once stable_seconds pass
    lock_active = False
    ready = watcher.tick_once(now=time.time() + 10)
    assert ready == [video.resolve()]
    assert key not in watcher._pending
    assert key in watcher._known
    assert any(event == "ready" for event, _ in events)


def test_watcher_readonly_file_is_not_stalled(tmp_path, monkeypatch):
    """FileWatcher must not stall read-only files whose open('r+b') fails with winerror 5 (Access Denied) but open('rb') succeeds."""
    settings = _settings(tmp_path)
    events = []
    watcher = FileWatcher(settings, lambda event, payload: events.append((event, payload)))
    video = Path(settings.watch_dir) / "readonly_recording.mp4"
    video.write_bytes(b"v" * 200 * 1024)
    key = str(video.resolve())

    watcher._pending[key] = (video.stat().st_size, time.time() - 10)

    orig_open = open

    def mock_open(file, mode="r", *args, **kwargs):
        if str(Path(file).resolve()) == key and "r+b" in mode:
            err = PermissionError(13, "Access is denied")
            err.winerror = 5
            raise err
        return orig_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open)

    ready = watcher.tick_once()
    assert ready == [video.resolve()]
    assert key not in watcher._pending
    assert key in watcher._known
    assert any(event == "ready" for event, _ in events)


def test_watch_extensions_includes_ts_flac_aac_opus():
    """WATCH_EXTENSIONS must include new video/audio formats (.ts, .flac, .aac, .opus)."""
    import config
    for ext in [".ts", ".flac", ".aac", ".opus"]:
        assert ext in config.WATCH_EXTENSIONS

