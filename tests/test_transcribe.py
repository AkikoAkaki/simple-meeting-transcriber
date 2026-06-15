"""Tests for pure-logic functions in transcribe.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_torch_load_not_globally_patched():
    """torch.load must not be replaced at module import time."""
    import torch
    original_load = torch.load
    # Force reimport to detect any module-level patches
    import importlib
    import transcribe
    importlib.reload(transcribe)
    assert torch.load is original_load, (
        "transcribe.py must not replace torch.load at module level; "
        "use the _allow_unsafe_torch_load() context manager instead"
    )


def test_run_whisper_deletes_corrupt_cache(tmp_path, monkeypatch):
    """A corrupt JSON cache must be deleted and transcription re-run."""
    import transcribe

    # Write a truncated JSON file
    whisper_json = tmp_path / "_test_whisper.json"
    whisper_json.write_text("{invalid", encoding="utf-8")

    # Track whether the file gets deleted
    deleted = []
    orig_unlink = Path.unlink
    def mock_unlink(self, *a, **kw):
        deleted.append(str(self))
        orig_unlink(self, *a, **kw)
    monkeypatch.setattr(Path, "unlink", mock_unlink)

    # Stub out the actual Whisper model loading so the test doesn't hang
    monkeypatch.setattr(
        transcribe, "_resolve_device", lambda: "cpu")

    class _FakeModel:
        def transcribe(self, *a, **kw):
            from types import SimpleNamespace
            return iter([]), SimpleNamespace(language="en", language_probability=1.0)

    import sys
    sys.modules.setdefault("faster_whisper", type(sys)("faster_whisper"))
    sys.modules["faster_whisper"].WhisperModel = lambda *a, **kw: _FakeModel()

    result = transcribe.run_whisper(tmp_path / "fake.wav", whisper_json, None)
    assert str(whisper_json) in deleted or not whisper_json.exists(), \
        "corrupt cache file should be deleted before re-transcribing"
