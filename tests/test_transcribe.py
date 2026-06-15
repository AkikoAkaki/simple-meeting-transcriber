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


def test_generate_srt_basic():
    """generate_srt produces valid SRT with correct block structure."""
    import transcribe
    segments = [
        {"start": 0.0, "end": 3.5, "text": "Hello world", "speaker": "SPEAKER_A"},
        {"start": 4.1, "end": 7.0, "text": "How are you?", "speaker": "SPEAKER_B"},
    ]
    srt = transcribe.generate_srt(segments)
    lines = srt.strip().split("\n")
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:03,500"
    assert lines[2] == "Hello world"
    assert lines[3] == ""
    assert lines[4] == "2"
    assert lines[5] == "00:00:04,100 --> 00:00:07,000"
    assert lines[6] == "How are you?"


def test_generate_srt_timestamp_format():
    """SRT timestamps use comma as decimal separator and HH:MM:SS,mmm format."""
    import transcribe
    segments = [{"start": 3661.5, "end": 3665.123, "text": "Test", "speaker": "S"}]
    srt = transcribe.generate_srt(segments)
    assert "01:01:01,500 --> 01:01:05,123" in srt


def test_generate_srt_empty():
    import transcribe
    assert transcribe.generate_srt([]) == ""
