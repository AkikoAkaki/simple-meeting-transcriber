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


def test_run_whisper_serializes_word_timestamps_and_hotwords(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import transcribe

    class FakeModel:
        def transcribe(self, _path, **kwargs):
            assert kwargs["word_timestamps"] is True
            assert kwargs["hotwords"] == "Alice,vLLM"
            segment = SimpleNamespace(
                start=0.0,
                end=2.0,
                text=" Alice uses vLLM",
                words=[
                    SimpleNamespace(start=0.0, end=0.8, word=" Alice"),
                    SimpleNamespace(start=1.0, end=2.0, word=" uses vLLM"),
                ],
            )
            return iter([segment]), SimpleNamespace(
                duration=2.0, language="en", language_probability=1.0)

    monkeypatch.setattr(transcribe, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(transcribe, "get_whisper_model", lambda *args: FakeModel())
    cache = tmp_path / "whisper.json"
    result = transcribe.run_whisper(
        tmp_path / "audio.wav", cache, None, "Alice,vLLM", "test-key")

    assert result[0]["words"][1]["word"] == " uses vLLM"
    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert payload["schema_version"] == transcribe.CACHE_SCHEMA_VERSION
    assert payload["cache_key"] == "test-key"


def test_convert_to_wav_publishes_only_complete_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import transcribe

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "audio.wav"

    def fake_run(command, **kwargs):
        partial = Path(command[-1])
        assert partial.name == "audio.part.wav"
        partial.write_bytes(b"w" * 2048)
        assert not output.exists()
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(transcribe.subprocess, "run", fake_run)
    transcribe.convert_to_wav(source, output)
    assert output.stat().st_size == 2048
    assert not (tmp_path / "audio.part.wav").exists()


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


def test_merge_results_two_pointer():
    """merge_results correctly matches speakers using the two-pointer approach."""
    import transcribe
    whisper_segments = [
        {"start": 1.0, "end": 5.0, "text": "Segment one"},
        {"start": 6.0, "end": 10.0, "text": "Segment two"},
    ]
    speaker_turns = [
        {"start": 0.0, "end": 4.5, "speaker": "SPEAKER_A"},
        {"start": 5.5, "end": 11.0, "speaker": "SPEAKER_B"},
    ]
    result = transcribe.merge_results(whisper_segments, speaker_turns)
    assert len(result) == 2
    assert result[0]["speaker"] == "SPEAKER_A"
    assert result[1]["speaker"] == "SPEAKER_B"


def test_merge_results_consecutive_same_speaker():
    import transcribe
    whisper_segments = [
        {"start": 1.0, "end": 5.0, "text": "Hello"},
        {"start": 6.0, "end": 10.0, "text": "world"},
    ]
    speaker_turns = [
        {"start": 0.0, "end": 12.0, "speaker": "SPEAKER_A"},
    ]
    result = transcribe.merge_results(whisper_segments, speaker_turns)
    # Gap is 6.0 - 5.0 = 1.0s (<= 2s), same speaker SPEAKER_A. They should be merged.
    assert len(result) == 1
    assert result[0]["text"] == "Hello world"
    assert result[0]["start"] == 1.0
    assert result[0]["end"] == 10.0
    assert result[0]["speaker"] == "SPEAKER_A"


def test_merge_results_splits_words_when_speaker_changes_mid_segment():
    import transcribe

    whisper_segments = [{
        "start": 0.0,
        "end": 4.0,
        "text": "Hello world yes",
        "words": [
            {"start": 0.0, "end": 1.0, "word": "Hello"},
            {"start": 1.1, "end": 2.0, "word": " world"},
            {"start": 2.2, "end": 3.0, "word": " yes"},
        ],
    }]
    speaker_turns = [
        {"start": 0.0, "end": 2.1, "speaker": "SPEAKER_A"},
        {"start": 2.1, "end": 4.0, "speaker": "SPEAKER_B"},
    ]

    result = transcribe.merge_results(whisper_segments, speaker_turns)

    assert [(item["speaker"], item["text"]) for item in result] == [
        ("SPEAKER_A", "Hello world"),
        ("SPEAKER_B", "yes"),
    ]


def test_merge_results_preserves_original_cjk_and_subword_spacing():
    import transcribe

    whisper_segments = [{
        "start": 0.0,
        "end": 3.0,
        "text": "你又有 cost integration",
        "words": [
            {"start": 0.0, "end": 0.3, "word": "你"},
            {"start": 0.3, "end": 0.6, "word": "又"},
            {"start": 0.6, "end": 0.9, "word": "有"},
            {"start": 0.9, "end": 1.3, "word": " cost"},
            {"start": 1.3, "end": 1.8, "word": " integr"},
            {"start": 1.8, "end": 2.2, "word": "ation"},
        ],
    }]
    speaker_turns = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_A"}]

    result = transcribe.merge_results(whisper_segments, speaker_turns)

    assert len(result) == 1
    assert result[0]["text"] == whisper_segments[0]["text"]


def test_merge_results_separates_adjacent_whisper_segments():
    import transcribe

    whisper_segments = [
        {
            "start": 0.0,
            "end": 0.8,
            "text": "甲",
            "words": [{"start": 0.0, "end": 0.8, "word": "甲"}],
        },
        {
            "start": 0.9,
            "end": 1.7,
            "text": "乙",
            "words": [{"start": 0.9, "end": 1.7, "word": "乙"}],
        },
    ]
    speaker_turns = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A"}]

    result = transcribe.merge_results(whisper_segments, speaker_turns)

    assert result[0]["text"] == "甲 乙"


def test_merge_results_falls_back_without_dropping_segments_missing_words():
    import transcribe

    whisper_segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "有时间戳",
            "words": [{"start": 0.0, "end": 1.0, "word": "有时间戳"}],
        },
        {"start": 1.2, "end": 2.0, "text": "没有时间戳"},
    ]
    speaker_turns = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_A"}]

    result = transcribe.merge_results(whisper_segments, speaker_turns)

    assert len(result) == 1
    assert "有时间戳" in result[0]["text"]
    assert "没有时间戳" in result[0]["text"]


def test_merge_results_coalesces_short_isolated_speaker_flicker():
    import transcribe

    whisper_segments = [{
        "start": 0.0,
        "end": 2.4,
        "text": "甲乙丙",
        "words": [
            {"start": 0.0, "end": 0.8, "word": "甲"},
            {"start": 0.9, "end": 1.3, "word": "乙"},
            {"start": 1.4, "end": 2.4, "word": "丙"},
        ],
    }]
    speaker_turns = [
        {"start": 0.0, "end": 0.85, "speaker": "SPEAKER_A"},
        {"start": 0.85, "end": 1.35, "speaker": "SPEAKER_B"},
        {"start": 1.35, "end": 2.5, "speaker": "SPEAKER_A"},
    ]

    result = transcribe.merge_results(whisper_segments, speaker_turns)

    assert [(item["speaker"], item["text"]) for item in result] == [
        ("SPEAKER_A", "甲乙丙"),
    ]


def test_stage_cache_rejects_changed_key(tmp_path):
    import transcribe

    path = tmp_path / "whisper.json"
    transcribe._write_stage_cache(path, "whisper", "key-a", [{"text": "old"}])

    assert transcribe._read_stage_cache(path, "whisper", "key-a") == [{"text": "old"}]
    assert transcribe._read_stage_cache(path, "whisper", "key-b") is None


def test_rename_speakers_in_segments_only_changes_display_labels():
    import transcribe

    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": "World", "speaker": "[unknown]"},
    ]
    renamed = transcribe.rename_speakers_in_segments(
        segments, {"SPEAKER_00": "Alice", "[unknown]": ""})

    assert renamed[0]["speaker"] == "Alice"
    assert renamed[1]["speaker"] == "[unknown]"
    assert segments[0]["speaker"] == "SPEAKER_00"


def test_rename_output_reuses_merged_cache(tmp_path, monkeypatch):
    import transcribe

    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "meeting.md"
    merged_cache = tmp_path / "merged.json"
    paths = {"segments_json": merged_cache, "wav": tmp_path / "audio.wav"}
    monkeypatch.setattr(transcribe, "derive_paths", lambda _path: paths)
    transcribe._write_stage_cache(
        merged_cache,
        "merged",
        "merged-key",
        [{"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00"}],
        metadata={
            "source_file": source.name,
            "total_sec": 1.0,
            "language": "en",
            "has_diarization": True,
        },
    )

    transcribe.rename_output(source, output, {"SPEAKER_00": "Alice"})

    assert "Alice" in output.read_text(encoding="utf-8")
    assert "Hello" in output.read_text(encoding="utf-8")
    payload = transcribe._read_stage_payload(merged_cache, "merged", None)
    assert payload["speaker_aliases"] == {"SPEAKER_00": "Alice"}

    regenerated = tmp_path / "meeting.txt"
    transcribe.rename_output(source, regenerated, {})
    assert regenerated.read_text(encoding="utf-8").strip() == "Hello"


def test_server_jobs_reset_max_speakers_and_end_with_terminal_event(tmp_path, monkeypatch, capsys):
    import json
    import transcribe

    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"data")
    output = tmp_path / "meeting.md"
    paths = {
        "wav": tmp_path / "meeting.wav",
        "whisper_json": tmp_path / "whisper.json",
        "diarize_json": tmp_path / "diarize.json",
        "output_md": output,
    }
    observed = []
    cleared = []
    monkeypatch.setattr(transcribe, "derive_paths", lambda _: paths)
    monkeypatch.setattr(transcribe, "convert_to_wav", lambda *_: None)
    monkeypatch.setattr(
        transcribe,
        "run_whisper",
        lambda *_: [{"start": 0.0, "end": 1.0, "text": "hello"}],
    )
    monkeypatch.setattr(
        transcribe,
        "run_diarization",
        lambda *args, **kwargs: observed.append((transcribe.config.MAX_SPEAKERS, args[-1])) or [],
    )
    monkeypatch.setattr(
        transcribe,
        "merge_results",
        lambda segments, _: [{**segments[0], "speaker": "[unknown]"}],
    )
    monkeypatch.setattr(transcribe, "get_wav_duration", lambda _: 1.0)
    monkeypatch.setattr(transcribe, "clear_diarize_pipeline", lambda: cleared.append(True))
    monkeypatch.setenv("TRANSCRIBE_JOB_ID", "job-2")

    base = {
        "input_path": str(source), "output_dir": str(tmp_path),
        "model": "tiny", "device": "cpu", "language": "auto",
        "output_format": "md",
    }
    transcribe.run_job_from_json({**base, "max_speakers": 3, "token": "secret"})
    transcribe.run_job_from_json({**base, "max_speakers": None, "token": ""})

    assert observed == [(3, "secret"), (None, "")]
    assert cleared == [True]
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    terminal = json.loads(lines[-1].removeprefix("@@EVENT "))
    assert terminal["event"] == "completed"
    assert terminal["job_id"] == "job-2"


def test_cached_diarization_pipeline_moves_back_to_cpu(monkeypatch):
    import transcribe

    moves = []

    class Pipeline:
        def to(self, device):
            moves.append(str(device))

    pipeline = Pipeline()
    monkeypatch.setattr(transcribe, "_diarize_pipeline", pipeline)
    monkeypatch.setattr(transcribe, "_diarize_token", "token")
    monkeypatch.setattr(transcribe, "_diarize_device", "cuda")

    assert transcribe.get_diarize_pipeline("token", "cpu") is pipeline
    assert moves == ["cpu"]
    assert transcribe._diarize_device == "cpu"


def test_failed_device_move_invalidates_cached_device_marker(monkeypatch):
    import pytest
    import transcribe

    class Pipeline:
        def to(self, device):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(transcribe, "_diarize_pipeline", Pipeline())
    monkeypatch.setattr(transcribe, "_diarize_token", "token")
    monkeypatch.setattr(transcribe, "_diarize_device", "cpu")

    with pytest.raises(transcribe.DiarizationDeviceError):
        transcribe.get_diarize_pipeline("token", "cuda")
    assert transcribe._diarize_device is None


def test_explicit_empty_token_does_not_fall_back_to_previous_source(tmp_path, monkeypatch):
    import transcribe

    monkeypatch.setattr(
        transcribe,
        "get_hf_token",
        lambda: (_ for _ in ()).throw(AssertionError("explicit empty token must not fall back")),
    )
    assert transcribe.run_diarization(tmp_path / "audio.wav", tmp_path / "missing.json", "") == []


def test_get_wav_duration_handles_odd_sized_chunks(tmp_path):
    import struct
    import transcribe

    sample_rate = 16000
    data = b"\0" * (sample_rate * 2)
    junk = b"x"
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
    body = (
        b"JUNK" + struct.pack("<I", len(junk)) + junk + b"\0"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", len(data)) + data
    )
    wav = tmp_path / "odd-chunk.wav"
    wav.write_bytes(b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body)

    assert transcribe.get_wav_duration(wav) == 1.0
