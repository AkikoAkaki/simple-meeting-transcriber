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
        assert "-nostdin" in command
        assert "-vn" in command
        assert "-sn" in command
        assert "-dn" in command
        partial = Path(command[-1])
        assert partial.name == "audio.part.wav"
        partial.write_bytes(b"w" * 2048)
        assert not output.exists()
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(transcribe.subprocess, "run", fake_run)
    transcribe.convert_to_wav(source, output)
    assert output.stat().st_size == 2048
    assert not (tmp_path / "audio.part.wav").exists()


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

    # CJK + CJK: no space
    cjk_segments = [
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
    result = transcribe.merge_results(cjk_segments, speaker_turns)
    assert result[0]["text"] == "甲乙"

    # English + English: single space
    en_segments = [
        {
            "start": 0.0,
            "end": 0.8,
            "text": "Hello",
            "words": [{"start": 0.0, "end": 0.8, "word": "Hello"}],
        },
        {
            "start": 0.9,
            "end": 1.7,
            "text": "world",
            "words": [{"start": 0.9, "end": 1.7, "word": "world"}],
        },
    ]
    result_en = transcribe.merge_results(en_segments, speaker_turns)
    assert result_en[0]["text"] == "Hello world"

    # CJK + English: single space
    mixed_segments = [
        {
            "start": 0.0,
            "end": 0.8,
            "text": "项目",
            "words": [{"start": 0.0, "end": 0.8, "word": "项目"}],
        },
        {
            "start": 0.9,
            "end": 1.7,
            "text": "roadmap",
            "words": [{"start": 0.9, "end": 1.7, "word": "roadmap"}],
        },
    ]
    result_mixed = transcribe.merge_results(mixed_segments, speaker_turns)
    assert result_mixed[0]["text"] == "项目 roadmap"


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
        lambda *args, **kwargs: observed.append((transcribe.config.MAX_SPEAKERS, args[-1])) or [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
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
        "model": "large-v3-turbo", "device": "cpu", "language": "auto",
    }
    transcribe.run_job_from_json({**base, "max_speakers": 3, "token": "secret"})
    transcribe.run_job_from_json({**base, "max_speakers": None, "token": ""})

    assert observed == [(3, "secret"), (None, "")]
    # Lifecycle: each job clears previous pyannote before Whisper (2), plus one
    # extra clear for the empty-token job that skips diarization.
    assert cleared == [True, True, True]
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
    import pytest
    with pytest.raises(transcribe.TranscriptionError, match="requires a HuggingFace token"):
        transcribe.run_diarization(tmp_path / "audio.wav", tmp_path / "missing.json", "")


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


def test_is_cjk_char_coverage():
    """_is_cjk_char must cover Hanzi, Hiragana, Katakana, Hangul, and fullwidth punctuation."""
    import transcribe

    # Chinese Hanzi
    assert transcribe._is_cjk_char("中")
    assert transcribe._is_cjk_char("文")
    # Japanese Hiragana & Katakana
    assert transcribe._is_cjk_char("あ")
    assert transcribe._is_cjk_char("ん")
    assert transcribe._is_cjk_char("ア")
    assert transcribe._is_cjk_char("ン")
    # Korean Hangul
    assert transcribe._is_cjk_char("한")
    assert transcribe._is_cjk_char("글")
    # Bopomofo & Extended CJK
    assert transcribe._is_cjk_char("ㄅ")
    assert transcribe._is_cjk_char("ㄆ")
    assert transcribe._is_cjk_char("㊀")
    assert transcribe._is_cjk_char("㌀")
    assert transcribe._is_cjk_char("㆐")
    # CJK / Fullwidth Punctuation
    assert transcribe._is_cjk_char("。")
    assert transcribe._is_cjk_char("、")
    assert transcribe._is_cjk_char("，")
    assert transcribe._is_cjk_char("！")
    assert transcribe._is_cjk_char("？")
    assert transcribe._is_cjk_char("；")
    assert transcribe._is_cjk_char("：")
    assert transcribe._is_cjk_char("（")
    assert transcribe._is_cjk_char("）")
    assert transcribe._is_cjk_char("“")
    assert transcribe._is_cjk_char("”")
    assert transcribe._is_cjk_char("—")
    assert transcribe._is_cjk_char("…")

    # Non-CJK
    assert not transcribe._is_cjk_char("a")
    assert not transcribe._is_cjk_char("Z")
    assert not transcribe._is_cjk_char("1")
    assert not transcribe._is_cjk_char(",")
    assert not transcribe._is_cjk_char(".")
    assert not transcribe._is_cjk_char(" ")
    assert not transcribe._is_cjk_char("")


def test_typography_spacing_rules():
    """Verify Chinese/CJK and Western typography rules:
    - CJK + CJK: no space
    - CJK + punct: no space
    - punct + CJK: no space
    - CJK + Western / Western + CJK: single space
    - Western + Western: single space
    """
    import transcribe

    # 中+中不加空格
    assert transcribe._join_display_text("你好", "世界") == "你好世界"
    assert transcribe._join_whisper_segment_text("今天天气", "真好") == "今天天气真好"
    # 日文 / 韩文
    assert transcribe._join_display_text("食", "べる") == "食べる"
    assert transcribe._join_display_text("안녕", "하세요") == "안녕하세요"

    # 中+标点不加空格
    assert transcribe._join_display_text("你好", "，世界") == "你好，世界"
    assert transcribe._join_display_text("很好", "。") == "很好。"
    assert transcribe._join_display_text("很好", "!") == "很好!"
    assert transcribe._join_display_text("“", "你好") == "“你好"
    assert transcribe._join_display_text("你好", "”") == "你好”"
    assert transcribe._join_display_text("（", "会议") == "（会议"
    assert transcribe._join_display_text("会议", "）") == "会议）"
    assert transcribe._join_display_text("他说", '"Hello"') == '他说"Hello"'
    assert transcribe._join_display_text('"Hello"', "他说") == '"Hello"他说'

    # 中+英 / 英+中保留单空格
    assert transcribe._join_display_text("使用", "vLLM") == "使用 vLLM"
    assert transcribe._join_display_text("vLLM", "模型") == "vLLM 模型"
    assert transcribe._join_display_text("项目", "roadmap") == "项目 roadmap"
    assert transcribe._join_display_text("roadmap", "评审") == "roadmap 评审"
    assert transcribe._join_display_text("共", "3") == "共 3"

    # 英+英保留单空格
    assert transcribe._join_display_text("Hello", "world") == "Hello world"
    assert transcribe._join_whisper_segment_text("Machine", "learning") == "Machine learning"

    # 英+标点 / 标点+英 / 引号与括号
    assert transcribe._join_display_text("Hello", ",") == "Hello,"
    assert transcribe._join_display_text("Hello,", "world") == "Hello, world"
    assert transcribe._join_display_text("(", "example") == "(example"
    assert transcribe._join_display_text("你好，", "world") == "你好，world"
    assert transcribe._join_display_text("She said", "(whispered)") == "She said (whispered)"
    assert transcribe._join_display_text("(whispered)", "she said") == "(whispered) she said"
    assert transcribe._join_display_text('"Hello,"', "she said") == '"Hello," she said'
    assert transcribe._join_display_text("She said", '"Hello"') == 'She said "Hello"'
    assert transcribe._join_display_text("She said.", "(whispered)") == "She said. (whispered)"


def test_merge_results_paragraph_fallback_cjk_spacing():
    """Paragraph-level fallback (without word timestamps) must not inject spaces between CJK segments."""
    import transcribe

    cjk_whisper_segments = [
        {"start": 0.0, "end": 1.0, "text": "第一段讨论"},
        {"start": 1.2, "end": 2.0, "text": "第二段结论"},
    ]
    turns = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]
    res = transcribe.merge_results(cjk_whisper_segments, turns)
    assert len(res) == 1
    assert res[0]["text"] == "第一段讨论第二段结论"

    en_whisper_segments = [
        {"start": 0.0, "end": 1.0, "text": "First topic"},
        {"start": 1.2, "end": 2.0, "text": "second topic"},
    ]
    res_en = transcribe.merge_results(en_whisper_segments, turns)
    assert len(res_en) == 1
    assert res_en[0]["text"] == "First topic second topic"


def test_whisper_initial_prompt_passed_for_punctuation(tmp_path, monkeypatch):
    """run_whisper must pass the correct initial_prompt per language (auto -> None)."""
    import json
    from types import SimpleNamespace
    import transcribe

    observed = []

    class FakeModel:
        def transcribe(self, _path, **kwargs):
            observed.append((kwargs.get("language"), kwargs.get("initial_prompt")))
            seg = SimpleNamespace(
                start=0.0, end=1.0, text="测试，标点。",
                words=[SimpleNamespace(start=0.0, end=1.0, word="测试，标点。")],
            )
            return iter([seg]), SimpleNamespace(duration=1.0, language="zh", language_probability=1.0)

    monkeypatch.setattr(transcribe, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(transcribe, "get_whisper_model", lambda *args: FakeModel())

    # Auto-detect (None) must not inject a Chinese or English prompt.
    transcribe.run_whisper(tmp_path / "audio.wav", tmp_path / "w1.json", None)
    assert observed[-1] == (None, None)

    # Explicit "auto" string normalizes the same way.
    transcribe.run_whisper(tmp_path / "audio.wav", tmp_path / "w1b.json", "auto")
    assert observed[-1] == (None, None)

    # Chinese forced
    transcribe.run_whisper(tmp_path / "audio.wav", tmp_path / "w0.json", "zh")
    assert observed[-1] == ("zh", "以下是普通话的会议记录，包含完整的标点符号。")

    # English forced
    transcribe.run_whisper(tmp_path / "audio.wav", tmp_path / "w2.json", "en")
    assert observed[-1] == ("en", "Here is a transcript of the meeting with complete punctuation.")

    # Japanese forced
    transcribe.run_whisper(tmp_path / "audio.wav", tmp_path / "w3.json", "ja")
    assert observed[-1] == ("ja", "これは会議の書き起こしです。句読点を含めます。")

    # Korean forced
    transcribe.run_whisper(tmp_path / "audio.wav", tmp_path / "w4.json", "ko")
    assert observed[-1] == ("ko", "다음은 회의 녹취록이며 완전한 구두점이 포함되어 있습니다.")

    # Other explicit languages must not receive a wrong-language prompt.
    transcribe.run_whisper(tmp_path / "audio.wav", tmp_path / "w5.json", "fr")
    assert observed[-1] == ("fr", None)
    transcribe.run_whisper(tmp_path / "audio.wav", tmp_path / "w6.json", "de")
    assert observed[-1] == ("de", None)


def test_select_initial_prompt_single_function():
    import transcribe

    assert transcribe.select_initial_prompt(None) is None
    assert transcribe.select_initial_prompt("auto") is None
    assert transcribe.select_initial_prompt("AUTO") is None
    assert transcribe.select_initial_prompt("zh") == "以下是普通话的会议记录，包含完整的标点符号。"
    assert transcribe.select_initial_prompt("en") == "Here is a transcript of the meeting with complete punctuation."
    assert transcribe.select_initial_prompt("ja") == "これは会議の書き起こしです。句読点を含めます。"
    assert transcribe.select_initial_prompt("ko") == "다음은 회의 녹취록이며 완전한 구두점이 포함되어 있습니다."
    assert transcribe.select_initial_prompt("fr") is None
    assert transcribe.select_initial_prompt("de") is None


def test_whisper_cache_key_includes_initial_prompt():
    import transcribe

    base = dict(model_name="large-v3-turbo", device="cpu", language="zh",
                hotwords="", word_timestamps=True)
    key_zh = transcribe._whisper_cache_key(**base)
    key_auto = transcribe._whisper_cache_key(
        model_name="large-v3-turbo", device="cpu", language=None,
        hotwords="", word_timestamps=True)
    # Auto (None prompt) must differ from zh (Chinese prompt): old biased cache misses.
    assert key_zh != key_auto

    # Same inputs with different explicit prompts must differ.
    key_a = transcribe._whisper_cache_key(
        model_name="large-v3-turbo", device="cpu", language="en",
        hotwords="", word_timestamps=True, initial_prompt="prompt-a")
    key_b = transcribe._whisper_cache_key(
        model_name="large-v3-turbo", device="cpu", language="en",
        hotwords="", word_timestamps=True, initial_prompt="prompt-b")
    assert key_a != key_b

    # Diarization cache stays reusable (no prompt leaked into it).
    dia_a = transcribe._diarization_cache_key(None, None)
    dia_b = transcribe._diarization_cache_key(None, None)
    assert dia_a == dia_b


def test_run_whisper_emits_preview_in_progress_events(tmp_path, monkeypatch):
    """run_whisper must include preview in the emitted progress events."""
    from types import SimpleNamespace
    import transcribe

    emitted_events = []

    def mock_emit_event(event, **payload):
        emitted_events.append((event, payload))

    monkeypatch.setattr(transcribe, "_emit_event", mock_emit_event)
    monkeypatch.setattr(transcribe, "_resolve_device", lambda: "cpu")

    class FakeModel:
        def transcribe(self, _path, **kwargs):
            seg1 = SimpleNamespace(
                start=0.0, end=2.0, text="First transcribed segment",
                words=[SimpleNamespace(start=0.0, end=2.0, word="First transcribed segment")],
            )
            seg2 = SimpleNamespace(
                start=2.0, end=4.0, text="Second transcribed segment",
                words=[SimpleNamespace(start=2.0, end=4.0, word="Second transcribed segment")],
            )
            return iter([seg1, seg2]), SimpleNamespace(duration=4.0, language="en", language_probability=1.0)

    monkeypatch.setattr(transcribe, "get_whisper_model", lambda *args: FakeModel())

    cache = tmp_path / "test_preview_whisper.json"
    transcribe.run_whisper(tmp_path / "audio.wav", cache, None)

    progress_events = [payload for event, payload in emitted_events if event == "progress"]
    assert len(progress_events) >= 1
    assert any(p.get("preview") == "First transcribed segment" for p in progress_events)


    # Also verify MODEL_SIZES has large-v3-turbo hint
    # (Notice: in run_whisper, MODEL_SIZES is defined or cached)
    # Check that config.py default is large-v3-turbo


def test_worker_rejects_unsupported_model(tmp_path):
    import pytest
    import transcribe

    source = tmp_path / "m.mp4"
    source.write_bytes(b"data")
    with pytest.raises(ValueError, match="Unsupported model"):
        transcribe.run_job_from_json({
            "input_path": str(source), "output_dir": str(tmp_path),
            "model": "medium", "device": "cpu",
        })
    with pytest.raises(ValueError, match="Unsupported model"):
        transcribe.run_job_from_json({
            "input_path": str(source), "output_dir": str(tmp_path),
            "model": "tiny", "device": "cpu",
        })


def test_cli_model_choices_come_from_config():
    import pytest
    import config
    import transcribe

    assert list(config.SUPPORTED_MODELS) == ["large-v3-turbo", "large-v3"]
    parser = transcribe._build_cli_parser()
    for action in parser._actions:
        if "--model" in getattr(action, "option_strings", []):
            assert list(action.choices) == list(config.SUPPORTED_MODELS)
            break
    else:
        raise AssertionError("--model argument missing")

    with pytest.raises(SystemExit):
        parser.parse_args(["input.mp4", "--model", "medium"])
    args = parser.parse_args(["input.mp4", "--model", "large-v3-turbo"])
    assert args.model == "large-v3-turbo"
    args = parser.parse_args(["input.mp4", "--model", "large-v3"])
    assert args.model == "large-v3"


def test_full_pipeline_model_lifecycle_order(tmp_path, monkeypatch):
    """Two full pipelines must never hold previous pyannote + current Whisper together."""
    import transcribe

    calls = []

    def _paths_for(input_path):
        from pathlib import Path
        stem = Path(input_path).stem
        return {
            "wav": tmp_path / f"{stem}.wav",
            "whisper_json": tmp_path / f"{stem}_whisper.json",
            "diarize_json": tmp_path / f"{stem}_diarize.json",
            "segments_json": tmp_path / f"{stem}_segments.json",
            "output_md": tmp_path / f"{stem}.md",
        }

    monkeypatch.setattr(transcribe, "derive_paths", _paths_for)
    monkeypatch.setattr(transcribe, "convert_to_wav", lambda *_: None)
    monkeypatch.setattr(transcribe, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(transcribe, "get_wav_duration", lambda _: 1.0)
    monkeypatch.setattr(transcribe, "merge_results",
                         lambda segs, turns: [{**segs[0], "speaker": "SPEAKER_00"}])
    monkeypatch.setattr(transcribe, "generate_markdown", lambda *a, **k: "# md")
    monkeypatch.setattr(transcribe, "clear_diarize_pipeline",
                         lambda: calls.append("clear_diarize"))
    monkeypatch.setattr(transcribe, "_release_whisper_before_diarization",
                         lambda: calls.append("release_whisper"))
    monkeypatch.setattr(transcribe, "run_whisper",
                         lambda *a, **k: calls.append("run_whisper") or [{"start": 0.0, "end": 1.0, "text": "hi"}])
    monkeypatch.setattr(transcribe, "run_diarization",
                         lambda *a, **k: calls.append("run_diarize") or [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}])

    s1 = tmp_path / "job1.mp4"
    s1.write_bytes(b"1")
    s2 = tmp_path / "job2.mp4"
    s2.write_bytes(b"2")

    transcribe._run_job_impl(s1, "large-v3-turbo", None, None, None, False, False, "", "tok")
    transcribe._run_job_impl(s2, "large-v3-turbo", None, None, None, False, False, "", "tok")

    assert calls == [
        "clear_diarize", "run_whisper", "release_whisper", "run_diarize",
        "clear_diarize", "run_whisper", "release_whisper", "run_diarize",
    ]
    # Second job frees previous pyannote before loading current Whisper,
    # and frees Whisper before loading pyannote.
    first_diarize = calls.index("run_diarize")
    second_clear = calls.index("clear_diarize", first_diarize + 1)
    second_whisper = calls.index("run_whisper", second_clear + 1)
    assert first_diarize < second_clear < second_whisper


def test_transcribe_only_reuses_whisper(tmp_path, monkeypatch):
    """Consecutive transcribe-only jobs with the same model load Whisper once."""
    import sys
    import transcribe

    loads = []

    class FakeWhisper:
        def __init__(self, *args, **kwargs):
            loads.append((args, kwargs))

        def transcribe(self, *_a, **_k):
            from types import SimpleNamespace
            seg = SimpleNamespace(start=0.0, end=1.0, text="hi",
                                  words=[SimpleNamespace(start=0.0, end=1.0, word="hi")])
            return iter([seg]), SimpleNamespace(duration=1.0, language="en", language_probability=1.0)

    mod = type(sys)("faster_whisper")
    mod.WhisperModel = FakeWhisper
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    monkeypatch.setattr(transcribe, "_whisper_model", None)
    monkeypatch.setattr(transcribe, "_whisper_model_params", None)
    monkeypatch.setattr(transcribe, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(transcribe, "convert_to_wav", lambda *_: None)
    monkeypatch.setattr(transcribe, "get_wav_duration", lambda _: 1.0)
    monkeypatch.setattr(transcribe, "merge_results",
                         lambda segs, turns: [{**segs[0], "speaker": "[unknown]"}])
    monkeypatch.setattr(transcribe, "generate_markdown", lambda *a, **k: "# md")
    # Ensure no leftover pyannote interferes.
    monkeypatch.setattr(transcribe, "clear_diarize_pipeline", lambda: None)
    real_release_calls = []

    def _counting_release():
        real_release_calls.append(True)
        transcribe.clear_whisper_model()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    monkeypatch.setattr(transcribe, "_release_whisper_before_diarization", _counting_release)

    s1 = tmp_path / "t1.mp4"
    s1.write_bytes(b"1")
    s2 = tmp_path / "t2.mp4"
    s2.write_bytes(b"2")

    def _paths_for(input_path):
        from pathlib import Path
        stem = Path(input_path).stem
        return {
            "wav": tmp_path / f"{stem}.wav",
            "whisper_json": tmp_path / f"{stem}_whisper.json",
            "diarize_json": tmp_path / f"{stem}_diarize.json",
            "segments_json": tmp_path / f"{stem}_segments.json",
            "output_md": tmp_path / f"{stem}.md",
        }
    monkeypatch.setattr(transcribe, "derive_paths", _paths_for)

    transcribe._run_job_impl(s1, "large-v3-turbo", None, None, None, True, False, "", "")
    transcribe._run_job_impl(s2, "large-v3-turbo", None, None, None, True, False, "", "")
    assert len(loads) == 1, f"expected single Whisper load, got {len(loads)}"
    assert real_release_calls == []


def test_diarize_only_does_not_load_whisper(tmp_path, monkeypatch):
    import transcribe

    def _fail_load(*_a, **_k):
        raise AssertionError("diarize-only must not load Whisper")

    monkeypatch.setattr(transcribe, "get_whisper_model", _fail_load)
    monkeypatch.setattr(transcribe, "derive_paths", lambda _p: {
        "wav": tmp_path / "w.wav",
        "whisper_json": tmp_path / "w_whisper.json",
        "diarize_json": tmp_path / "w_diarize.json",
        "segments_json": tmp_path / "w_segments.json",
        "output_md": tmp_path / "w.md",
    })
    monkeypatch.setattr(transcribe, "convert_to_wav", lambda *_: None)
    monkeypatch.setattr(transcribe, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(transcribe, "_read_stage_cache",
                         lambda *_a, **_k: [{"start": 0.0, "end": 1.0, "text": "hi"}])
    monkeypatch.setattr(transcribe, "run_diarization", lambda *_a, **_k: [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}])
    monkeypatch.setattr(transcribe, "merge_results",
                         lambda segs, turns: [{**segs[0], "speaker": "[unknown]"}])
    monkeypatch.setattr(transcribe, "get_wav_duration", lambda _: 1.0)
    monkeypatch.setattr(transcribe, "generate_markdown", lambda *a, **k: "# md")
    monkeypatch.setattr(transcribe, "clear_diarize_pipeline", lambda: None)
    monkeypatch.setattr(transcribe, "_release_whisper_before_diarization", lambda: None)

    src = tmp_path / "src.mp4"
    src.write_bytes(b"x")
    transcribe._run_job_impl(src, "large-v3-turbo", None, None, None, False, True, "", "tok")


def test_whisper_failure_releases_model(tmp_path, monkeypatch):
    import pytest
    import transcribe

    released = []
    monkeypatch.setattr(transcribe, "derive_paths", lambda _p: {
        "wav": tmp_path / "w.wav",
        "whisper_json": tmp_path / "w_whisper.json",
        "diarize_json": tmp_path / "w_diarize.json",
        "segments_json": tmp_path / "w_segments.json",
        "output_md": tmp_path / "w.md",
    })
    monkeypatch.setattr(transcribe, "convert_to_wav", lambda *_: None)
    monkeypatch.setattr(transcribe, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(transcribe, "clear_diarize_pipeline", lambda: None)
    monkeypatch.setattr(transcribe, "_release_whisper_before_diarization",
                         lambda: released.append(True))

    def _boom(*_a, **_k):
        raise RuntimeError("whisper exploded")

    monkeypatch.setattr(transcribe, "run_whisper", _boom)
    src = tmp_path / "fail.mp4"
    src.write_bytes(b"x")
    with pytest.raises(RuntimeError):
        transcribe._run_job_impl(src, "large-v3-turbo", None, None, None, False, False, "", "tok")
    assert released != []




def _stub_job_audio(tmp_path, monkeypatch):
    import transcribe

    source = tmp_path / "lecture.m4a"
    source.write_bytes(b"audio")
    monkeypatch.setattr(transcribe.config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(transcribe.config, "TRANSCRIPT_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(transcribe, "convert_to_wav", lambda *_: None)
    monkeypatch.setattr(transcribe, "get_wav_duration", lambda _: 2.0)
    monkeypatch.setattr(transcribe, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(transcribe, "clear_diarize_pipeline", lambda: None)
    monkeypatch.setattr(transcribe, "_release_whisper_before_diarization", lambda: None)

    def recognize(_wav, cache, _language, _hotwords, key):
        segments = [{"start": 0.0, "end": 2.0, "text": "Lecture content."}]
        transcribe._write_stage_cache(cache, "whisper", key, segments)
        return segments

    monkeypatch.setattr(transcribe, "run_whisper", recognize)
    return source


def test_cli_and_server_produce_identical_transcripts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import transcribe

    source = _stub_job_audio(tmp_path, monkeypatch)
    class Pipeline:
        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(itertracks=lambda **_: iter([
                (SimpleNamespace(start=0.0, end=2.0), None, "SPEAKER_00")]))

    monkeypatch.setattr(transcribe, "get_diarize_pipeline", lambda *_: Pipeline())
    monkeypatch.setattr(transcribe, "get_hf_token", lambda: "test-token")
    monkeypatch.setattr(sys, "argv", ["transcribe.py", str(source), "--device", "cpu", "--language", "en"])
    transcribe._main()
    output = transcribe.derive_paths(source)["output_md"]
    cli_text = output.read_text(encoding="utf-8")
    output.unlink()
    transcribe.run_job_from_json({
        "input_path": str(source), "output_dir": str(output.parent),
        "device": "cpu", "language": "en", "token": "test-token",
    })
    assert output.read_text(encoding="utf-8") == cli_text
    assert "SPEAKER_00" in cli_text
    assert "Lecture content." in cli_text


import pytest


@pytest.mark.parametrize("entry", ["cli", "server"])
@pytest.mark.parametrize("failure", ["missing_token", "model_load", "inference", "empty"])
def test_diarization_failure_is_terminal_and_preserves_whisper_cache(
    tmp_path, monkeypatch, capsys, entry, failure,
):
    import io
    import json
    from types import SimpleNamespace
    import transcribe

    source = _stub_job_audio(tmp_path, monkeypatch)
    token = "" if failure == "missing_token" else "test-token"
    monkeypatch.setattr(transcribe, "get_hf_token", lambda: token)

    class Pipeline:
        def __call__(self, *_args, **_kwargs):
            if failure == "inference":
                raise RuntimeError("diarization inference failed")
            return SimpleNamespace(itertracks=lambda **_: iter([]))

    def load(*_args):
        if failure == "model_load":
            raise RuntimeError("diarization model failed to load")
        return Pipeline()

    monkeypatch.setattr(transcribe, "get_diarize_pipeline", load)
    paths = transcribe.derive_paths(source)
    # A failed rerun must not replace an existing archived transcript.
    paths["output_md"].write_text("Archived transcript", encoding="utf-8")
    if entry == "cli":
        monkeypatch.setattr(sys, "argv", ["transcribe.py", str(source), "--device", "cpu"])
        with pytest.raises(SystemExit) as exc:
            transcribe.main()
        assert exc.value.code == 1
    else:
        job = {"command": "transcribe", "job_id": "failure-job",
               "input_path": str(source), "output_dir": str(paths["output_md"].parent),
               "device": "cpu", "token": token}
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(job) + "\n"))
        transcribe.run_server()

    events = [json.loads(line.removeprefix("@@EVENT "))
              for line in capsys.readouterr().out.splitlines() if line.startswith("@@EVENT ")]
    assert events[-1]["event"] == "failed"
    assert not any(e["event"] in {"completed", "completed_with_warning"} for e in events)
    if entry == "server":
        assert events[-1]["job_id"] == "failure-job"
    assert paths["output_md"].read_text(encoding="utf-8") == "Archived transcript"
    assert paths["whisper_json"].is_file()
    assert not paths["diarize_json"].exists()
