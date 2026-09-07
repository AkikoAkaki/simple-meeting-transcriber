# Meeting Transcriber

个人 Windows 转录工具：把 OBS 会议录屏、iPhone 语音备忘录以及其他音视频转为带时间戳和说话人标注的 Markdown，供长期归档、查阅和 agent 获取上下文。识别在本机运行；首次使用需要联网下载模型，不对原稿自动摘要、改写或推测说话人姓名。

[English](README.md)

## 使用

- **手动转录**：双击 `start.bat`，或点击托盘图标打开 dashboard，选择或拖入文件，点击 **Transcribe**。
- **OBS 录屏**：在 dashboard 设置录制目录并开启 **Watch in background**。只自动处理监听期间出现的新文件，不批量处理启动前的存档。暂不需要时关闭监听即可。
- **agent / 命令行**：调用 `transcribe.py`，与 dashboard 使用同一转录流程。成功退出码为 0，失败为非零。

默认 **Full pipeline** 包含说话人分离，需要在 dashboard 保存 HuggingFace Token，并接受 [speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) 和 [segmentation-3.0](https://hf.co/pyannote/segmentation-3.0) 的使用条款。缺少 Token 且无可用分离缓存、模型加载失败、分离运行失败或没有分离结果时，任务标记为失败，不发布本次 Markdown。

确定是单人录音且不需要区分说话人时，可显式选择 **Transcribe only**。包含学生互动的 lecture、seminar 或多人会议使用 **Full pipeline**。人数不确定时保持自动估计；已知准确人数时填写 **Exact speakers**。人名和术语可以填入 **Names / terms**，但说话人标签仍是编号，不代表已核实身份。

## 安装与启动

需要 Windows、Python 3.10+ 和 ffmpeg。`install.bat` 检查 Python、尝试通过 winget 安装缺失的 ffmpeg，并在 `.venv` 安装 `requirements.txt`。安装后用 `start.bat` 启动。

要登录后自动运行托盘服务：

```powershell
powershell -ExecutionPolicy Bypass -File setup_autostart.ps1
```

取消自启动时追加 `-Uninstall`。自启动由此脚本管理，不由 `config.py` 控制。

## 命令行

使用安装依赖的 Python；通过安装脚本配置的环境可调用 `.venv\Scripts\python.exe`：

```powershell
python transcribe.py "lecture.m4a" --language en
python transcribe.py "meeting.mp4" --num-speakers 3 --hotwords "vLLM,KV Cache"
python transcribe.py "solo-lecture.m4a" --transcribe-only
python transcribe.py "meeting.mp4" --diarize-only
```

`--diarize-only` 复用匹配当前识别参数的 Whisper 缓存；匹配的说话人分离缓存也会复用。失败后重试不需要删除识别缓存。可用 `--model large-v3`、`--device cpu`、`--max-speakers 3`、`--output-dir ./transcripts` 覆盖默认值。Token 可从 dashboard 存储、`HF_TOKEN` 环境变量或 `config.py` 读取。

## 输出与配置

- `transcripts/`：Markdown 文件，包含源文件名、时间戳和说话人编号。源文件指纹用于区分同名文件及被替换的内容。
- `cache/`：音频中间文件和识别、分离、合并结果。**Clear Audio Cache** 只清理与已结束任务关联的音频缓存，保留 JSON 和 Markdown。
- `%LOCALAPPDATA%/SimpleVideoTranscriber/`：dashboard 设置、任务数据库和文本日志。已有历史事件表保留，但不再写入新事件。

`config.py` 提供默认设置；dashboard 保存的设置供托盘使用，命令行参数覆盖对应默认值。默认模型为 `large-v3-turbo`，也可选 `large-v3`；设备默认为 `auto`。监听目录默认为用户 Videos 文件夹，可在 dashboard 修改，或用 `MEETING_TRANSCRIBER_WATCH_DIR` 指定默认值。

监听器等待文件大小稳定约 15 秒并检查文件能否打开；这是完成录制的近似判断，暂停录制或传输可能提前触发。输出不会移动或删除原始音视频，也不会自动添加媒体链接。
