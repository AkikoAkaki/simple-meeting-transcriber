# Simple Video Transcriber

> 自动转录视频或音频并识别说话人 — 完全本地运行，数据不会离开你的电脑。

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-lightgrey)]()

[English](README.md)

---

## 功能介绍

把任意视频或音频文件拖入应用——会议录像、课程讲座、采访播客都行——即可得到带说话人标注和时间戳的转录稿。所有处理在本地完成，基于 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 和 [pyannote.audio](https://github.com/pyannote/pyannote-audio)，无需联网，无需上传。

```
### [00:00:00 – 00:00:16] SPEAKER_00
我们先来看看你最近在做什么。

### [00:00:16 – 00:01:10] SPEAKER_01
能看到我的屏幕吗？我跑了一下布局实验……
```

输出格式可选：**Markdown**（`.md`）、**纯文本**（`.txt`）。

---

## 主要特性

- **说话人识别** — 每段文字标注对应的说话人 ID
- **Markdown 与纯文本输出** — Markdown 适合做笔记，纯文本适合输入 LLM
- **灵活的处理模式** — 完整流程、仅转录、或仅重新做说话人分离
- **完全离线** — 所有模型本地运行，不需要 API Key，不依赖云服务
- **GPU 加速** — 检测到 CUDA 自动使用，否则回退到 CPU
- **后台自动监听** — 开机后在系统托盘监听 `I:\视频档案`，文件完成后自动触发转录
- **跨平台** — 支持 Windows 和 macOS，提供一键安装脚本

---

## 安装

### Windows

1. [下载 zip 包](https://github.com/AkikoAkaki/simple-video-transcriber/releases) 并解压
2. 双击 `install.bat`
3. 双击 `start.bat`

### macOS

1. [下载 zip 包](https://github.com/AkikoAkaki/simple-video-transcriber/releases) 并解压
2. 双击 `install.command`（如有提示，输入密码以允许 Homebrew 安装）
3. 双击 `start.command`

> **GPU 加速（可选）：** 默认安装使用仅 CPU 的 PyTorch。安装完成后，可前往 [pytorch.org](https://pytorch.org/get-started/locally/) 替换为对应 CUDA 版本以获得更快速度。

---

## 快速上手

1. 首次启动后应用会在系统托盘运行；打开 dashboard 设置一次 HuggingFace Token（可选）
2. 将视频/音频拖入 dashboard（或点击 **Browse file** 选择文件）
3. 选择输出格式和处理模式
4. 点击 **Transcribe**（开始转录）
5. 完成后点击 **Open transcript →** 查看结果

HuggingFace Token 仅用于说话人识别。不填写 Token 也能正常转录，只是输出中不会有说话人标注。

---

## 性能参考

RTX 4060（8 GB 显存），25 分钟视频，`large-v3` 模型：

| 步骤 | 时间 |
|------|------|
| 音频提取 | ~10 秒 |
| Whisper 转录 | ~8 分钟 |
| 说话人分离 | ~12 分钟 |

纯 CPU 环境预计慢 5–10 倍。

---

## 常见问题

**没有 GPU 能用吗？**
可以。CPU 可以正常运行，只是速度较慢。应用会在状态栏提示当前使用 CPU。

**选哪个模型？**
默认 `large-v3-turbo`：又快又准。`large-v3` 在嘈杂音频上准一点，但更慢更吃显存。

**识别出的语言不对？**
在设置中手动选择语言（自动检测 / English / 中文 / 日本語 / …）。

**输出没有说话人标注？**
需要配置 HuggingFace Token 并接受 pyannote 模型的使用条款。引导面板会一步步带你完成。

**能不重新转录，只重新做说话人分离吗？**
可以 — 在 Pipeline 下拉菜单中选择 **Re-diarize only**，已缓存的 Whisper 结果会直接复用。

**可以把 SPEAKER_00 改成真实姓名吗？**
可以 — 在 Recent tasks 中选中已完成任务，点击 **Rename speakers**。这只重新生成输出，不会重新运行模型。
已填写的名字会保存在这份音频的结果缓存中，之后重新生成其他输出格式时会自动沿用；不会跨不同会议自动猜测身份。

---

## 系统要求

- Windows 10+ 或 macOS 12+
- Python 3.10+（安装脚本自动安装）
- ffmpeg（安装脚本自动安装）

---

<details>
<summary>进阶：命令行用法</summary>

```bash
# 基本转录
python transcribe.py path/to/meeting.mp4

# 强制指定语言
python transcribe.py meeting.mp4 --language zh

# 仅转录（无需 Token）
python transcribe.py meeting.mp4 --transcribe-only

# 仅重新做说话人分离（复用已缓存的 Whisper 结果）
python transcribe.py meeting.mp4 --diarize-only

# 覆盖模型、设备、说话人数量或输出目录
python transcribe.py meeting.mp4 --model large-v3 --device cpu --max-speakers 3 --output-dir ./my-transcripts

# 已知准确人数时，使用准确人数；可附加人名和专业术语提示
python transcribe.py meeting.mp4 --num-speakers 3 --hotwords "Alice,vLLM,KV Cache"
```

</details>

<details>
<summary>进阶：自动监听</summary>

后台服务监听 `I:\视频档案`，只处理启动后出现的新文件。

```bash
python tray_app.py         # 启动后台托盘服务和 dashboard
```

文件大小连续约 15 秒不变后才开始转录。转录成功后不会移动或重命名 OBS 原文件。

</details>

<details>
<summary>进阶：配置参考</summary>

编辑 `config.py` 修改默认行为：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `WATCH_DIR` | `I:\视频档案` | 监听目录；也可在 dashboard 中修改 |
| `TRANSCRIPT_DIR` | `transcripts/` | 输出目录 |
| `CACHE_DIR` | `cache/` | 中间文件缓存，随时可删 |
| `WHISPER_MODEL` | `large-v3-turbo` | 模型大小：`large-v3-turbo` / `large-v3` |
| `LANGUAGE` | `None` | `"zh"` / `"en"` / `"ja"` / … — `None` = 自动检测 |
| `DEVICE` | `"auto"` | `"cuda"` / `"cpu"` / `"auto"` |
| `MAX_SPEAKERS` | `None` | 已知说话人数量时填整数，提高准确率 |
| `NUM_SPEAKERS` | `None` | 确定人数时填整数；优先于 `MAX_SPEAKERS` |
| `HOTWORDS` | `""` | 可选的人名和专业术语，逗号分隔 |
| `MIN_FILE_SIZE_KB` | `100` | 忽略小于此大小的文件 |
| `WATCH_EXTENSIONS` | `{".mp4", …}` | 监听的文件类型 |

</details>

---

## 许可证

MIT
