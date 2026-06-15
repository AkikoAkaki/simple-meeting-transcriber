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

输出格式可选：**Markdown**（`.md`）、**SRT 字幕**（`.srt`）、**纯文本**（`.txt`）。

---

## 主要特性

- **说话人识别** — 每段文字标注对应的说话人 ID
- **三种输出格式** — Markdown 适合做笔记，SRT 适合视频剪辑，纯文本适合输入 LLM
- **灵活的处理模式** — 完整流程、仅转录、或仅重新做说话人分离
- **完全离线** — 所有模型本地运行，不需要 API Key，不依赖云服务
- **GPU 加速** — 检测到 CUDA 自动使用，否则回退到 CPU
- **自动监听** — 文件丢入 `inbox/` 后自动触发转录，无需手动操作
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

1. 首次启动会弹出引导面板，按步骤申请免费的 [HuggingFace Token](https://huggingface.co/settings/tokens)，大约 2 分钟，只需操作一次
2. 将视频/音频拖入窗口上方区域（或点击 **Browse…** 选择文件）
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

纯 CPU 环境预计慢 5–10 倍。硬件较弱时，可在设置中切换到 `medium` 或 `small` 模型以换取速度。

---

## 常见问题

**没有 GPU 能用吗？**
可以。CPU 可以正常运行，只是速度较慢。应用会在状态栏提示当前使用 CPU。

**选哪个模型？**
默认的 `large-v3` 精度最高，推荐直接使用。硬件有限可选 `medium`，速度与精度较为平衡。`small` 和 `base` 更快，但精度明显下降。

**识别出的语言不对？**
在设置中手动选择语言（自动检测 / English / 中文 / 日本語 / …）。

**输出没有说话人标注？**
需要配置 HuggingFace Token 并接受 pyannote 模型的使用条款。引导面板会一步步带你完成。

**能不重新转录，只重新做说话人分离吗？**
可以 — 在 Pipeline 下拉菜单中选择 **Re-diarize only**，已缓存的 Whisper 结果会直接复用。

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
python transcribe.py meeting.mp4 --mode transcribe-only

# 仅重新做说话人分离（复用已缓存的 Whisper 结果）
python transcribe.py meeting.mp4 --mode diarize-only

# 选择输出格式
python transcribe.py meeting.mp4 --output-format srt
python transcribe.py meeting.mp4 --output-format txt

# 覆盖模型、设备、说话人数量或输出目录
python transcribe.py meeting.mp4 --model medium --device cpu --max-speakers 3 --output-dir ./my-transcripts
```

</details>

<details>
<summary>进阶：自动监听</summary>

监听 `inbox/` 目录，新文件到达时自动转录。

```bash
python watch.py            # 启动监听
python watch.py --dry-run  # 仅检测文件，不触发转录
```

文件大小连续 10 秒不变后才开始转录（可在 `config.py` 中调整）。转录成功后，视频可自动移入 `YYYY/` 子目录（由 `ORGANIZE_BY_YEAR` 控制）。

</details>

<details>
<summary>进阶：配置参考</summary>

编辑 `config.py` 修改默认行为：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `WATCH_DIR` | `inbox/` | 监听目录 |
| `TRANSCRIPT_DIR` | `transcripts/` | 输出目录 |
| `CACHE_DIR` | `cache/` | 中间文件缓存，随时可删 |
| `WHISPER_MODEL` | `large-v3` | 模型大小：`tiny` / `base` / `small` / `medium` / `large-v3` |
| `LANGUAGE` | `None` | `"zh"` / `"en"` / `"ja"` / … — `None` = 自动检测 |
| `DEVICE` | `"auto"` | `"cuda"` / `"cpu"` / `"auto"` |
| `MAX_SPEAKERS` | `None` | 已知说话人数量时填整数，提高准确率 |
| `ORGANIZE_BY_YEAR` | `True` | 转录完成后将视频移入 `YYYY/` 子目录 |
| `STABLE_SECONDS` | `10` | 文件大小稳定多少秒后开始转录 |
| `MIN_FILE_SIZE_KB` | `100` | 忽略小于此大小的文件 |
| `WATCH_EXTENSIONS` | `{".mp4", …}` | 监听的文件类型 |

</details>

---

## 许可证

MIT
