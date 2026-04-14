<p align="center">
  <img src="assets/logo.png" alt="AccentCoach" width="480"/>
</p>

<p align="center">
  <b>Self-hosted AI speaking coach for American English learners.</b><br/>
  Real conversation practice · Pronunciation coaching · Vocabulary notebook · Spaced-repetition review
</p>

---

## Features

- **Real-time conversation** with an AI coach (streaming LLM)
- **Speech-to-text** via local [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (GPU or CPU)
- **Text-to-speech** via [edge-tts](https://github.com/rany2/edge-tts) (free, no API key)
- **Tap-to-lookup** — click any word in coach replies for instant meaning + auto-save to notebook
- **Phrase selection** — select a phrase for contextual translation
- **Spaced-repetition notebook** with Leitner box review
- **6 conversation scenarios** (Daily Chat, Business, Travel, Interview, Discussion, Stories)
- **3 difficulty levels** (Beginner / Intermediate / Advanced)
- **Zero external STT API cost** — Whisper runs 100% locally
- **Single-file frontend** — no build step, no Node.js

## Architecture

```text
┌───────────────────────────────────┐
│         Browser (index.html)      │
│   mic → WebM → /api/transcribe   │
│   text → /api/chat (SSE stream)  │
│   audio ← /api/tts               │
└──────────────┬────────────────────┘
               │ HTTPS (self-signed)
┌──────────────▼────────────────────┐
│          FastAPI backend          │
│  ┌──────────┐  ┌───────────────┐  │
│  │ Whisper  │  │ OpenAI-compat │  │
│  │ (local)  │  │   LLM API     │  │
│  └──────────┘  └───────────────┘  │
│  ┌──────────┐  ┌───────────────┐  │
│  │ edge-tts │  │ JSON storage  │  │
│  └──────────┘  └───────────────┘  │
└───────────────────────────────────┘
```

## Directory Structure

```text
ai-accent-coach/
├── app.py              # FastAPI backend (all APIs)
├── coach.sh            # Service manager (start/stop/restart/status/log)
├── requirements.txt
├── .env.example        # ← copy to .env and fill in
├── static/
│   └── index.html      # Single-file frontend
└── assets/
    └── logo.png
```

## Quick Start

```bash
git clone https://github.com/Carl-AI-Lab/ai-accent-coach.git
cd ai-accent-coach

python3 -m venv venv && source venv/bin/activate   # recommended
pip install -r requirements.txt

cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

Start the server:

```bash
bash coach.sh start          # background, logs to accent-coach.log
# or
python app.py                # foreground with live output
```

Open **https://localhost:8443** in your browser (accept the self-signed certificate).

Stop:

```bash
bash coach.sh stop
```

## Configuration

All settings live in **`.env`** (see [.env.example](.env.example)):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | API key for any OpenAI-compatible service |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | LLM endpoint. Works with OpenAI, DeepSeek, Ollama, etc. |
| `LLM_MODEL` | `gpt-4.1-mini` | Model name |
| `WHISPER_DEVICE` | `cuda` | `cuda` for GPU, `cpu` for CPU-only |
| `WHISPER_MODEL` | `distil-large-v3` | Whisper model size |
| `WHISPER_UNLOAD_TIMEOUT` | `120` | Seconds idle before auto-unloading the model from memory |
| `TTS_VOICE` | `en-US-AndrewMultilingualNeural` | edge-tts voice name |
| `PORT` | `8443` | HTTPS listen port |

### Using a Different LLM Provider

AccentCoach works with any OpenAI-compatible API:

```bash
# OpenAI (default)
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4.1-mini

# DeepSeek
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=sk-...
LLM_MODEL=deepseek-chat

# Ollama (local, free)
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
LLM_MODEL=llama3.1
```

## GPU vs CPU

Whisper runs locally for speech-to-text. You can choose GPU or CPU mode.

### GPU Mode (recommended)

```bash
WHISPER_DEVICE=cuda
```

Requires NVIDIA GPU with **cuBLAS** and **cuDNN** for CUDA 12.

If you installed CUDA libraries via pip (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`), you need to set `LD_LIBRARY_PATH` so that `faster-whisper` can find them:

```bash
# Find the library paths installed by pip
python3 -c "import nvidia.cublas.lib, nvidia.cudnn.lib; print(nvidia.cublas.lib.__path__[0]); print(nvidia.cudnn.lib.__path__[0])"

# Example output:
#   /home/you/venv/lib/python3.11/site-packages/nvidia/cublas/lib
#   /home/you/venv/lib/python3.11/site-packages/nvidia/cudnn/lib

# Set before starting (or uncomment the line in coach.sh):
export LD_LIBRARY_PATH="/home/you/venv/.../nvidia/cublas/lib:/home/you/venv/.../nvidia/cudnn/lib"
```

### CPU Mode

```bash
WHISPER_DEVICE=cpu
```

No GPU required. Works on any machine. Significantly slower — see benchmark below.

> **Tip:** On CPU, consider using a smaller model for faster response:
> ```bash
> WHISPER_MODEL=base     # fastest, lower accuracy
> WHISPER_MODEL=small    # good balance
> WHISPER_MODEL=medium   # better accuracy, slower
> ```

### Benchmark (actual measurements)

Tested on the same machine with a 10-second audio clip, `distil-large-v3` model, `beam_size=5`, `vad_filter=True`:

| | Device | Compute Type | Model Load | Transcription (avg of 3) |
|---|---|---|---|---|
| **GPU** | NVIDIA RTX 4060 Laptop (8 GB) | float16 | 16.5 s | **0.51 s** |
| **CPU** | Intel i7-12700H (20 threads) | int8 | 5.7 s | **5.52 s** |

> **CPU is ~11× slower for transcription**, but model loading is actually 3× faster since it skips CUDA initialization. For casual practice the CPU latency (~5 s per utterance) is acceptable. For fluid conversation, **GPU is strongly recommended**.

## Requirements

- Python ≥ 3.9
- An OpenAI-compatible API key (for the LLM)
- *(GPU mode)* NVIDIA GPU + CUDA 12 + cuBLAS + cuDNN 9
- *(CPU mode)* No extra hardware — works on any x86_64 or ARM machine

## License

[Apache License 2.0](LICENSE)
