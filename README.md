# Revenant Echo

A local-only voice assistant for Windows. Wake-word triggered, GPU-accelerated,
no cloud calls, no API keys. Designed as the voice layer that sits in front of
a local LLM (Ollama by default) and turns spoken input into spoken output.

The whole pipeline runs on your machine: microphone → wake-word detection →
speech-to-text → local LLM → text-to-speech → speakers. Nothing leaves the box.

## Pipeline

```
  [you speak the wake word]
              │
              ▼
   OpenWakeWord (CPU, always-on)
              │ wake fires
              ▼
   Microphone opens, captures speech
              │
              ▼
   faster-whisper medium (GPU)
              │ transcript
              ▼
   Ollama local LLM (GPU)
              │ streaming tokens
              ▼
   Kokoro TTS (GPU, sentence-by-sentence)
              │ audio
              ▼
        Speakers
              │
              ▼
   5-second follow-up window
   (speak again → loop, silent → idle)
```

**Streaming TTS:** as the LLM emits tokens, completed sentences are pushed onto
a queue and synthesized in parallel. The assistant starts speaking ~1 sentence
into the LLM's response, not after the whole reply is generated.

**Warm-window lifecycle:** Whisper and Kokoro stay loaded in VRAM for 60 seconds
after the last activity, then unload. Conversational use stays fast; idle
periods release the GPU.

## Hardware

Built and tested on:

- Windows 11 Pro
- AMD Ryzen 7 5800XT, 64 GB RAM
- NVIDIA RTX 3070 Ti (8 GB VRAM, CUDA 13)
- USB microphone (any will do; gain matters more than brand)

Will run on any CUDA-capable NVIDIA GPU with ≥6 GB VRAM. CPU-only mode works
but Whisper-medium on CPU is slow enough to cook the chip — drop to
`whisper-small` in `config/config.yaml` if you have no GPU.

## Requirements

- **Python 3.11** (system install, not a venv — see `scripts/install.ps1`)
- **Ollama** for the LLM backend ([ollama.com](https://ollama.com)).
  Any model you've pulled works; configure it in `config/config.yaml`.
- **NVIDIA driver** new enough that `nvidia-smi` works.

## Install

Clone, install Python 3.11 if you don't have it, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

The install script:
1. Verifies Python 3.11 is available via the `py` launcher
2. Installs all Python dependencies (including Kokoro with a `--no-deps`
   workaround for its over-conservative numpy pin)
3. Installs CUDA-enabled PyTorch (cu121)
4. Downloads the default "Hey Friday" wake-word model
5. Downloads OpenWakeWord's support models

## Configure

Edit `config/config.yaml`:

- **`audio.input_device`** — set to your mic's PyAudio device index.
  Run V once; the boot log lists every audio device with its index.
- **`backend.ollama.model`** — change to whatever Ollama model you want.
  V's reference model (`revenant/v-9b`) is a personal build of a 9B
  model with a baked-in system prompt; substitute your own.
- **`tts.voice`** — choose a Kokoro voice (`af_bella`, `af_sarah`,
  `af_nicole`, `am_michael`, `am_adam`, etc.).
- **`wake_word.model`** — point at any OpenWakeWord-format `.onnx` file.
  See [`models/README.md`](models/README.md) for the wake-word options.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1
```

Or directly:

```powershell
py -3.11 src\main.py
```

Boot takes ~10–20 seconds (Ollama warmup). When you see
`Wake word listener started`, V is listening.

Say the wake word, then your command. Ctrl+C to stop.

## Project layout

```
Revenant-Echo/
├── config/
│   ├── config.yaml          # all tuneable settings
│   └── .env.example         # optional API keys (only for non-Ollama backends)
├── models/                  # wake-word .onnx files (gitignored)
│   └── README.md            # how to get them
├── scripts/
│   ├── install.ps1          # one-shot setup
│   ├── run.ps1              # launcher (checks Ollama, runs V)
│   ├── test_hardware.py     # CUDA + audio device check
│   └── enumerate_voices.py  # list / sample Kokoro voices
├── src/
│   ├── main.py              # orchestrator
│   ├── audio_io.py          # mic capture + speaker output, VAD recording
│   ├── wake_word.py         # OpenWakeWord listener
│   ├── stt.py               # faster-whisper wrapper
│   ├── tts.py               # Kokoro wrapper
│   ├── backend.py           # Ollama (streaming + non-streaming)
│   ├── vram_manager.py      # GPU lifecycle helpers
│   └── config_loader.py     # YAML config + logging
└── requirements.txt
```

## Architecture notes

- **One mic, two streams.** The wake-word listener and the speech recorder
  each open their own PyAudio input stream so the wake word can't be
  poisoned by the recorder's state and vice versa.
- **Hidden Whisper load.** When the wake word fires, V starts loading
  Whisper *on a background thread* while the mic is already open. The
  load cost is hidden behind your speech, not added to your latency.
- **Streaming-aware sentence splitter.** The Ollama → Kokoro pipeline
  finds sentence boundaries (`.`, `!`, `?`, `:`) as tokens arrive and
  flushes each completed sentence to a synthesis queue immediately.
  A dedicated speech-worker thread pulls from the queue so the token
  consumer never blocks on playback.
- **Single listening lock.** Wake-word and any future trigger (push-to-talk,
  hotkey, etc.) share one lock. While a turn is in flight, additional
  triggers no-op rather than starting a parallel mic stream.

## Roadmap

- Custom-trained "Wake up V" wake-word model (via OpenWakeWord's Colab
  notebook) to replace the placeholder "Hey Friday"
- Pluggable backends beyond Ollama — OpenCLAW is the planned next adapter
- Conversation memory across turns (currently each turn is stateless)
- A second persona (Keystone) on a different trigger, with its own
  Ollama model — design exists, build is paused while V is stabilized

## License

MIT. See `LICENSE`.
