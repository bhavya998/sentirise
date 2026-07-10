<div align="center">

# sentirise

**Fine-tune a 0.6B LLM with QLoRA for sentiment classification. Train on IMDB, serve via API, classify from the browser — all on local GPU.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/Tests-26%20passing-22c55e)]()
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

</div>

---

## What This Is

A complete QLoRA fine-tuning pipeline that turns **Qwen3-0.6B** into a sentiment classifier:

1. **Train** — Fine-tune with 4-bit QLoRA (PEFT) on IMDB reviews
2. **Serve** — FastAPI backend with `/classify` endpoint
3. **Classify** — Next.js frontend for real-time sentiment prediction

Only **4.5M trainable params** (0.76% of the model) — the LoRA adapter is ~18MB.

---

## Quick Start

### Prerequisites

- **Python 3.12+** and [`uv`](https://docs.astral.sh/uv/)
- **NVIDIA GPU** with CUDA (tested on RTX 3070 Ti, 8GB VRAM)
- **Node.js 18+** for the frontend

```bash
git clone https://github.com/bhavya998/sentirise.git
cd sentirise

# Install
uv sync
cd ui && npm install && cd ..

# Train (downloads Qwen3-0.6B + IMDB, ~2 min on GPU)
uv run sentirise train --epochs 1 --max-train 500 --max-eval 100

# Serve backend
uv run sentirise serve         # FastAPI on :8000

# Frontend (separate terminal)
cd ui && npm run dev           # Next.js on :3000
```

Open `http://localhost:3000`, type text, get sentiment.

---

## How It Works

```
IMDB Reviews (500 samples)
        ↓
  Instruction Format
  "Classify the sentiment of: ... Review → positive/negative"
        ↓
  Qwen3-0.6B (4-bit NF4 quantized)
  + LoRA adapters (r=16, α=32)
  on q_proj, k_proj, v_proj, o_proj
        ↓
  SFTTrainer (TRL) — 1 epoch, lr=2e-4
        ↓
  ~18MB adapter saved to data/adapters/
        ↓
  Inference: base model + adapter → "positive"/"negative"
```

---

## CLI Commands

| Command | Description |
|---|---|
| `sentirise train` | Fine-tune QLoRA adapter on IMDB |
| `sentirise serve` | Start FastAPI server on :8000 |
| `sentirise classify "text"` | Classify sentiment from CLI |
| `sentirise batch file.json` | Batch classify from JSON |

## API

```bash
# Health
GET /health → {"status": "ok"}

# Model info
GET /info → {"model": "Qwen/Qwen3-0.6B", "adapter_loaded": true, "labels": ["negative", "positive"]}

# Classify
POST /classify
  Body: {"text": "This movie was fantastic!"}
  Response: {"label": "positive", "confidence": 0.92, "raw_output": "positive"}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Base Model | `Qwen/Qwen3-0.6B` (600M params, 4-bit quantized) |
| Fine-tuning | PEFT (LoRA) + TRL (SFTTrainer) + bitsandbytes |
| Dataset | Stanford IMDB (subsampled) |
| Backend | FastAPI + Uvicorn |
| Frontend | Next.js 16 + React 19 + Tailwind 4 |
| Inference | HuggingFace transformers |
| CLI | Typer + Rich |

## QLoRA Config

```python
r = 16              # LoRA rank
lora_alpha = 32     # scaling
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
trainable = 4,587,520 / 600,637,440 (0.76%)
```

---

## Project Structure

```
sentirise/
├── src/sentirise/
│   ├── config.py         Model IDs, LoRA config, training hyperparams
│   ├── dataset.py        IMDB loading + instruction prompt formatting
│   ├── train.py          QLoRA fine-tuning with PEFT + TRL
│   ├── classifier.py     Load base + adapter, classify sentiment
│   ├── api.py            FastAPI (health, info, classify)
│   └── cli.py            CLI (train, serve, classify, batch)
├── tests/                26 tests (config, dataset, classifier, API, E2E)
├── scripts/              Training check + classifier test scripts
├── ui/                   Next.js 16 frontend
├── data/                 Adapter output (gitignored)
├── Makefile
└── pyproject.toml
```

---

## Testing

```bash
make test         # 26 tests: config, dataset, classifier, API, E2E
make lint         # ruff + eslint
```

| Suite | Tests |
|---|---|
| `test_config.py` | LoRA config, labels, training args |
| `test_dataset.py` | IMDB loading, prompt formatting |
| `test_classifier.py` | Output parsing (positive/negative/unknown/case) |
| `test_api.py` | All endpoints via TestClient |
| `test_e2e.py` | Live server + real HTTP + CORS |

---

## Notes

- Training on **500 samples / 1 epoch** proves the pipeline works end-to-end. For production accuracy, use 2000+ samples and 3+ epochs.
- The adapter (`data/adapters/`) is ~18MB — tiny compared to the 1.2GB base model.
- 4-bit quantization keeps VRAM usage under **2GB** during training.

## License

MIT
