# Distilbert Emotion Classifier — FastAPI + MLOps

[![CI](https://github.com/Priyrajsinh/distilbert-emotion-classifier-fastapi/actions/workflows/ci.yml/badge.svg)](https://github.com/Priyrajsinh/distilbert-emotion-classifier-fastapi/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🤗 Live Demos

| Project | Link |
|---|---|
| 🎭 Emotion Classifier (DistilBERT + GoEmotions) | [![HF Space](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/Priyrajsinh/distilbert-emotion-classifier) |

Fine-tune DistilBERT on GoEmotions, serve it via production-grade FastAPI, and track experiments with MLflow — all in one reproducible project.

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/Priyrajsinh/distilbert-emotion-classifier-fastapi.git
cd distilbert-emotion-classifier-fastapi

# 2. Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Start the API (model must be trained first; see Training below)
uvicorn src.api.app:app --reload --port 8000

# 4. (Optional) Launch the Gradio demo in a second terminal
python -m src.api.gradio_demo
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Training

```bash
# Download and preprocess GoEmotions
python -m src.data.load_raw
python -m src.data.preprocessing

# Fine-tune DistilBERT (3 epochs, tracked in MLflow)
python -m src.training.train

# Evaluate and write reports/results.json
python -m src.evaluation.evaluate
```

---

## Docker

```bash
# Multi-stage build (~500 MB final image)
docker build -t emotion-api .
docker run -p 8000:8000 emotion-api
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Client / Gradio UI                  │
└───────────────────────────┬─────────────────────────────┘
                            │  HTTP POST /api/v1/predict
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI  (port 8000)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  SlowAPI     │  │   Pandera    │  │  Prometheus    │  │
│  │  rate limit  │  │  validation  │  │  /metrics      │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
│                         │                                 │
│                         ▼                                 │
│              SentimentClassifier                          │
│        (DistilBERT fine-tuned on GoEmotions)              │
└───────────────────────────┬─────────────────────────────┘
                            │  7-class softmax
                            ▼
               { label, confidence, probabilities, trace_id }
```

**Key design decisions**

| Layer | Choice | Why |
|-------|--------|-----|
| Model | DistilBERT-base-uncased | 40% smaller than BERT, 97% accuracy |
| Dataset | GoEmotions → 7 macro labels | Reduces class imbalance vs 28 fine-grained labels |
| Serving | FastAPI + SlowAPI | Async, type-safe, rate-limited out of the box |
| Validation | Pandera + Pydantic | Schema contracts at both data-pipeline and API layers |
| Observability | Prometheus + MLflow | Industry-standard metrics scraping + experiment tracking |

---

## Benchmarks

Results from `reports/results.json` on the held-out GoEmotions test split (4 545 samples):

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| Zero-shot baseline (`bhadresh-savani/distilbert-base-uncased-emotion`, 200 samples) | 50.0% | 45.7% |
| **Fine-tuned (this project)** | **61.4%** | **62.3%** |
| **Delta** | +11.4 pp | **+16.6 pp** |

Class breakdown (fine-tuned model):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| joy | 0.44 | 0.61 | 0.51 | 495 |
| sadness | 0.39 | 0.56 | 0.46 | 219 |
| anger | 0.34 | 0.71 | 0.46 | 173 |
| fear | 0.84 | 0.77 | 0.80 | 1 480 |
| surprise | 0.76 | 0.45 | 0.57 | 1 602 |
| disgust | 0.50 | 0.68 | 0.57 | 238 |
| neutral | 0.41 | 0.67 | 0.51 | 338 |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Liveness check — status, uptime, memory |
| `POST` | `/api/v1/predict` | Classify text → label + probabilities |
| `GET` | `/api/v1/model_info` | Model name, num_labels, evaluation metrics |
| `GET` | `/metrics` | Prometheus scrape endpoint |
| `GET` | `/docs` | Swagger UI |

**Predict request / response**

```json
// POST /api/v1/predict
{ "text": "I am so happy today!" }

// 200 OK
{
  "label": "joy",
  "confidence": 0.87,
  "probabilities": { "joy": 0.87, "sadness": 0.03, ... },
  "trace_id": "a1b2c3d4-..."
}
```

---

## Project Structure

```
B1-HuggingFace-FastAPI/
├── config/config.yaml          # All hyperparameters (never hardcoded)
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI application
│   │   └── gradio_demo.py      # Gradio UI
│   ├── data/
│   │   ├── load_raw.py         # GoEmotions download
│   │   ├── preprocessing.py    # Train/val/test split
│   │   ├── validation.py       # Pandera schema
│   │   └── schemas.py          # Pydantic I/O schemas
│   ├── models/
│   │   ├── base.py             # Abstract BaseModel
│   │   └── model.py            # SentimentClassifier
│   ├── training/train.py       # MLflow-tracked training loop
│   ├── evaluation/evaluate.py  # Metrics + reports/results.json
│   ├── exceptions.py           # Custom exception hierarchy
│   └── logger.py               # Structured logging
├── tests/                      # pytest suite (93% coverage)
├── Dockerfile                  # Multi-stage Docker build
├── requirements.txt
└── requirements-dev.txt
```

---

## References

- Demszky et al. (2020). **GoEmotions: A Dataset of Fine-Grained Emotions.** ACL 2020. [arXiv:2005.00547](https://arxiv.org/abs/2005.00547)
- Sanh et al. (2019). **DistilBERT, a distilled version of BERT.** NeurIPS 2019 Workshop. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
- Devlin et al. (2018). **BERT: Pre-training of Deep Bidirectional Transformers.** [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
