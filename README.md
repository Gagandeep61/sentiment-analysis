---
title: Sentiment Analysis API
emoji: 🎯
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
---

# 🎯 Sentiment Analysis API

**Production ML API** classifying text as POSITIVE, NEGATIVE, or NEUTRAL using Twitter RoBERTa.  
Built with FastAPI · Docker · HuggingFace Transformers · Deployed on HuggingFace Spaces + Vercel.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://sentiment-analysis-lake.vercel.app)
[![API Docs](https://img.shields.io/badge/API%20Docs-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://gagan61-sentiment-analysis-api.hf.space/docs)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/Gagandeep61/sentiment-analysis/deploy.yml?style=for-the-badge&logo=github)](https://github.com/Gagandeep61/sentiment-analysis/actions)

---

## What This Is

A production-style REST API that wraps Twitter RoBERTa for real-time 3-class sentiment classification. This project bridges the gap between training ML models and **serving them in production** — the skill that matters in real engineering roles.

---

## Live Links

| Resource | URL |
|----------|-----|
| **Frontend UI** | https://sentiment-analysis-lake.vercel.app |
| **API Base URL** | https://gagan61-sentiment-analysis-api.hf.space |
| **API Docs** | https://gagan61-sentiment-analysis-api.hf.space/docs |
| **GitHub** | https://github.com/Gagandeep61/sentiment-analysis |

---

## Architecture

```
Browser (Vercel)
     ↓  POST /analyze {"text": "I love this!"}
FastAPI Backend (HuggingFace Spaces / Docker)
     ↓
Twitter RoBERTa Inference
     ↓
{"sentiment": "POSITIVE", "confidence": 0.9998, "processing_time_ms": 45.2}
```

### Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Model** | Twitter RoBERTa (cardiffnlp) | Trained on 124M tweets — handles slang, emojis, informal text |
| **Backend** | FastAPI + Uvicorn | Auto-docs, Pydantic validation, async-ready |
| **Rate Limiting** | slowapi | 30 req/min (single), 10 req/min (batch) — prevents free-tier abuse |
| **CORS** | FastAPI Middleware | Required for Vercel → HF Spaces cross-origin calls |
| **Container** | Docker | Reproducible builds, model baked in at build time |
| **Deployment** | HF Spaces + Vercel | Free Docker + static hosting, auto-deploy from GitHub |

---

## Project Structure

```
sentiment-analysis/
├── app/
│   ├── __init__.py
│   └── main.py           ← FastAPI: endpoints, CORS, rate limiting, validation
├── frontend/
│   └── index.html        ← Vanilla JS UI
├── Dockerfile            ← python:3.10-slim, port 7860
├── requirements.txt
└── README.md
```

---

## API Endpoints

### `GET /health`
```json
{"status": "healthy", "model": "twitter-roberta-sentiment"}
```

### `POST /analyze`
**Rate limit**: 30 req/min · **Input limit**: 5000 chars

```json
// Request
{"text": "I absolutely love this product!"}

// Response
{"text": "...", "sentiment": "POSITIVE", "confidence": 0.9998, "processing_time_ms": 45.2}
```

### `POST /batch-analyze`
**Rate limit**: 10 req/min · **Max**: 20 items

```json
// Request
{"texts": [{"text": "Amazing!"}, {"text": "Terrible."}, {"text": "It was okay."}]}

// Response
{"results": [...], "count": 3}
```

---

## Run Locally

```bash
git clone https://github.com/Gagandeep61/sentiment-analysis.git
cd sentiment-analysis

python -m venv venv
source venv/bin/activate     # Mac/Linux
# venv\Scripts\activate      # Windows

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Docs: http://localhost:8000/docs
```

**Docker:**
```bash
docker build -t sentiment-api .
docker run -p 7860:7860 sentiment-api
```

---

## Key Technical Decisions

**Why Twitter RoBERTa over DistilBERT?**  
DistilBERT was trained on SST-2 (movie reviews) — no slang, no emojis, no informal text. Twitter RoBERTa was trained on 124M real tweets, making it significantly better for real-world text.

**Why 3-class?**  
Binary positive/negative misses neutral cases. "The service was okay" is neither — forcing it into two buckets gives wrong results.

**Why CORS middleware?**  
Browsers block cross-origin requests by default. Frontend on Vercel and backend on HF Spaces have different domains — without `CORSMiddleware`, every browser request silently fails.

**Why rate limiting?**  
Free-tier CPU is shared. Without `slowapi`, one bad actor can flood the API and crash it for everyone. Limits are set at 30 req/min (single) and 10 req/min (batch) — enough for real usage, not enough to abuse.

**Why pre-download model in Dockerfile?**  
Model is ~500MB. Downloading at container startup adds 30-45s to every cold start. Pre-downloading during `docker build` means startup is 2-3s.

---

## Known Limitations

| Limitation | Cause | Fix |
|-----------|-------|-----|
| Cold starts (30-45s) | HF Spaces sleeps after 15min idle | Migrate to Render |
| CPU-only inference | Free tier, no GPU | Paid GPU tier or RunPod |
| In-memory rate limits | Resets on container restart | Redis-backed rate limiting |
| Sarcasm detection | Model sees "great" → positive | Fine-tune on sarcasm dataset |
| No API key auth | Public demo — acceptable for now | Required before any production use |

---

## Performance

| Metric | Value |
|--------|-------|
| Warm inference latency | 45–100ms |
| Model size | ~500MB |
| Batch throughput | ~10–15 texts/second |

---

## Portfolio Notes

```
Sentiment Analysis API | FastAPI · Twitter RoBERTa · Docker · HuggingFace Spaces · Vercel

• Production ML API serving 3-class sentiment (POSITIVE/NEGATIVE/NEUTRAL) on social media text
• FastAPI backend with rate limiting (30 req/min), CORS middleware, Pydantic input validation
• Twitter RoBERTa (124M tweet training) — handles modern language better than SST-2 models
• Docker containerized, deployed on HF Spaces (backend) + Vercel (frontend), auto-deploy from GitHub
```

---

## Next Steps

- [ ] Migrate to Render for always-on serving (no cold starts)
- [ ] Add request logging with Prometheus
- [ ] Fine-tune on sarcasm dataset
- [ ] Add API key authentication
- [ ] Redis-backed rate limiting for multi-replica support

---

## Author

**Gagandeep Singh** · [@Gagandeep61](https://github.com/Gagandeep61)

---

*Model selection is 20% of the work. Deployment, rate limiting, and reliability are the other 80%.*
