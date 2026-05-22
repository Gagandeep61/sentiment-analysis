---
title: Sentiment Analysis API
emoji: 🎯
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
---
 
<div align="center">
# 🎯 Sentiment Analysis API
 
**DEPLOYED ML API** classifying text as POSITIVE, NEGATIVE, or NEUTRAL using Twitter RoBERTa.  
Built with FastAPI · Docker · HuggingFace Transformers · Deployed on HuggingFace Spaces + Vercel.
 
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://sentiment-analysis-lake.vercel.app)
[![API Docs](https://img.shields.io/badge/API%20Docs-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://gagan61-sentiment-analysis-api.hf.space/docs)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
 
</div>
---
 
## 📌 What This Is
 
A production-style REST API wrapping Twitter RoBERTa for real-time 3-class sentiment classification. This project bridges training ML models and **serving them in production ready form** — the skill that matters in real engineering roles.
 
```
Browser (Vercel)
     ↓  POST /analyze {"text": "I love this!"}
FastAPI Backend (HuggingFace Spaces / Docker)
     ↓
Twitter RoBERTa Inference
     ↓
{"sentiment": "POSITIVE", "confidence": 0.9998, "processing_time_ms": 45.2}
```
 
---
 
## 🚀 Live Links
 
| Resource | URL |
|----------|-----|
| **Frontend UI** | https://sentiment-analysis-lake.vercel.app |
| **API Base URL** | https://gagan61-sentiment-analysis-api.hf.space |
| **API Docs (Swagger)** | https://gagan61-sentiment-analysis-api.hf.space/docs |
| **GitHub Repo** | https://github.com/Gagandeep61/sentiment-analysis |
 
---
 
## 🏗️ Architecture
 
```
sentiment-analysis/
├── app/
│   ├── __init__.py
│   └── main.py           ← FastAPI: endpoints, CORS, rate limiting, validation
├── frontend/
│   └── index.html        ← Vanilla JS UI
├── Dockerfile            ← python:3.10-slim, port 7860
├── requirements.txt      ← Pinned dependencies
└── README.md
```
 
### Tech Stack
 
| Layer | Technology | Why |
|-------|-----------|-----|
| **Model** | Twitter RoBERTa (cardiffnlp) | Trained on 124M tweets — handles slang, emojis, informal text |
| **Backend** | FastAPI + Uvicorn | Auto-docs, Pydantic validation, async-ready |
| **Rate Limiting** | slowapi | 30 req/min (single), 10 req/min (batch) — prevents free-tier abuse |
| **CORS** | FastAPI Middleware | Required for Vercel → HF Spaces cross-origin calls |
| **Container** | Docker | Reproducible builds, model pre-downloaded at build time |
| **Deployment** | HF Spaces + Vercel | Free Docker + static hosting, auto-deploy from GitHub |
 
---
 
## 📡 API Endpoints
 
### `GET /health`
```json
{"status": "healthy", "model": "twitter-roberta-sentiment"}
```
 
### `POST /analyze`
**Rate limit**: 30 req/min · **Input**: 1–5000 characters
 
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
 
## 🖥️ Run Locally
 
### Prerequisites
- Python 3.10+
- 2GB RAM (for model)
### Direct (No Docker)
```bash
git clone https://github.com/Gagandeep61/sentiment-analysis.git
cd sentiment-analysis
 
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
 
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```
 
### Docker
```bash
docker build -t sentiment-api .
docker run -p 7860:7860 sentiment-api
```
 
---
 
## 🧪 Test the API
 
```python
import requests
r = requests.post("https://gagan61-sentiment-analysis.hf.space/analyze",
                  json={"text": "I love this!"})
print(r.json())
```
 
Or visit the **[Swagger UI](https://gagan61-sentiment-analysis.hf.space/docs)** directly.
 
---
 
## 🔑 Key Technical Decisions
 
**Why Twitter RoBERTa over DistilBERT?**  
DistilBERT was trained on SST-2 (movie reviews) — no slang, no emojis, no informal text. Twitter RoBERTa was trained on 124M real tweets, making it significantly better for real-world text.
 
**Why 3-class (not 2-class)?**  
Binary positive/negative misses neutral cases. "The service was okay" is neither — forcing it into two buckets gives wrong results.
 
**Why CORS middleware?**  
Browsers block cross-origin requests by default. Frontend (Vercel) and backend (HF Spaces) have different domains — without `CORSMiddleware`, every browser request silently fails.
 
**Why rate limiting?**  
Free-tier CPU is shared. Without `slowapi`, one bad actor can flood the API and crash it for everyone.
 
**Why pre-download model in Dockerfile?**  
Model is ~500MB. Downloading at startup adds 30–45s to every cold start. Pre-downloading during `docker build` brings startup down to 2–3s.
 
---
 
## ⚠️ Known Limitations
 
| Limitation | Cause | Fix |
|-----------|-------|-----|
| Cold starts (30–45s) | HF Spaces sleeps after 15min idle | Migrate to Render |
| CPU-only inference | Free tier, no GPU | Paid GPU tier or RunPod |
| In-memory rate limits | Resets on container restart | Redis-backed rate limiting |
| Sarcasm detection | Model sees "great" → positive | Fine-tune on sarcasm dataset |
| Non-English text | Model is English-only | Use multilingual model |
| Very short inputs ("lol", "ok") | Low-confidence outputs, not filtered |  Preprocess and Normalize Slang | 
 | Synchronous inference | Transformer model blocks event loop under concurrent load | Use run_in_executor |
---
 
## 📈 Performance
 
| Metric | Value |
|--------|-------|
| Inference latency (warm) | 45–100ms |
| Model size | ~500MB |
| Docker image | ~2.5GB |
| Batch throughput | ~10–15 texts/second |
| Cold start | 30–45 seconds |
 
---
 
## 🎤 Resume Bullet
 
```
Sentiment Analysis API | FastAPI · Twitter RoBERTa · Docker · HuggingFace Spaces · Vercel
 
• Built and deployed a REST API serving 3-class sentiment classification (POSITIVE/NEGATIVE/NEUTRAL)
• Used Twitter RoBERTa (trained on 124M tweets) over DistilBERT for better real-world text handling
• Containerized with Docker; model pre-loaded at build time to reduce warm startup to ~2–3s
• Rate limiting (slowapi), CORS middleware, Pydantic validation — handles cross-origin browser requests
• Deployed backend on HuggingFace Spaces, frontend on Vercel with auto-deploy from GitHub
```
 
---
 
## 🛣️ Next Steps
 
- [ ] Migrate to Render for always-on serving (no cold starts)
- [ ] Add request logging with Prometheus
- [ ] Fine-tune on sarcasm/mixed sentiment data
- [ ] Add API key authentication
- [ ] Redis-backed rate limiting for multi-replica support
---
 
## 👤 Author
 
**Gagandeep Singh** · [@Gagandeep61](https://github.com/Gagandeep61)
 
---
 
*Deployed ML engineering: serving models beats training them.*
 