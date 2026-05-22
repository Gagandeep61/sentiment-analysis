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
 
**Production ML API** that classifies text as POSITIVE, NEGATIVE, or NEUTRAL using Twitter RoBERTa.  
Built with FastAPI · Docker · HuggingFace Transformers · Deployed on HuggingFace Spaces + Vercel.
 
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://sentiment-analysis-lake.vercel.app)
[![API Docs](https://img.shields.io/badge/API%20Docs-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://gagan61-sentiment-analysis.hf.space/docs)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
 
</div>
---
 
## 📌 What This Is
 
A **production-style REST API** that wraps Twitter RoBERTa transformer model for real-time 3-class sentiment classification (POSITIVE/NEGATIVE/NEUTRAL). This project bridges training ML models and **serving them in production** — the skill that matters in real engineering roles.
 
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
| **API Base URL** | https://gagan61-sentiment-analysis.hf.space |
| **API Docs (Swagger)** | https://gagan61-sentiment-analysis.hf.space/docs |
| **GitHub Repo** | https://github.com/Gagandeep61/sentiment-analysis |
 
---
 
## 🏗️ Architecture
 
```
sentiment-analysis/
├── app/
│   ├── __init__.py
│   └── main.py           ← FastAPI: endpoints, CORS, rate limiting, validation
├── frontend/
│   └── index.html        ← Vanilla JS UI + model education panel
├── Dockerfile            ← Containerization (python:3.10-slim, port 7860)
├── requirements.txt      ← Pinned dependencies
└── README.md
```
 
### Tech Stack
 
| Layer | Technology | Why |
|-------|-----------|-----|
| **Model** | Twitter RoBERTa (cardiffnlp) | Trained on 124M tweets, handles modern slang & informal text better than SST-2 |
| **Backend** | FastAPI + Uvicorn | Auto-docs, Pydantic validation, lightweight, async-ready |
| **Rate Limiting** | slowapi | 30 req/min single, 10 req/min batch — prevents abuse on free tier |
| **CORS** | FastAPI Middleware | Enables Vercel → HF Spaces cross-origin calls |
| **Container** | Docker | Reproducible builds, model pre-downloaded at build time |
| **Deployment** | HF Spaces + Vercel | Free Docker hosting + static frontend hosting, auto-deploy from GitHub |
 
---
 
## 📡 API Endpoints
 
### `GET /health`
Health check.
 
```json
Response: {"status": "healthy", "model": "twitter-roberta-sentiment"}
```
 
---
 
### `POST /analyze`
Single text sentiment analysis.
 
**Request:**
```json
{
  "text": "I absolutely love this product!"
}
```
 
**Response:**
```json
{
  "text": "I absolutely love this product!",
  "sentiment": "POSITIVE",
  "confidence": 0.9998,
  "processing_time_ms": 45.2
}
```
 
**Rate limit**: 30 requests/minute per IP  
**Validation**: Text must be 1-5000 characters
 
---
 
### `POST /batch-analyze`
Multiple texts (max 20).
 
**Request:**
```json
{
  "texts": [
    {"text": "Amazing service!"},
    {"text": "Terrible experience."},
    {"text": "It was okay."}
  ]
}
```
 
**Response:**
```json
{
  "results": [
    {"text": "Amazing service!", "sentiment": "POSITIVE", "confidence": 0.9997, "processing_time_ms": 42.1},
    {"text": "Terrible experience.", "sentiment": "NEGATIVE", "confidence": 0.9991, "processing_time_ms": 41.8},
    {"text": "It was okay.", "sentiment": "NEUTRAL", "confidence": 0.8123, "processing_time_ms": 41.5}
  ],
  "count": 3
}
```
 
**Rate limit**: 10 requests/minute per IP
 
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
# Visit http://localhost:8000/docs
```
 
### Docker
 
```bash
docker build -t sentiment-api .
docker run -p 7860:7860 sentiment-api
```
 
---
 
## 🧪 Test the API
 
**Python:**
```python
import requests
r = requests.post("https://gagan61-sentiment-analysis.hf.space/analyze",
                  json={"text": "I love this!"})
print(r.json())
```
 
**Browser**: Visit https://sentiment-analysis-lake.vercel.app
 
**Swagger UI**: Visit https://gagan61-sentiment-analysis.hf.space/docs
 
---
 
## 🔑 Key Decisions
 
**Why Twitter RoBERTa?**  
Trained on 124M real tweets. Handles modern language (slang, emojis, informal) better than academic datasets. Still pre-trained — no training needed.
 
**Why 3-class (not 2-class)?**  
Real sentiment has neutral/mixed cases. "The service was okay" isn't positive or negative.
 
**Why CORS middleware?**  
Browsers block cross-origin requests by default. Frontend (Vercel) and backend (HF Spaces) have different domains → need explicit CORS.
 
**Why rate limiting?**  
Free tier has shared CPU. Without limits, one user can crash the service for everyone.
 
**Why pre-download model in Docker?**  
Model is 500MB. Downloading at startup = 30-45s cold start. Pre-downloading during build = 2-3s startup.
 
---
 
## ⚠️ Known Limitations
 
| Limitation | Why | Workaround |
|-----------|-----|-----------|
| Cold starts (30-45s) | HF Spaces free tier sleeps after 15 min | Use Render for always-on |
| CPU only (~50-100ms) | No GPU on free tier | Use paid GPU tier or RunPod |
| In-memory rate limits | Reset on restart | Add Redis for production |
| Sarcasm misses | Model trained on straightforward sentiment | Fine-tune on sarcasm data |
 
---
 
## 📈 Performance
 
| Metric | Value |
|--------|-------|
| Inference latency (warm) | 45–100ms |
| Model size | ~500MB (Twitter RoBERTa) |
| Batch throughput | ~10–15 texts/second |
| Cold start (HF Spaces) | 30–45 seconds |
| Docker image | ~2.5GB |
 
---
 
## 🎤 Resume Bullet
 
```
Sentiment Analysis API | FastAPI · Twitter RoBERTa · Docker · HF Spaces · Vercel
 
• Built production ML API serving 3-class sentiment (POSITIVE/NEGATIVE/NEUTRAL) on social media text
• Backend: FastAPI with rate limiting (30 req/min), CORS, Pydantic validation, batch processing
• Model: Twitter RoBERTa (124M tweet training) for modern language understanding
• Deployment: Docker containerized on HuggingFace Spaces (backend) + Vercel (frontend)
• Auto-deploy from GitHub with model pre-cached at build time for 2-3s cold starts
```
 
---
 
## 🛣️ Next Steps
 
- [ ] Migrate to Render for 24/7 uptime
- [ ] Add request logging (Prometheus)
- [ ] Fine-tune on sarcasm/mixed sentiment data
- [ ] Add API authentication (API keys)
- [ ] Cache frequent queries (Redis)
---
 
## 👤 Author
 
**Gagandeep Singh**  
GitHub: [@Gagandeep61](https://github.com/Gagandeep61)
 
---
 
*Production ML engineering: serving models beats training them.*
 