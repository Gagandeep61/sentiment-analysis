---
title: Sentiment Analysis Api
emoji: 🐳
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 8000
---

<div align="center">
# 🎯 Sentiment Analysis API
 
**Production ML API** that classifies text as POSITIVE or NEGATIVE using DistilBERT.  
Built with FastAPI · Docker · HuggingFace Transformers · Deployed on HuggingFace Spaces + Vercel.
 
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://sentiment-analysis-lake.vercel.app)
[![API Docs](https://img.shields.io/badge/API%20Docs-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://gagandeep61-sentiment-analysis.hf.space/docs)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
 
</div>
---
 
## 📌 What This Is
 
A **production-style REST API** that wraps the DistilBERT transformer model for real-time sentiment classification. This project bridges the gap between training ML models and actually **serving** them — the skill that matters in production engineering.
 
```
Browser (Vercel)
     ↓  POST /analyze {"text": "I love this!"}
FastAPI Backend (HuggingFace Spaces / Docker)
     ↓
DistilBERT Inference
     ↓
{"sentiment": "POSITIVE", "confidence": 0.9998, "processing_time_ms": 82.4}
```
 
---
 
## 🚀 Live Links
 
| Resource | URL |
|----------|-----|
| **Frontend UI** | https://sentiment-analysis-lake.vercel.app |
| **API Base URL** | https://gagandeep61-sentiment-analysis.hf.space |
| **Interactive API Docs** | https://gagandeep61-sentiment-analysis.hf.space/docs |
| **GitHub Repo** | https://github.com/Gagandeep61/sentiment-analysis |
 
---
 
## 🏗️ Architecture
 
```
sentiment-analysis/
├── app/
│   ├── __init__.py
│   └── main.py              ← FastAPI app: endpoints, CORS, rate limiting, validation
├── frontend/
│   └── index.html           ← Vanilla JS UI — calls backend via Fetch API
├── .github/
│   └── workflows/           ← CI/CD pipeline
├── Dockerfile               ← Containerization (python:3.10-slim, port 7860)
├── requirements.txt         ← Pinned dependencies
└── README.md
```
 
### Tech Stack
 
| Layer | Technology | Why |
|-------|-----------|-----|
| **ML Model** | DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) | Pre-trained, no retraining needed, 97% of BERT's accuracy at 40% the size |
| **Backend** | FastAPI + Uvicorn | Auto-generates OpenAPI docs, Pydantic validation, async-ready |
| **Validation** | Pydantic v2 | Rejects empty strings, enforces 5000-char limit before model runs |
| **Rate Limiting** | slowapi | Per-IP limiting (30 req/min single, 10 req/min batch) — prevents free-tier abuse |
| **CORS** | FastAPI CORSMiddleware | Required for browser → cross-origin API calls (Vercel → HF Spaces) |
| **Container** | Docker | Reproducible builds; required by HuggingFace Spaces |
| **Backend Deploy** | HuggingFace Spaces | Free Docker hosting with auto-deploy from GitHub |
| **Frontend Deploy** | Vercel | Static hosting with auto-deploy from GitHub |
 
---
 
## 📡 API Endpoints
 
### `GET /health`
Health check — used by monitoring tools and HF Spaces.
 
```json
Response: {"status": "healthy", "model": "distilbert-base-uncased-finetuned-sst-2-english"}
```
 
---
 
### `POST /analyze`
Classify sentiment for a single text string.
 
**Rate limit**: 30 requests/minute per IP
 
**Request:**
```json
{
  "text": "This product completely exceeded my expectations!"
}
```
 
**Response:**
```json
{
  "text": "This product completely exceeded my expectations!",
  "sentiment": "POSITIVE",
  "confidence": 0.9998,
  "processing_time_ms": 82.4
}
```
 
**Validation errors (422)**:
- Empty or whitespace-only text
- Text exceeding 5000 characters
---
 
### `POST /batch-analyze`
Classify sentiment for multiple texts in one call.
 
**Rate limit**: 10 requests/minute per IP  
**Max batch size**: 20 items
 
**Request:**
```json
{
  "texts": [
    {"text": "Amazing service, highly recommend!"},
    {"text": "Worst experience I've ever had."},
    {"text": "It was okay, nothing special."}
  ]
}
```
 
**Response:**
```json
{
  "results": [
    {"text": "Amazing service...", "sentiment": "POSITIVE", "confidence": 0.9997, "processing_time_ms": 76.1},
    {"text": "Worst experience...", "sentiment": "NEGATIVE", "confidence": 0.9991, "processing_time_ms": 74.8},
    {"text": "It was okay...", "sentiment": "NEGATIVE", "confidence": 0.6124, "processing_time_ms": 75.2}
  ],
  "count": 3
}
```
 
---
 
## 🖥️ Run Locally
 
### Prerequisites
- Python 3.10+
- Docker Desktop (optional, for container testing)
### Option A: Direct (No Docker)
 
```bash
# 1. Clone the repo
git clone https://github.com/Gagandeep61/sentiment-analysis.git
cd sentiment-analysis
 
# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
 
# 3. Install dependencies (~1GB due to PyTorch — takes a few minutes)
pip install -r requirements.txt
 
# 4. Run the API
uvicorn app.main:app --reload --port 8000
 
# 5. Open interactive docs
# http://localhost:8000/docs
```
 
### Option B: Docker
 
```bash
# Build (model baked into image at build time — ~2GB)
docker build -t sentiment-api .
 
# Run
docker run -p 7860:7860 sentiment-api
 
# Test
curl -X POST http://localhost:7860/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this!"}'
```
 
---
 
## 🧪 Testing the API
 
### via curl
 
```bash
# Single analysis
curl -X POST https://gagandeep61-sentiment-analysis.hf.space/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This is an incredible product, I highly recommend it!"}'
 
# Health check
curl https://gagandeep61-sentiment-analysis.hf.space/health
 
# Batch analysis
curl -X POST https://gagandeep61-sentiment-analysis.hf.space/batch-analyze \
  -H "Content-Type: application/json" \
  -d '{"texts": [{"text": "Great!"}, {"text": "Terrible."}]}'
```
 
### via Python
 
```python
import requests
 
response = requests.post(
    "https://gagandeep61-sentiment-analysis.hf.space/analyze",
    json={"text": "FastAPI makes building APIs genuinely enjoyable."}
)
print(response.json())
# {'text': '...', 'sentiment': 'POSITIVE', 'confidence': 0.9989, 'processing_time_ms': 91.2}
```
 
### via Interactive Docs
 
Visit https://gagandeep61-sentiment-analysis.hf.space/docs — FastAPI auto-generates a full Swagger UI where you can test every endpoint directly in the browser.
 
---
 
## 🔑 Key Technical Decisions
 
**Why DistilBERT (not a custom-trained model)?**  
Training a sentiment classifier takes weeks and labeled datasets. For production ML engineering, the key skill is **serving** models, not training them. DistilBERT is pre-trained on SST-2 (67M parameters) with 97% of BERT's accuracy at 40% the model size — ideal for CPU inference.
 
**Why CORS middleware?**  
Browsers block cross-origin requests by default. Since the frontend (Vercel domain) and backend (HF Spaces domain) are on different origins, `CORSMiddleware` is required — without it, every browser request would silently fail with a CORS error.
 
**Why rate limiting?**  
Running on HuggingFace Spaces free tier with a shared CPU. Without `slowapi`, a single bad actor could flood the endpoint, exhausting the container's resources for all users. Per-IP limits prevent this.
 
**Why validate before inference?**  
Pydantic validators run before the model touches the input. Empty strings and oversized payloads are rejected at the schema layer (422) — the model never runs on invalid data, which saves latency and prevents potential crashes.
 
**Why bake the model into the Docker image?**  
Downloading DistilBERT (~260MB) at container startup would add 30+ seconds to every cold start. The Dockerfile runs a Python one-liner during `docker build` to pre-download and cache the model in the image layer — cold starts are now just Python + FastAPI startup (~5 seconds).
 
---
 
## ⚠️ Known Limitations
 
- **Cold starts**: HuggingFace Spaces free tier sleeps containers after 15 minutes of inactivity. First request after sleep takes 25-45 seconds. This is a platform constraint, not a code issue.
- **CPU only**: No GPU acceleration on free tier — inference is ~80-120ms per request (acceptable for this use case).
- **Rate limits are in-memory**: `slowapi` resets on container restart. For multi-replica production, Redis-backed rate limiting would be needed.
- **CORS allows all origins** (`*`): Acceptable for a demo API. A production deployment should restrict to specific frontend domains.
---
 
## 📈 Performance
 
| Metric | Value |
|--------|-------|
| Inference latency (warm) | 80–120ms per request |
| Model size | ~260MB (DistilBERT) |
| Batch throughput | ~8–12 texts/second |
| Cold start (HF Spaces) | 25–45 seconds |
| Docker image size | ~2.1GB (PyTorch + model) |
 
---
 
## 🛣️ What's Next
 
- [ ] Migrate backend to Render for always-on serving (eliminates cold starts)
- [ ] Add Redis-backed rate limiting for production multi-replica support
- [ ] Add request logging + Prometheus metrics endpoint
- [ ] Integrate with Claude API for more nuanced multi-label sentiment (beyond POSITIVE/NEGATIVE)
- [ ] Add authentication (API keys) for per-user usage tracking
---
 
## 🎤 Interview Talking Points
 
1. **Serving vs Training**: "Most ML projects focus on training. This project focuses on the production side — FastAPI, Docker, REST API design, and deployment — which is what's actually used in engineering roles."
2. **Why CORS was necessary**: "The frontend on Vercel and backend on HuggingFace Spaces are different origins. Without CORSMiddleware, the browser refuses to send the request due to the Same-Origin Policy — I had to explicitly opt in to cross-origin calls."
3. **Docker's role**: "Docker guarantees the app runs the same in local dev, CI, and HuggingFace's infrastructure. I bake the model into the image at build time so cold starts don't require downloading 260MB on every wake-up."
4. **Rate limiting trade-offs**: "slowapi works for a single container. In a multi-replica setup, each instance has independent in-memory state, so a user could hit 30 req/min on each replica. Production-grade rate limiting needs a shared store like Redis — I know this limitation and documented it."
5. **Input validation**: "Pydantic validators reject invalid inputs before the model runs. This prevents edge cases like passing an empty string to the tokenizer and getting a nonsensical confidence score."
---
 
## 👤 Author
 
**Gagandeep Singh**  
GitHub: [@Gagandeep61](https://github.com/Gagandeep61)
 
---
 
*Built as a bridge project between ML training and ML production engineering.*
 