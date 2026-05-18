# ─── IMPORTS ───────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from pydantic import BaseModel, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from typing import List
import os
import time

# ─── ENVIRONMENT ───────────────────────────────────────────────────────────────

load_dotenv()  # Loads .env file for local dev

# ─── RATE LIMITER SETUP ────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ─── APP INITIALIZATION ────────────────────────────────────────────────────────

app = FastAPI(
    title="Sentiment Analysis API",
    description="Classifies text as POSITIVE or NEGATIVE using DistilBERT",
    version="1.0.0"
)

# Register rate limiter with app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── CORS MIDDLEWARE ───────────────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing
# Without this, your browser will BLOCK the frontend from calling the backend
# because they are on different domains (Vercel vs HuggingFace)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production, replace * with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODEL LOADING ─────────────────────────────────────────────────────────────
# This runs ONCE when the server starts — not on every request
# HuggingFace auto-downloads the model on first run (~260MB)

print("Loading DistilBERT model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Model loaded successfully.")

# ─── INPUT SCHEMAS (Pydantic) ──────────────────────────────────────────────────
# Pydantic auto-validates incoming JSON. If validation fails, FastAPI
# returns a clear 422 error BEFORE the model even runs.

class TextInput(BaseModel):
    text: str

    @validator("text")
    def validate_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Text cannot be empty or whitespace only.")
        if len(value) > 5000:
            raise ValueError("Text must be under 5000 characters.")
        return value


class BatchInput(BaseModel):
    texts: List[TextInput]

    @validator("texts")
    def validate_batch(cls, value):
        if len(value) == 0:
            raise ValueError("Batch cannot be empty.")
        if len(value) > 20:
            raise ValueError("Batch cannot exceed 20 items.")
        return value


# ─── OUTPUT SCHEMA ─────────────────────────────────────────────────────────────

class SentimentResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    processing_time_ms: float


# ─── HELPER FUNCTION ───────────────────────────────────────────────────────────

def run_inference(text: str) -> dict:
    """Runs DistilBERT inference and returns structured result."""
    start = time.time()
    result = sentiment_pipeline(text)[0]
    elapsed = round((time.time() - start) * 1000, 2)
    return {
        "text": text,
        "sentiment": result["label"],
        "confidence": round(result["score"], 4),
        "processing_time_ms": elapsed
    }


# ─── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Root endpoint — confirms API is reachable."""
    return {"message": "Sentiment Analysis API is running.", "version": "1.0.0"}


@app.get("/health")
def health():
    """Health check endpoint — used by HF Spaces and monitoring tools."""
    return {"status": "healthy", "model": "distilbert-base-uncased-finetuned-sst-2-english"}


@app.post("/analyze", response_model=SentimentResult)
@limiter.limit("30/minute")
def analyze(request: Request, body: TextInput):
    """
    Analyze sentiment of a single text string.
    
    - Input: {"text": "I love this product!"}
    - Output: {"text": "...", "sentiment": "POSITIVE", "confidence": 0.9998, "processing_time_ms": 45.2}
    """
    try:
        return run_inference(body.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")


@app.post("/batch-analyze")
@limiter.limit("10/minute")
def batch_analyze(request: Request, body: BatchInput):
    """
    Analyze sentiment for multiple texts (max 20).
    
    - Input: {"texts": [{"text": "..."}, {"text": "..."}]}
    - Output: list of SentimentResult objects
    """
    try:
        results = []
        for item in body.texts:
            results.append(run_inference(item.text))
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")