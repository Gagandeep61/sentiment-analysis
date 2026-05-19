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
    allow_origins=[
    "https://YOUR_APP.vercel.app",
    "http://localhost:8000",  # Keep for local dev
],
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
    model="distilroberta-base"
)
print("Model loaded successfully.")

# ─── SARCASM DETECTION LAYER ───────────────────────────────────────────────

def detect_sarcasm_contradiction(text: str) -> dict:
    """
    Detects sarcasm by looking for positive/negative word contradictions.
    
    Returns:
        {
            "is_sarcastic": bool,
            "confidence": float (0-1),
            "reason": str
        }
    """
    text_lower = text.lower().strip()
    
    # POSITIVE WORDS - things people say when they mean it positively
    positive_indicators = {
        'great', 'amazing', 'wonderful', 'awesome', 'perfect', 'love', 
        'excellent', 'fantastic', 'brilliant', 'superb', 'fantastic',
        'marvelous', 'gorgeous', 'beautiful', 'happy', 'glad', 'pleased'
    }
    
    # NEGATIVE CONTEXT WORDS - things that indicate a negative situation
    negative_context = {
        'hate', 'fail', 'failed', 'crash', 'crashed', 'broke', 'broken',
        'worst', 'awful', 'terrible', 'horrible', 'disgusting', 'bad',
        'issue', 'problem', 'broken', 'stuck', 'lost', 'disaster',
        'useless', 'waste', 'disappointing', 'frustrating', 'annoying'
    }
    
    # SARCASM MARKERS - phrases that often signal sarcasm
    sarcasm_markers = [
        r'\b(oh\s+)?(great|perfect|wonderful|amazing|sure|yeah)\s*[,\.!]',  # "Oh great." or "Perfect,"
        r'(right|sure|yeah)\s+(right|ok|buddy)',  # "Yeah, right"
        r'just\s+what\s+i\s+(need|needed)',  # "Just what I needed" (usually sarcastic)
        r'(that\s+)?would\s+be\s+(great|perfect|wonderful)',  # "That would be great" (often sarcastic)
        r'oh\s+(wonderful|fantastic|marvelous)',  # "Oh wonderful" (tone matters)
    ]
    
    import re
    
    # Check for sarcasm markers
    for marker_pattern in sarcasm_markers:
        if re.search(marker_pattern, text_lower):
            return {
                "is_sarcastic": True,
                "confidence": 0.6,
                "reason": "Sarcasm phrase pattern detected"
            }
    
    # Check for contradiction: positive words + negative context
    words_in_text = set(text_lower.split())
    has_positive = bool(words_in_text & positive_indicators)
    has_negative_context = bool(words_in_text & negative_context)
    
    if has_positive and has_negative_context:
        return {
            "is_sarcastic": True,
            "confidence": 0.5,
            "reason": "Contradiction detected (positive words + negative situation)"
        }
    
    return {
        "is_sarcastic": False,
        "confidence": 0.0,
        "reason": "No sarcasm indicators found"
    }

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
    sarcasm_detected: bool = False
    sarcasm_note: str | None = None


# ─── HELPER FUNCTION ───────────────────────────────────────────────────────────

def run_inference(text: str) -> dict:
    """Runs RoBERTa inference with sarcasm detection layer."""
    start = time.time()
    
    # Get base sentiment from RoBERTa
    result = sentiment_pipeline(text)[0]
    base_sentiment = result["label"]
    base_confidence = result["score"]
    
    # Check for sarcasm using rule-based layer
    sarcasm_detection = detect_sarcasm_contradiction(text)
    is_sarcastic = sarcasm_detection["is_sarcastic"]
    sarcasm_confidence = sarcasm_detection["confidence"]
    
    # If strong sarcasm detected, flip the sentiment
    if is_sarcastic and sarcasm_confidence >= 0.5:
        final_sentiment = "NEGATIVE" if base_sentiment == "POSITIVE" else "POSITIVE"
        # Lower confidence because the model might be confused
        final_confidence = max(0.5, base_confidence * 0.7)
    else:
        final_sentiment = base_sentiment
        final_confidence = base_confidence
    
    elapsed = round((time.time() - start) * 1000, 2)
    
    return {
        "text": text,
        "sentiment": final_sentiment,
        "confidence": round(final_confidence, 4),
        "processing_time_ms": elapsed,
        "sarcasm_detected": is_sarcastic,
        "sarcasm_note": sarcasm_detection["reason"] if is_sarcastic else None
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