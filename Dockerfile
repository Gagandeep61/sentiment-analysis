# ── BASE IMAGE ────────────────────────────────────────────────────────────────
# Official Python 3.10 on Debian Slim (small, production-grade Linux)
FROM python:3.10-slim

# ── SYSTEM SETUP ──────────────────────────────────────────────────────────────
# Set working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files (saves space)
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python output buffering (so logs appear in real time)
ENV PYTHONUNBUFFERED=1

# ── DEPENDENCIES ──────────────────────────────────────────────────────────────
# Copy requirements first (Docker caches this layer — speeds up rebuilds)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── PRE-DOWNLOAD MODEL ────────────────────────────────────────────────────────
# Download DistilBERT into the image at build time
# This means deployment startup is fast (model is already there)
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"

# ── APPLICATION CODE ──────────────────────────────────────────────────────────
COPY app/ ./app/

# ── PORT EXPOSURE ─────────────────────────────────────────────────────────────
# HuggingFace Spaces requires port 7860
EXPOSE 7860

# ── STARTUP COMMAND ───────────────────────────────────────────────────────────
# 0.0.0.0 = listen on all network interfaces (required in containers)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]