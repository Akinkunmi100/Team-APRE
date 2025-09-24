# Multi-stage Dockerfile for AI Review Engine

# Stage 1: Base image with Python and dependencies
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    chromium-driver \
    chromium \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy language model
RUN python -m spacy download en_core_web_sm

# Stage 2: Application
FROM base as app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/scraped data/models data/cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/app/data/cache
ENV HF_HOME=/app/data/cache

# Expose ports
EXPOSE 8000 8501

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
