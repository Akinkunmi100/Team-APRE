FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./worker ./worker
COPY ./config ./config

# Start Celery worker
CMD ["celery", "-A", "worker.tasks", "worker", "--loglevel=info"]