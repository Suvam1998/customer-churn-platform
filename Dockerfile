# ==========================================================================
# Base image for API and dashboard services.
# Uses Python 3.12 (stable ML wheel availability) rather than the host's 3.14.
# ==========================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for scientific/ML wheels (kept minimal).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000 8501

# Default: run the API. docker-compose overrides the command per service.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
