FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code changes don't invalidate the pip layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ml ml
COPY app app
COPY agent agent

# Reproducible build: regenerate the (seeded) dataset, retrain, and rebuild the
# vector index inside the image. Same seeds + pinned versions -> same metrics,
# which the CI regression check verifies.
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}
# auto = bge-small embeddings (downloaded into the image at build time), tfidf if unavailable
ARG PLANTPULSE_EMBEDDER=auto
ENV PLANTPULSE_EMBEDDER=${PLANTPULSE_EMBEDDER}
RUN python -m ml.data_generation \
    && python -m ml.train \
    && python -m ml.vector_store

# Hugging Face Spaces runs containers as uid 1000; Render/Fly set $PORT.
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
