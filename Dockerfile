# Reproducible, CPU-only runtime for the TV genre classification pipeline
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

# make is used as the task runner; build-essential covers any wheel that needs compiling
RUN apt-get update \
 && apt-get install -y --no-install-recommends make build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU build of PyTorch first (much smaller than the CUDA build),
# then the remaining pinned requirements.
COPY requirements.txt .
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY . .
RUN pip install -e .

# Non-root user for safety
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Default: classify the 400 unlabelled shows. Override, e.g.
#   docker run --rm -v "$PWD":/app tv-genre-classification make all
CMD ["python", "-m", "src.models.predict_model"]
