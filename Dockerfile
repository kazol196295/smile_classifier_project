# Stage 1: Builder - install dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime - minimal production image
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY main.py config.py database.py models.py schemas.py inference.py train_local.py ./
COPY routers/ ./routers/
COPY templates/ ./templates/
COPY static/ ./static/
COPY model/ ./model/

# Create non-root user and set permissions
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p static/uploaded_images model \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600"]
