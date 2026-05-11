# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application ──────────────────────────────────────────────────────────
COPY . .

# ── Ensure catalog and index exist ────────────────────────────────────────────
# catalog.json and lancedb_index/ are committed to repo.
# Verify the table exists at image build time so deploys fail fast if artifacts
# are missing or incomplete.
RUN test -f catalog/catalog.json || (echo "ERROR: catalog/catalog.json missing. Run: python catalog/scraper.py" && exit 1)
RUN python -c "from agent.retriever import _load_lancedb_table; _load_lancedb_table(); print('LanceDB table verified')"

# ── Create logs directory ─────────────────────────────────────────────────────
RUN mkdir -p logs

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]
