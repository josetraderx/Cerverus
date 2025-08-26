FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar metadatos primero para cache
COPY pyproject.toml poetry.lock* README.md /app/

# Copiar código fuente para Poetry install
COPY src /app/src

# Instalar Poetry y dependencias
RUN python -m pip install --upgrade pip && \
    python -m pip install poetry && \
    poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --only main --no-interaction --no-ansi

# ✅ AGREGAR: Copiar carpetas faltantes
COPY api /app/api
COPY tools /app/tools

# Variables de entorno
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# CMD correcto
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]