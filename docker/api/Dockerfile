# Build arguments for reproducibility - DEBE ir ANTES de FROM
ARG PYTHON_VERSION=3.11-slim
ARG POETRY_VERSION=1.6.1

# Stage 1: Dependency Builder
FROM python:${PYTHON_VERSION} AS dependency-builder   

# RE-DECLARAR ARG en este stage
ARG POETRY_VERSION=1.6.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=3000

# Install system dependencies required for compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc-dev \
    libffi-dev \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency files
COPY pyproject.toml poetry.lock* README.md ./

# Install Poetry with dependency management configuration
# Separate layer for Poetry setup to enable caching
RUN python -m pip install --upgrade pip \
    && python -m pip install "poetry==$POETRY_VERSION" \
    && poetry config virtualenvs.create true \
    && poetry config virtualenvs.in-project true \
    && poetry config installer.max-workers 1

# Create minimal package structure
RUN mkdir -p src/cerverus
COPY src/cerverus/__init__.py src/cerverus/__init__.py

# Install dependencies and package separately for better debugging
# --no-root prevents Poetry from installing the local package
RUN poetry install --no-root --only main --no-interaction --no-ansi

# Install cerverus as editable package
RUN /build/.venv/bin/pip install -e .

# Stage 2: Runtime
FROM python:${PYTHON_VERSION} AS runtime

# Metadata
LABEL maintainer="cerverus-team"
LABEL description="Cerverus Fraud Detection API"
LABEL version="2.0"

# Runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH"

# Runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with fixed UID/GID for reproducibility
RUN groupadd -r -g 1001 appuser && useradd -r -u 1001 -g appuser appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=dependency-builder /build/.venv /app/.venv

# Copy application source with ownership
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser tools/ ./tools/

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Port exposition
EXPOSE 8000