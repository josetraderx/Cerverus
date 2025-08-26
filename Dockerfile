FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar metadatos de dependencias primero para aprovechar cache
# Include README.md so Poetry can read the declared readme during install
COPY pyproject.toml poetry.lock* README.md /app/

# Copy package sources so Poetry can install the project package during build
# We keep this minimal to allow layer caching for dependency metadata
COPY src /app/src

# Instalar Poetry y dependencias en un venv in-project (.venv)
RUN python -m pip install --upgrade pip && \
    python -m pip install poetry && \
    poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --only main --no-interaction --no-ansi

# Copiar el resto del código
COPY . /app

# Exponer venv en PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

EXPOSE 8000

# Comando por defecto (usa uvicorn del venv)
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]