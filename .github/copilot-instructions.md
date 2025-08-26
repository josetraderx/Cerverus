

# Copilot / AI Assistant Instructions for Cerverus

Purpose
- Help AI coding assistants become productive quickly in this repo (Cerverus: financial anomaly detection API + pipelines).
- This is an enterprise-grade financial anomaly detection system following AWS Well-Architected Framework principles, balancing rapid delivery with production reliability, security, and scalability.

Quick orientation
- Project layout: top-level `api/` (API FastAPI app), `src/cerverus/` (library code: data, features, models, monitoring, utils), `pipelines/` (DAGs and tasks), `notebooks/` (EDA and experiments), and `Dockerfile` + `docker-compose.yml` for local development.
- The FastAPI entrypoint is in `api/app/main.py` (root GET `/` smoke check). Endpoints live under `api/app/endpoints/`.
- Core configuration and security helpers are in `api/app/core/` (e.g., `config.py`, `security.py`).
- The project follows a phased implementation approach (Phases 0-7) with specific deliverables and timelines.

Big picture / architecture notes
- The system is split into a micro-library (`src/cerverus/) containing data extraction/cleaning/transformation, features (microstructure + technical indicators), and tiered models (tier1..tier4). The API (`api/`) uses these library modules to expose detection endpoints and ingestion.
- Data flow: ingestion -> `src/cerverus/data` (extraction, cleaning, transformation) -> `src/cerverus/features` -> `src/cerverus/models` -> API endpoints in `api/app/endpoints`.
- Notebooks under `notebooks/` are used for EDA and model training experiments; they are not intended in production images.
- Architecture follows zero-trust security model with defense-in-depth, automated recovery mechanisms, and cost-aware design principles.
- System is designed for failure with graceful degradation and multi-AZ deployment for high availability.

Developer workflows (how to build, test, debug)
- Virtualenv: project contains a `venv/` locally; prefer creating a fresh venv or use Poetry once initialized.
- Tests: pytest is used (see `src/tests/` and `api/tests/` patterns). Run `pytest -q` from repo root. All tests must avoid network calls and use mocks for external services.
- Run API locally:
  - With Python directly: `uvicorn api.app.main:app --reload --port 8000` (from repo root).
  - With Docker Compose (dev): `docker-compose up --build` (note: compose mounts `./api` and `./notebooks` for dev hot-reload).
- Lint/format: the project uses `black`, `pylint`, and `isort` in dev dependencies. Use them with default configs unless file contains project-specific overrides.
- Infrastructure as Code: Terraform configurations are in `infra/` for AWS resources. Always validate with `terraform plan` before applying.

Project-specific conventions & patterns
- Models are organized by tiers in `src/cerverus/models/tier{1..4}.py`. Tier1 is statistical, tier2 uses IsolationForest/XGBoost, tier3/4 reserved for advanced models. Follow the established tier structure when adding new models.
- Minimal placeholder implementations are common; follow existing function names/signatures (e.g., `clean(df)`, `transform(df)`, `extract()` in `src/cerverus/data`). Never remove placeholders without adding implementations and tests.
- API routers use FastAPI `APIRouter` under `api/app/endpoints/` and mount under prefixes like `/anomaly`. Keep router prefixes stable to avoid breaking clients.
- Config via Pydantic BaseSettings in `api/app/core/config.py`. Place secrets in `.env` and prefer `python-dotenv` usage when running locally. Never commit secrets to version control.
- Feature engineering follows medallion architecture (bronze/silver/gold) with feature store implementation for online/offline features.
- All external dependencies must be mocked in tests using fixtures in `tests/conftest.py`.

Integration points & external dependencies
- External services: Postgres (docker-compose `db`), Kafka is referenced in requirements (dev), MLflow for model tracking, and S3/RDS in roadmap — treat these as optional in local dev but required in production.
- AWS Services: S3 for data storage, RDS for transactional data, ECR for container registry, EKS for container orchestration, CloudWatch for monitoring, Secrets Manager for secrets.
- Docker: use the root `Dockerfile` (production) and `docker-compose.yml` (dev). The Dockerfile must use multi-stage builds and not bundle notebooks into production images.
- Monitoring: Prometheus for metrics, Grafana for dashboards, Jaeger for distributed tracing. All services must emit structured logs and metrics.

What to change or watch for (guidance for AI agents editing code)
- Do not remove placeholder functions without adding implementations and tests. Instead, add unit tests covering new behavior and validate with `pytest`.
- When editing API routes, keep router prefixes and response contracts stable; add new endpoints under `api/app/endpoints/` and update the router inclusion in `api/app/main.py` if needed.
- When changing Dockerfiles, prefer multi-stage builds and avoid installing dev tools in the runtime image. Do not include `notebooks/` in production images.
- Tests must run with `pytest` and avoid network calls; mock external services (Postgres, Kafka, S3). Use `tests/conftest.py` fixtures to setup mocks.
- Always follow the phased approach when implementing new features. Phase 0 focuses on minimal viable infrastructure, Phase 1 on MVP with enterprise foundations, etc.
- Security is paramount: never hardcode credentials, always use secrets management, implement least privilege access, and encrypt sensitive data at rest and in transit.
- Implement proper error handling with circuit breakers for external service calls and dead letter queues for failed messages.
- All code changes must include appropriate observability: structured logging, metrics emission, and tracing spans.

Files to inspect when making changes
- `api/app/main.py` — API entrypoint and mounted routers
- `api/app/core/config.py`, `api/app/core/security.py` — settings and auth helpers
- `src/cerverus/data/*` — ingestion/cleaning/transform
- `src/cerverus/features/*` — feature engineering
- `src/cerverus/models/*` — model implementations
- `docker-compose.yml`, `Dockerfile` — container/build patterns
- `infra/` — Terraform configurations for AWS resources
- `.github/workflows/` — CI/CD pipeline definitions
- `ROADMAP.md` — project goals and priorities (useful for feature/context decisions)

Examples (copyable guidance for common tasks)
- Add a new anomaly endpoint that calls tier1 detector:
  - Create `api/app/endpoints/new_detector.py` exposing an `APIRouter(prefix="/detect")` and implement `@router.post("/tier1")` that imports `from src.cerverus.models.tier1 import detect` and returns `detect(payload)`.
  - Add circuit breaker pattern using the `@circuit` decorator for external service calls.
  - Add a test in `tests/test_endpoints.py` using `fastapi.testclient.TestClient` pointing at `api.app.main`.
  - Update the OpenAPI documentation by adding appropriate request/response models.
- Export Poetry lock to requirements for Docker builds:
  - `poetry export -f requirements.txt --without-hashes -o requirements.txt`
- Add new AWS infrastructure:
  - Create Terraform module in `infra/modules/`
  - Reference module in `infra/environments/{dev|prod}/main.tf`
  - Run `terraform plan` and `terraform apply` with appropriate workspace
  - Update outputs in `infra/environments/{dev|prod}/outputs.tf`

Limitations
- The repository contains many placeholder files and incomplete tests; some behaviors are intentionally unimplemented. Validate changes with unit tests and `uvicorn` smoke runs.
- Phase 3-7 features are in development and may change. Check `ROADMAP.md` for current priorities.
- Always follow the principle of "mechanical sympathy" - understand the underlying systems before making changes.
- Performance is critical - all changes must be benchmarked and monitored for impact.

If you want me to iterate
- I can add more concrete examples (snippets) or initialize `.github/workflows/ci.yml` for lint/tests. Tell me which area to expand.
- I can provide detailed guidance on implementing specific phases from the roadmap.
- I can expand on security patterns, monitoring strategies, or compliance requirements.