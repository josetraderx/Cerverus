# Arquitectura

Descripción de la arquitectura del sistema.
Phase 0 — Agile Preparation (2-4 weeks)
Hybrid approach:

From v1: Rapid and reproducible setup
From v2: Cloud infrastructure from the start
Scope:

Basic infrastructure:
Cloud account (AWS/Azure/GCP) with S3, EC2, RDS
Docker Compose for local development
Basic CI/CD:
GitHub Actions with tests + linting
Deployment to automatic staging
Rapid PoC:
Isolation Forest in 100 symbols (not 500)
Simple dashboard in Grafana
Deliverables:

Functional docker-compose.yml
Basic CI/CD pipeline
Detection PoC with initial metrics
PoC recomendado (detalle)

- Objetivo: validar que un modelo de Machine Learning (Isolation Forest) puede detectar anomalías relevantes en datos de mercado reales.
- Alcance técnico propuesto: probar sobre un conjunto de ~500 símbolos (acciones) usando un pipeline reproducible:
	- Origen de datos: Yahoo Finance (u otra API de mercado) →
	- Almacenamiento temporal: PostgreSQL local para el experimento →
	- Procesamiento y análisis: scripts en Python que limpian, generan features y aplican Isolation Forest.
- Criterio de éxito: con datos históricos conocidos, el PoC debe detectar más de 3 anomalías históricas inyectadas o previamente identificadas.

Recomendación práctica (para evitar bloqueos de desarrollo):

- Implementar dos variantes del PoC:
	1) PoC rápido para desarrollo: datos reducidos (ej. 100 símbolos) y flujo local, enfoque en velocidad de iteración.
	2) Piloto de validación: ejecutar la versión de ~500 símbolos para comprobar escalabilidad y tasa de detección real.

- Por qué esta doble vía: el PoC rápido permite iterar sin coste alto; el piloto de 500 aporta evidencia más representativa y sirve como criterio de aceptación para avanzar a Phase 1.
Why this combination?
Like brushing with a manual toothbrush but with professional toothpaste: fast but effective.

Phase 1 — MVP with Enterprise Foundations (2-3 months)
Hybrid Approach:

From v1: Minimal Functional API
From v2: Resilient Patterns from the Start
Scope:

Multi-Source Ingestion:
Yahoo Finance + Alpha Vantage (as v2)
Polymorphic Adapters (as v2)
Robust API:
Endpoints /ingest, /events, /alerts (as v1)
Circuit Breakers + Dead Letter Queues (as v2)
Hybrid Storage:
ETL for Sensitive Data (v2)
ELT for Aggregated Data (v2)
Deliverables:

Enterprise-Resilient API
MVP Pipeline with Design Patterns
Contract Testing + Integration
Why this combination?
Like building a house on a skyscraper foundation: simple on the outside, strong on the inside.

Phase 2 — Scalable Detection Engine (3-4 months)
Hybrid approach:

From v1: Simple models first
From v2: Architecture for complex models
Scope:

Basic feature store:
Transactional + temporal features (v2)
Market microstructure (v2)
Layered models:
Tier 1: Z-score, CUSUM (v2)
Tier 2: Isolation Forest + XGBoost (v1 + v2)
Tier 3: Reserved for LSTM/GNN (future)
MLflow for tracking:
Model versioning (v2)
Reproducible experiments (v2)
Deliverables:

3 operational detection levels
Automated training pipeline
Performance metrics (precision, recall)
Why this combination?
Like having a workshop with manual tools and space for industrial machines later.

Phase 3 — Intelligent Orchestration (3-4 months)
Hybrid approach:

From v1: Airflow for orchestration
From v2: Lightweight streaming
Scope:

Modular DAGs:
Airflow with TaskGroups (v2)
Ingestion → preprocessing → ML → alerting (v1)
Practical streaming:
Kafka + Flink for speed layer (v2)
Only for critical alerts (not everything)
Containerization:
Docker multi-stage (v2)
Basic Kubernetes (staging only) (v2)
Deliverables:

Batch orchestration + streaming
Deployment to Kubernetes (staging)
Basic load testing
Why this combination?
Like having an automatic irrigation system: drip for normal plants, sprinklers for important ones.

Phase 4 — Enterprise Observability (2-3 months)
Hybrid Approach:

From v1: Essential Monitoring
From v2: Security and Compliance
Scope:

Full Observability:
Prometheus + Grafana (v2)
Jaeger for tracing (v2)
Structured Logs (v2)
Practical Security:
OAuth2/JWT (v2)
Secret Management (v2)
Basic Encryption (v1)
Essential Compliance:
GDPR Implemented (v2)
Access Audit (v2)
Deliverables:

Operational and Business Dashboards
Basic Security System
Compliance Documentation
Why this combination?
Like having an alarm system: detect intruders without turning your house into a vault.

Phase 5 — Smart Scaling (2-3 months)
Hybrid Approach:

From v1: Practical Optimization
From v2: Preparing for Greatness
Scope:

Model Optimization:
Automatic Retraining (v2)
A/B Testing Champion/Challenger (v2)
Controlled Scaling:
Data Partitioning (v2)
Caching with Redis (v2)
Kubernetes in Production (optional)
Cost Optimization:
Cost Monitoring (v2)
Horizontal Scaling Strategies (v2)
Deliverables:

Automatic Retraining System
Documented Scaling Strategy
Cost-Benefit Reports
Why this combination?
Like having an affordable car but with room to put a powerful engine in it later.

Phase 6 — Business Intelligence (2-3 months)
Hybrid Approach:

From v1: Useful Reporting
From v2: Regulatory Readiness
Scope:

Data Warehouse Medallion:
Bronze/Silver/Gold (v2)
Only for critical data (not all)
Executive Dashboards:
Business Metrics (v2)
System ROI (v2)
Regulatory Reports:
Automated SARs (v2)
Basic FINRA (v2)
Deliverables:

Dashboards for Executives
Automated Regulatory Reports
Functional Data Warehouse
Why this combination?
How to have financial reports: sufficient for the bank, without a full audit.

Phase 7 — Governance and Future (1-2 months)
Hybrid Approach:

From v1: Practical Maintenance
From v2: Controlled Innovation
Scope:

Basic Governance:
Simplified RACI (v2)
Model Audit (v2)
Practical Innovation:
Experiments with LLMs for documents (v2)
Edge Computing for Specific Use Cases (v2)
Quantum-Resistant Cryptography (optional)
Deliverables:

Governance Plan
Innovation Lab
Future Roadmap