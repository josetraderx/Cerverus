# Cerverus

Financial Anomaly Detection System with Machine Learning

![CI](https://github.com/josetraderx/Cerverus/actions/workflows/ci.yml/badge.svg)

## Overview

Cerverus is an advanced financial anomaly detection system that combines multiple machine learning algorithms to identify suspicious patterns in market data. The system is designed with a scalable and modular architecture that enables real-time detection and historical analysis.

## System Architecture

### Main Components

#### 1. API Layer (`api/`)
- **Framework**: FastAPI with automatic documentation (Swagger/OpenAPI)
- **Main endpoints**:
  - `/api/v1/anomaly` - Anomaly detection
  - `/health` - System health check
  - `/docs` - Interactive documentation
- **Features**: Configurable CORS, structured error handling, Pydantic data validation

#### 2. Machine Learning Core (`src/cerverus/`)
- **Implemented algorithms**:
  - **Isolation Forest**: Tree-based anomaly detection
  - **Local Outlier Factor (LOF)**: Density-based detection
  - **Autoencoder**: Neural network for unsupervised detection
  - **LSTM Detector**: Recurrent neural network for time series
  - **Meta-Learner**: High-level ensemble combining multiple detectors

- **Model Architecture**:
  - Common base: [`BaseAnomalyDetector`](src/cerverus/models/base_detector.py)
  - Specialized feature engineering for financial data
  - Scoring system with uncertainty calibration
  - Temporal validation to prevent overfitting

#### 3. Data Processing (`src/cerverus/data/` and `src/cerverus/infrastructure/`)
- **ETL Pipeline**:
  - Extraction: External APIs (Yahoo Finance, SEC EDGAR)
  - Transformation: Data cleaning and normalization
  - Loading: PostgreSQL with optimized schema

- **Data Lake Architecture**:
  - **Bronze**: Raw unprocessed data
  - **Silver**: Clean and structured data
  - **Gold**: Aggregated data for analysis and modeling

#### 4. Orchestration (`airflow/`)
- **Implemented DAGs**:
  - [`fraud_detection_pipeline.py`](airflow/dags/fraud_detection_pipeline.py): Main detection pipeline
  - [`data_validation_dag.py`](airflow/dags/data_validation_dag.py): Data quality validation
- **Scheduling**: Daily executions with status monitoring

#### 5. Infrastructure (`infrastructure/` and `docker/`)
- **Containerization**:
  - Docker Compose for local development
  - Optimized images for production
  - Health checks and container monitoring

- **Cloud Infrastructure**:
  - Terraform for infrastructure as code
  - Kubernetes for container orchestration
  - Multiple environments: development, staging, production

#### 6. Monitoring and Observability
- **Prometheus**: System and model metrics
- **Grafana**: Real-time monitoring dashboards
- **Alertmanager**: Alert management and notifications

## Data Flow

```
External Sources → Ingestion → Data Lake (Bronze) → Processing → Data Lake (Silver) →
Feature Engineering → ML Models → Anomaly Detection → Alerts → Dashboard
```

## Configuration

### Configuration Structure (`config/`)

```
config/
├── environments/
│   ├── local.yml
│   ├── staging.yml
│   └── production.yml
├── models/
│   ├── tier1_config.yml
│   ├── tier2_config.yml
│   ├── tier3_config.yml
│   └── tier4_config.yml
└── pipelines/
    ├── data_ingestion.yml
    └── ml_training.yml
```

Purpose: Centralize parameters to facilitate deployments and testing in multiple environments.

## Infrastructure

### Infrastructure Structure (`infrastructure/`)

```
infrastructure/
├── terraform/
│   ├── environments/
│   │   ├── dev.tf
│   │   ├── staging.tf
│   │   └── prod.tf
│   ├── modules/
│   │   ├── database/
│   │   ├── kafka/
│   │   └── kubernetes/
│   └── main.tf
├── kubernetes/
│   ├── manifests/
│   │   ├── api-deployment.yaml
│   │   ├── ml-workers.yaml
│   │   └── kafka-cluster.yaml
│   └── helm/
│       └── cerverus-chart/
└── monitoring/
    ├── grafana/
    │   └── dashboards/
    ├── prometheus/
    │   └── rules/
    └── alertmanager/
        └── config/
```

## Development

### Main Technologies
- **Backend**: Python 3.11+, FastAPI, PostgreSQL
- **Machine Learning**: scikit-learn, TensorFlow, XGBoost
- **Orchestration**: Apache Airflow
- **Containerization**: Docker, Kubernetes
- **Monitoring**: Prometheus, Grafana
- **CI/CD**: GitHub Actions

### Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/josetraderx/Cerverus.git
   cd Cerverus
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Configure environment variables**:
   ```bash
   cp config/app.env.example config/app.env
   # Edit config/app.env with necessary configurations
   ```

4. **Start local services**:
   ```bash
   docker-compose up -d
   ```

5. **Run migrations**:
   ```bash
   python scripts/migrate_db.py
   ```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html
```

## Deployment

### Local Development
```bash
docker-compose -f docker-compose.yml up -d
```

### Production
```bash
docker-compose -f docker/docker-compose.prod.yml up -d
```

### Kubernetes
```bash
kubectl apply -f infrastructure/kubernetes/manifests/
```

## Project Roadmap

The project follows a phased implementation:

- **Phase 0**: Agile preparation and initial PoC
- **Phase 1**: MVP with enterprise foundations
- **Phase 2**: Scalable detection engine
- **Phase 3**: Intelligent orchestration
- **Phase 4**: Enterprise observability
- **Phase 5**: Smart scaling
- **Phase 6**: Business intelligence
- **Phase 7**: Governance and innovation

For more details on each phase, consult [`docs/architecture.md`](docs/architecture.md).

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Submit a Pull Request

## License

This project is licensed under the MIT license - see the [LICENSE](LICENSE) file for details.
