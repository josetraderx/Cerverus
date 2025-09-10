#!/bin/bash

set -e

echo "CERVERUS BUILD & DEPLOY PIPELINE"

# Variables
VERSION=${1:-latest}
REGISTRY=${REGISTRY:-localhost:5000}
ENVIRONMENT=${ENVIRONMENT:-development}

log_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

# Pre-build validations
log_info "Running pre-build validations..."
python3 scripts/validate_dependencies.py

# Build images
log_info "Building images with version: $VERSION"
docker-compose build

# Tag images
log_info "Tagging images..."
docker tag cerverus_cerverus-api:latest $REGISTRY/cerverus/api:$VERSION
docker tag cerverus_airflow-webserver:latest $REGISTRY/cerverus/airflow:$VERSION

# Push to registry
log_info "Pushing to registry: $REGISTRY"
docker push $REGISTRY/cerverus/api:$VERSION
docker push $REGISTRY/cerverus/airflow:$VERSION

# Deploy
log_info "Deploying environment: $ENVIRONMENT"
COMPOSE_FILE="docker-compose.yml"
if [ "$ENVIRONMENT" != "development" ]; then
    COMPOSE_FILE="docker-compose.${ENVIRONMENT}.yml"
fi

docker-compose -f $COMPOSE_FILE up -d

# Health checks
log_info "Running health checks..."
bash scripts/test_containers.sh

log_info "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY"
log_info "📊 Metrics available at: http://localhost:8000/metrics"
log_info "🌐 Airflow UI: http://localhost:8080"