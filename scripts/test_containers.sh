#!/bin/bash

echo "STARTING CERVERUS CONTAINERS TESTS"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging helpers
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Single service health test
test_service() {
    local service_name=$1
    local health_url=$2
    local max_retries=30
    local retry_count=0

    log_info "Testing $service_name..."
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -s -f "$health_url" > /dev/null 2>&1; then
            log_info "$service_name: HEALTHY"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        log_warn "$service_name: Attempt $retry_count/$max_retries - Waiting..."
        sleep 10
    done
    
    log_error "$service_name: FAILED - No response after $max_retries attempts"
    return 1
}

# Pre-validation
log_info "Validating dependencies..."
python3 scripts/validate_dependencies.py
if [ $? -ne 0 ]; then
    log_error "Dependency validation failed"
    exit 1
fi

# Build containers
log_info "Building containers..."
docker-compose build --no-cache
if [ $? -ne 0 ]; then
    log_error "Build failed"
    exit 1
fi

# Start services
log_info "Starting services..."
docker-compose up -d

# Wait for PostgreSQL
log_info "Waiting for PostgreSQL..."
sleep 20

# Initialize Airflow DB
log_info "Initializing Airflow DB..."
docker-compose exec -T airflow-webserver airflow db init
docker-compose exec -T airflow-webserver airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@cerverus.com

# Test services health
log_info "Testing services health..."

test_service "PostgreSQL" "postgresql://cerverus_user:cerverus_pass@localhost:5432/cerverus" # Special case
test_service "Redis" "http://localhost:6379" # Special case  
test_service "Cerverus API" "http://localhost:8000/health"
test_service "Airflow" "http://localhost:8080/health"

# Test ML functionality
log_info "Testing ML algorithms..."
python3 -c "
import sys
sys.path.append('src')
from cerverus.algorithms.meta_learner import MetaLearner
ml = MetaLearner()
print('ML algorithms imported successfully')
"

if [ $? -eq 0 ]; then
    log_info "ALL TESTS PASSED"
    log_info "Cerverus API: http://localhost:8000"
    log_info "Airflow UI: http://localhost:8080 (admin/admin)"
    log_info "PostgreSQL: localhost:5432"
else
    log_error "SOME TESTS FAILED"
    docker-compose logs
    exit 1
fi