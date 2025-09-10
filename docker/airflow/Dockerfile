FROM apache/airflow:2.8.1

# Airflow ambient variables
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags \ 
    AIRFLOW__CORE__PLUGINS_FOLDER=/opt/airflow/plugins \ 
    AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True \ 
    AIRFLOW__CORE__LOAD_EXAMPLES=False

# Switch to root to install dependencies
USER root

# System dependencies for ML
RUN apt-get update && apt-get install -y \ 
    gcc\ 
    g++ \ 
    && rm -rf /var/lib/apt/lists/*

# Return to user airflow
USER airflow

# Copy specific Airflow requirements
COPY docker/airflow/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy DAGs and plugins
COPY airflow/dags/ ${AIRFLOW_HOME}/dags/
COPY airflow/plugins/ ${AIRFLOW_HOME}/plugins/
COPY src/ /opt/airflow/src/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \ 
    CMD curl -f http://localhost:8080/health || exit 1