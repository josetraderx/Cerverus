#!/bin/bash

echo "CONFIGURING LOCAL DOCKER REGISTRY"

# Local registry for development
docker run -d \
  -p 5000:5000 \
  --restart=always \
  --name cerverus-registry \
  -v $(pwd)/registry-data:/var/lib/registry \
  registry:2

echo "Local registry running on localhost:5000"

# Tag and push images
echo "Tagging and pushing images..."

docker tag cerverus_cerverus-api:latest localhost:5000/cerverus/api:latest
docker tag cerverus_airflow-webserver:latest localhost:5000/cerverus/airflow:latest

docker push localhost:5000/cerverus/api:latest
docker push localhost:5000/cerverus/airflow:latest

echo "Images pushed to local registry"

# Verification
curl -s http://localhost:5000/v2/_catalog | jq '.'