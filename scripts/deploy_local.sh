#!/bin/bash
# Simple deploy wrapper for local development
set -e

ENV=${1:-development}
echo "Deploying environment: $ENV"

docker-compose -f docker-compose.yml up -d --build

echo "Deploy complete"
