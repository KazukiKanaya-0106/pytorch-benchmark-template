#!/bin/bash

MLFLOW_BACKEND_URI="file:./artifacts/mlflow"
MLFLOW_PORT=5000
MLFLOW_HOST="0.0.0.0"

echo "Starting MLflow UI at http://${MLFLOW_HOST}:${MLFLOW_PORT} ..."
mlflow ui \
    --host "${MLFLOW_HOST}" \
    --port "${MLFLOW_PORT}" \
    --backend-store-uri "${MLFLOW_BACKEND_URI}"