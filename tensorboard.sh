#!/bin/bash

TENSORBOARD_LOGDIR="./artifacts/tensorboard"
TENSORBOARD_PORT=6006
TENSORBOARD_HOST="0.0.0.0"

echo "Starting TensorBoard at http://${TENSORBOARD_HOST}:${TENSORBOARD_PORT} ..."
tensorboard \
    --logdir "${TENSORBOARD_LOGDIR}" \
    --host "${TENSORBOARD_HOST}" \
    --port "${TENSORBOARD_PORT}"