#!/bin/bash

IP=${IP:-0.0.0.0}
PORT=${PORT:-8070}
WORKERS=${WORKERS:-1}

# Command to run the application
python -m uvicorn embedding_service.app.main:app --host ${IP} --port ${PORT} --workers ${WORKERS}
