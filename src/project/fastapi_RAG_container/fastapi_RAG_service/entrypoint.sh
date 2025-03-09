#!/bin/bash

cd /fastapi_RAG_container

IP=${IP:-0.0.0.0}
PORT=${PORT:-8090}
WORKERS=${WORKERS:-1}

# Command to run the application
python -m uvicorn fastapi_RAG_service.app.main:app --host ${IP} --port ${PORT} --workers ${WORKERS}
