#!/bin/bash
# Start the FastAPI backend server

echo "Starting FastAPI backend on http://localhost:8001..."
poetry run uvicorn api.app.main:app --host 0.0.0.0 --port 8001 --reload
