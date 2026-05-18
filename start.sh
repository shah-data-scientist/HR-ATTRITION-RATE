#!/bin/bash
set -e

# Initialise the SQLite database (creates tables and seeds users)
python -m database.init_db

# Start FastAPI with a single worker — multi-worker (fork) is unreliable
# in constrained container environments like HuggingFace Spaces
uvicorn api.app.main:app --host 0.0.0.0 --port 8001 --workers 1 &
API_PID=$!

# Start background job worker (processes async report jobs)
python -m scripts.worker &
WORKER_PID=$!

# Kill background processes when this script exits (e.g. Streamlit crash)
trap 'kill $API_PID $WORKER_PID 2>/dev/null || true' EXIT

# Wait until the API is healthy before launching the UI
echo "Waiting for API to be ready..."
until curl -sf http://localhost:8001/health > /dev/null; do
    sleep 2
done
echo "API is ready."

# Start Streamlit in the foreground — keeps the container alive.
# Port 7860 is the only port HuggingFace Spaces exposes.
streamlit run ui/app_authenticated.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
