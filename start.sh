#!/bin/bash
set -e

# Initialise the SQLite database (creates tables and seeds users)
python -m database.init_db

# Start FastAPI in the background
uvicorn api.app.main:app --host 0.0.0.0 --port 8001 &
API_PID=$!

# Wait until the API is responding before launching the UI
echo "Waiting for API to be ready..."
until curl -sf http://localhost:8001/health > /dev/null; do
    sleep 2
done
echo "API is ready."

# Start Streamlit on port 7860 (the only port HuggingFace Spaces exposes)
streamlit run ui/app_authenticated.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false

# If Streamlit exits, stop the API too
kill $API_PID
