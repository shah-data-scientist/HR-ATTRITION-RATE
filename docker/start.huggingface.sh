#!/bin/bash
set -e

# Initialise SQLite database (schema + seed data)
python -u -m database.init_db || echo "DB init failed, continuing"

# Start FastAPI in the background
uvicorn api.app.main:app --host 0.0.0.0 --port 8001 --workers 2 &

# Start Streamlit in the foreground (keeps the container alive)
exec streamlit run ui/app_authenticated.py \
  --server.port=7860 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
