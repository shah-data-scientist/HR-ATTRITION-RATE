@echo off
REM Start the FastAPI backend server with DB disabled (no persistence)

echo Starting FastAPI backend (no DB) on http://localhost:8001...
poetry run uvicorn api.app.main:app --host 0.0.0.0 --port 8001 --reload --env-file .env.nodb
