@echo off
set DATABASE_URL=postgresql://user:password@127.0.0.1:5432/hr_attrition_db
set DISABLE_DB=0
poetry run uvicorn api.app.main:app --host 0.0.0.0 --port 8001 --reload
pause
