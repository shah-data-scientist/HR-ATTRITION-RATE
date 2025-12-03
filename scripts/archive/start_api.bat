@echo off
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
uvicorn api.app.main:app --host 0.0.0.0 --port 8001 --reload