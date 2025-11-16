@echo off
setlocal enabledelayedexpansion

set MAX_RETRIES=5
set RETRY_COUNT=0
set RETRY_DELAY=5

:START
echo.
echo ========================================
echo Starting Streamlit UI (Attempt !RETRY_COUNT! of %MAX_RETRIES%)
echo ========================================
echo.

cd /d "%~dp0"
set API_BASE_URL=http://localhost:8001

poetry run streamlit run ui/app.py --server.port 8501

set EXIT_CODE=!ERRORLEVEL!

if !EXIT_CODE! EQU 0 (
    echo UI exited normally.
    goto END
)

echo.
echo ========================================
echo UI crashed with exit code: !EXIT_CODE!
echo ========================================

set /a RETRY_COUNT+=1

if !RETRY_COUNT! GEQ %MAX_RETRIES% (
    echo.
    echo Maximum retry attempts reached. Exiting.
    goto END
)

echo Restarting in %RETRY_DELAY% seconds...
timeout /t %RETRY_DELAY% /nobreak >nul
goto START

:END
echo.
echo Press any key to close this window...
pause >nul
