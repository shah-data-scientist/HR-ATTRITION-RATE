# Project Context: HR Attrition Rate

## Overview
- **Goal**: Predict employee attrition using ML (FastAPI backend, Streamlit frontend).
- **Tech Stack**: Python 3.12, FastAPI, Streamlit, PostgreSQL, Docker, Poetry.
- **Infrastructure**: Docker Compose with profiles (`local`, `prod`, `huggingface`).

## Current Session Status
- **Date**: 2025-12-05
- **Active Task**: Commit changes.
- **Status**:
    - Added `types-requests` to fix Mypy error.
    - Amended commit "Github action ready" to include dependency changes.
    - All checks passed locally.
- **Next Steps**: Await user instructions (e.g., push to remote).

## Action Plan
1.  **Fix Dependencies**: Install `playwright` and `pytest-playwright` (Done).
2.  **Setup Environment**: Install playwright browsers (`chromium`) (Done).
3.  **Troubleshoot Docker**: User needs to start Docker Desktop.
4.  **Docker Setup**: Retry `docker-compose --profile local up -d --build`.
5.  **Testing**: Run `scripts/dev/e2e_test.py` and `tests/run_ui_test.py`.
