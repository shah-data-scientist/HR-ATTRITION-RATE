# Development Guide

## Local Setup (without Docker)

### Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- PostgreSQL 16+ (or use SQLite by setting `DISABLE_DB=true`)

### Install dependencies

```bash
poetry install --with dev
```

### Configure environment

```bash
cp .env.example .env
# Edit .env with your local database and API key settings
```

For local development without PostgreSQL, add `DISABLE_DB=true` to `.env`.

### Start the API

```bash
# Linux/Mac
./scripts/start-api.sh

# Windows
scripts\start-api.bat

# Or directly with uvicorn
poetry run uvicorn api.app.main:app --reload --port 8001
```

### Start the UI

```bash
# Linux/Mac
./scripts/start-ui.sh

# Windows
scripts\start-ui.bat

# Or directly with streamlit
poetry run streamlit run ui/app_authenticated.py --server.port 8501
```

### Start the background worker (optional)

The worker processes async report jobs. Only needed if you use `/jobs/report`.

```bash
poetry run python -m scripts.worker
```

---

## Testing

```bash
# All tests
poetry run pytest

# With coverage report
poetry run pytest --cov=api.app.main --cov=core --cov=database.models \
  --cov-report=term --cov-report=html

# Single module
poetry run pytest tests/test_core.py -v

# Skip E2E tests (require live API server)
poetry run pytest --ignore=tests/test_automated_e2e.py \
  --ignore=tests/test_ui_automation.py
```

Test configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`.

**Test layout:**

| Directory / file | What it tests |
|------------------|--------------|
| `tests/test_core.py` | Data cleaning, feature engineering |
| `tests/test_database.py` | SQLAlchemy ORM, schema integrity |
| `tests/test_api_comprehensive.py` | All API endpoints (mocked model) |
| `tests/test_api_integration_*.py` | Integration with live DB |
| `tests/test_ui_*.py` | UI auth, functions, Playwright automation |
| `tests/manual/` | Manual scripts, not auto-run |
| `tests/archive/` | Retired coverage-boost files, not auto-run |

---

## Code Quality

```bash
# Formatting (Black)
poetry run black .

# Type checking (Mypy)
poetry run mypy --ignore-missing-imports api/ core/ database/ ui/

# Linting (Ruff)
poetry run ruff check .

# All checks at once (mirrors CI)
poetry run black --check . && \
poetry run mypy --ignore-missing-imports api/ core/ database/ ui/ && \
poetry run ruff check .
```

Configuration for all tools is in `pyproject.toml`.

---

## Database

### Initialize schema (first time)

```bash
DATABASE_URL=postgresql://user:password@localhost:5432/hr_attrition_db \
  poetry run python database/init_db.py
```

### Migrate (when schema changes)

```bash
poetry run python scripts/migrate_db.py
```

### Seed with test data

```bash
poetry run python database/seed_data.py
```

### SQLite fallback (no PostgreSQL)

Set `DISABLE_DB=true` in `.env`. The API runs without database logging; all predictions still work but traceability is disabled.

---

## Adding a new endpoint

1. Add the route in `api/app/main.py`
2. Add request/response Pydantic models in `api/app/schemas.py` or `core/schema.py`
3. Add tests in `tests/test_api_comprehensive.py`
4. Run `poetry run black . && poetry run mypy --ignore-missing-imports api/`

---

## Environment variables reference

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `API_KEY` | Yes | — | Bearer key for prediction endpoints |
| `SECRET_KEY` | Yes | — | App secret (min 32 chars) |
| `DATABASE_URL` | No | SQLite fallback | PostgreSQL connection string |
| `POSTGRES_USER` | No | `user` | PostgreSQL user |
| `POSTGRES_PASSWORD` | No | `password` | PostgreSQL password |
| `POSTGRES_DB` | No | `hr_attrition_db` | PostgreSQL database name |
| `API_HOST` | No | `0.0.0.0` | API bind host |
| `API_PORT` | No | `8001` | API bind port |
| `API_BASE_URL` | No | `http://localhost:8001` | UI → API URL |
| `DISABLE_DB` | No | `false` | Skip DB; predictions still work |
| `WORKER_POLL_SEC` | No | `2` | Job polling interval (worker) |
| `WORKER_STALE_SEC` | No | `600` | Job stale timeout (worker) |
