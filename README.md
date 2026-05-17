# HR Attrition Rate — Employee Turnover Prediction

A production-ready machine learning system that predicts employee attrition risk. Built with FastAPI, Streamlit, PostgreSQL, and scikit-learn, it combines batch predictions, SHAP explainability, and full audit traceability.

---

## Architecture

```
┌──────────────────────┐
│   Streamlit UI        │  Port 8501 (prod) / 8581 (dev)
│   ui/app_authenticated│
└──────────┬───────────┘
           │ HTTP  X-API-Key
           ▼
┌──────────────────────┐
│   FastAPI Backend     │  Port 8001 (prod) / 8081 (dev)
│   api/app/main.py     │
└──────┬───────┬────────┘
       │       │
       ▼       ▼
  PostgreSQL  ML Model
  (6 tables)  models/employee_attrition_pipeline.pkl
```

**Data flow:** Three CSV sources (evaluation, SIRH, survey) → merge & clean (`core/`) → sklearn pipeline → SHAP explanation → PostgreSQL traceability → Excel/chart output.

---

## Quick Start

### Prerequisites

- Python 3.11+, Poetry
- Docker & Docker Compose (for the recommended path)

### Run with Docker (recommended)

```bash
git clone https://github.com/shah-data-scientist/HR-ATTRITION-RATE.git
cd HR-ATTRITION-RATE

cp .env.example .env
# Edit .env: set API_KEY and SECRET_KEY to strong random strings

docker compose --profile local up -d
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8581 |
| FastAPI docs | http://localhost:8081/docs |
| API health | http://localhost:8081/health |

### Run locally without Docker

```bash
poetry install
# Terminal 1
./scripts/start-api.sh        # Linux/Mac
scripts\start-api.bat         # Windows

# Terminal 2
./scripts/start-ui.sh
scripts\start-ui.bat
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for full local setup.

---

## Features

| Feature | Detail |
|---------|--------|
| Batch predictions | Predict attrition for N employees in one request |
| SHAP explanations | Per-employee waterfall charts and HTML force plots |
| Risk categories | Low / Medium / High (thresholds: 0–0.3 / 0.3–0.7 / 0.7–1.0) |
| Excel report | Downloadable report with predictions and SHAP values |
| Async jobs | Background report generation via job queue |
| Full traceability | Every prediction stored with inputs, outputs, SHAP in PostgreSQL |
| UI authentication | Username/password login (bcrypt), role-based (admin/user) |
| API authentication | `X-API-Key` header required on all prediction endpoints |
| Security headers | XSS, CSRF, CSP, HSTS, CORS, GZip |

---

## API Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/` | Root info | None |
| GET | `/health` | API health check | None |
| GET | `/db_health` | Database connectivity | None |
| POST | `/predict` | Batch prediction (JSON) | API key |
| POST | `/predict_report` | Prediction + Excel report | API key |
| POST | `/predict_excel` | Upload Excel → predictions | API key |
| POST | `/predict_shap_images` | SHAP waterfall charts (ZIP) | API key |
| POST | `/predict_shap_html` | SHAP force plot (HTML) | API key |
| POST | `/jobs/report` | Create async report job | API key |
| GET | `/jobs/{job_id}` | Job status | API key |
| GET | `/jobs/{job_id}/result` | Job result | API key |
| POST | `/auth/login` | Authenticate user | None |
| GET | `/auth/user/{username}` | Get user info | API key |

Full interactive documentation: http://localhost:8001/docs

---

## Input Data Format

The API accepts two input modes:

**Mode 1 — Raw 3-source merge** (`POST /predict` with `RawBatchPredictionInput`):
```json
{
  "eval_data": [{ "eval_number": "E_123", "note_evaluation_actuelle": 3, ... }],
  "sirh_data": [{ "id_employee": 123, "age": 35, "genre": "Homme", ... }],
  "sondage_data": [{ "code_sondage": 123, "niveau_education": 3, ... }]
}
```

**Mode 2 — Pre-merged** (`POST /predict` with `ProcessedBatchPredictionInput`):
```json
{
  "employees": [{ "id_employee": 123, "age": 35, "note_evaluation_actuelle": 3, ... }]
}
```

See the API schema at `/docs` for full field definitions and validation rules.

---

## Project Structure

```
HR-ATTRITION-RATE/
├── api/                        # FastAPI backend
│   ├── app/
│   │   ├── main.py             # All endpoints, model loading, prediction logic
│   │   └── schemas.py          # Pydantic request/response models
│   ├── auth.py                 # API key verification, bcrypt, key generation
│   ├── middleware.py           # Security headers (XSS, CSRF, CSP, HSTS)
│   └── security.py             # Authorization helpers
├── core/                       # Business logic (API/UI-independent)
│   ├── data_processing.py      # Data cleaning and feature engineering
│   ├── preprocess.py           # Schema enforcement and range validation
│   ├── schema.py               # Pydantic input/output schemas
│   └── validation.py           # Feature definitions (numeric/categorical cols)
├── database/                   # Database layer
│   ├── models.py               # 6 SQLAlchemy ORM models
│   ├── database.py             # Connection management (PostgreSQL / SQLite)
│   ├── init_db.py              # Schema creation + data seeding
│   └── seed_data.py            # CSV → database seeding
├── ui/                         # Streamlit frontend
│   ├── app_authenticated.py    # Entry point (login gate + session state)
│   ├── app.py                  # Main dashboard (predictions, charts, downloads)
│   └── auth.py                 # Login/logout, role checking
├── models/                     # Trained ML model artifacts
│   ├── employee_attrition_pipeline.pkl   # sklearn Pipeline (preprocessor + model)
│   ├── X_train.parquet         # Training features (for SHAP explainer)
│   ├── X_test.parquet          # Test features
│   ├── y_train.parquet         # Training labels
│   └── y_test.parquet          # Test labels
├── data/                       # Source CSV files
│   ├── extrait_eval.csv        # Employee evaluations
│   ├── extrait_sirh.csv        # HR system data
│   └── extrait_sondage.csv     # Employee survey responses
├── scripts/                    # Operational scripts
│   ├── worker.py               # Background job processor
│   ├── utils.py                # Shared data utilities
│   ├── start-api.sh/bat        # API startup
│   ├── start-ui.sh/bat         # UI startup
│   └── dev/                    # Development and debug scripts
├── tests/                      # Test suite (205 tests)
│   ├── conftest.py             # Fixtures and test configuration
│   ├── test_core.py            # Core business logic tests
│   ├── test_database.py        # ORM and schema tests
│   ├── test_api_comprehensive.py  # API endpoint tests
│   ├── test_api_integration_*.py  # Integration tests
│   ├── test_ui_*.py            # UI authentication and function tests
│   ├── manual/                 # Manual test scripts (not auto-run)
│   └── archive/                # Archived test files
├── archive/
│   ├── Testing Data Application UI/  # Sample employee records usable as UI test data
│   ├── database_extracts/            # Database schema diagram and CSV exports
│   └── workbook/                     # Analysis notebooks
├── docker/
│   ├── Dockerfile.api          # FastAPI container (Python 3.13-slim, multi-stage)
│   ├── Dockerfile.streamlit    # Streamlit container (Python 3.13-slim, multi-stage)
│   └── Dockerfile.database     # Database init container
├── docs/                       # All project documentation
├── docker-compose.yml          # Orchestration (local + prod profiles)
├── pyproject.toml              # Dependencies and tool config (Poetry)
└── README.md                   # This file
```

---

## Configuration

Copy `.env.example` to `.env` and set:

```bash
# Required — generate with: python -c "import secrets; print(secrets.token_hex(32))"
API_KEY=your_64_char_hex_key
SECRET_KEY=your_32_char_min_secret

# Database (PostgreSQL)
POSTGRES_USER=user
POSTGRES_PASSWORD=strong_password
POSTGRES_DB=hr_attrition_db
DATABASE_URL=postgresql://user:password@localhost:5432/hr_attrition_db

# API server
API_HOST=0.0.0.0
API_PORT=8001

# UI → API connection
API_BASE_URL=http://localhost:8001
```

**Never commit `.env` to git.** Use `.env.example` as the template only.

---

## Testing

```bash
# Full test suite
poetry run pytest

# With coverage
poetry run pytest --cov=api.app.main --cov=core --cov=database.models --cov-report=term

# Specific module
poetry run pytest tests/test_core.py -v
```

Current coverage: ~74% (core modules 85–98%, API 52%)

---

## HuggingFace Spaces Deployment

The live demo runs at **[huggingface.co/spaces/shah-data-scientist/HRApp](https://huggingface.co/spaces/shah-data-scientist/HRApp)**.

### Architecture difference

| This repo (source) | HuggingFace Space |
|--------------------|-------------------|
| 3 containers: API + UI + Database (Docker Compose) | 1 container (HF constraint) |
| PostgreSQL | SQLite (persisted at `/app/db_data/hr_attrition.db`) |
| `pyproject.toml` / Poetry | `requirements.txt` (pinned flat deps) |

### Repository relationship

The HuggingFace Space is maintained in a separate repo at `../hr-attrition-hf-space/` (sibling directory). **All application source files originate here** — the HF space repo contains only the deployment-specific wrapper (`Dockerfile`, `requirements.txt`, `README.md` with YAML front-matter, `.gitattributes`).

### HF deployment files (live in this repo)

The HF-specific deployment files are kept in [`docker/`](docker/) alongside the other Dockerfiles:

| File in this repo | Synced to HF space as | Purpose |
|-------------------|-----------------------|---------|
| `docker/Dockerfile.huggingface` | `Dockerfile` | Single-container build (API + UI + SQLite) |
| `docker/requirements.huggingface.txt` | `requirements.txt` | Pinned pip dependencies |
| `docker/start.huggingface.sh` | `start.sh` | Startup script (runs both services) |

### How to sync changes to HuggingFace

When you update application code in this repo, copy the changed files to the HF space and push:

```bash
HF="../hr-attrition-hf-space/hr-attrition-hf-space"

# Sync source files
cp -r api/          $HF/api/
cp -r core/         $HF/core/
cp -r database/     $HF/database/
cp -r ui/           $HF/ui/
cp -r data/         $HF/data/
cp -r models/       $HF/models/
cp scripts/utils.py $HF/scripts/utils.py
cp .streamlit/config.toml $HF/.streamlit/config.toml

# Sync deployment files
cp docker/Dockerfile.huggingface      $HF/Dockerfile
cp docker/requirements.huggingface.txt $HF/requirements.txt
cp docker/start.huggingface.sh         $HF/start.sh

# Commit and push to HuggingFace
cd $HF
git add -A
git commit -m "Sync from HR-ATTRITION-RATE $(date +%Y-%m-%d)"
git push origin main   # pushes to HuggingFace Spaces
git push github main   # mirrors to GitHub
```

> **Never edit files directly in the HF space repo.** Make all changes here, then sync.

### HF space demo credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | `admin` | `Admin@2025!Secure` |
| Analyst | `analyst` | `Analyst@2025!View` |

---

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci-cd.yml`):

1. **Code quality** — Black formatting, Mypy type checking, Ruff linting
2. **Security scan** — Trivy vulnerability scanning → GitHub Security dashboard
3. **Tests** — pytest with live PostgreSQL, coverage to Codecov
4. **Auth tests** — Standalone bcrypt and API key generation tests
5. **Docker builds** — API and UI images pushed to GitHub Container Registry (`ghcr.io`)

---

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Step-by-step first-run guide |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Local dev setup, testing, tooling |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker Compose, production config |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and data flow |
| [docs/ER_DIAGRAM.md](docs/ER_DIAGRAM.md) | Database schema |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Directory layout reference |

---

## Troubleshooting

**API returns 401/403**
- Check `API_KEY` is set in `.env` and matches what the UI sends
- Restart both API and UI after changing `.env`

**"Model file not found" on startup**
- Ensure `models/employee_attrition_pipeline.pkl` exists (it should be committed)
- Run `git ls-files models/` to confirm

**UI can't connect to API**
- Check API is running: `curl http://localhost:8001/health`
- For Docker: use service name `http://fastapi_app:8001`, not `localhost`

**Database connection refused**
- Start PostgreSQL: `docker compose --profile local up db -d`
- Check `DATABASE_URL` in `.env`
- Init schema: `poetry run python database/init_db.py`
