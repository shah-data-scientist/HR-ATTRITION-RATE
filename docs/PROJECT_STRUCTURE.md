# Project Structure

```
employee-attrition-prediction/
│
├── README.md                       # Project overview, quick start, API reference
│
├── api/                            # FastAPI backend
│   ├── app/
│   │   ├── main.py                 # All endpoints + prediction logic (~1200 lines)
│   │   └── schemas.py              # Unused legacy schemas (superseded by core/schema.py)
│   ├── auth.py                     # API key verification, bcrypt, key generation
│   ├── middleware.py               # Security headers middleware setup
│   ├── security.py                 # Authorization helpers
│   ├── get_model_features.py       # Extract expected columns from sklearn pipeline
│   ├── README.md                   # API-specific notes
│   └── tests/                      # Standalone tests (NOT auto-collected by pytest)
│
├── core/                           # Business logic — no FastAPI/Streamlit dependencies
│   ├── data_processing.py          # Merge 3 sources, clean, engineer features
│   ├── preprocess.py               # Schema enforcement, dtype coercion, range validation
│   ├── schema.py                   # Pydantic models (input/output schemas)
│   └── validation.py               # NUMERIC_COLS and CATEGORICAL_COLS definitions
│
├── database/                       # Database layer
│   ├── models.py                   # 7 SQLAlchemy ORM models
│   ├── database.py                 # Engine + session factory (PostgreSQL or SQLite)
│   ├── init_db.py                  # Create tables + seed from CSV
│   ├── seed_data.py                # Load data/ CSVs into the database
│   ├── schema.sql                  # Raw SQL schema reference
│   ├── add_shap_table.sql          # Migration: add shap_analysis table
│   └── migrate_add_user_id_to_jobs.sql  # Migration: add user_id to jobs
│
├── ui/                             # Streamlit frontend
│   ├── app_authenticated.py        # Entry point: login gate + session init
│   ├── app.py                      # Main dashboard (predictions, charts, downloads)
│   └── auth.py                     # Login/logout, role check, session management
│
├── models/                         # Trained ML model artifacts (committed to git)
│   ├── employee_attrition_pipeline.pkl  # sklearn Pipeline (preprocessor + model)
│   ├── X_train.parquet             # Training features (for SHAP LinearExplainer)
│   ├── X_test.parquet              # Test features
│   ├── y_train.parquet             # Training labels
│   └── y_test.parquet              # Test labels
│
├── data/                           # Source CSV data files
│   ├── extrait_eval.csv            # Evaluation scores, overtime, salary increase
│   ├── extrait_sirh.csv            # Demographics, tenure, monthly income
│   └── extrait_sondage.csv         # Survey: training, travel, education, distance
│
├── database_extracts/              # Database exports for reference
│   ├── Database Schema.png         # Visual schema diagram
│   ├── employees.csv
│   ├── jobs.csv
│   ├── model_inputs.csv
│   ├── model_outputs.csv
│   ├── predictions_traceability.csv
│   ├── shap_analysis.csv
│   └── users.csv
│
├── scripts/                        # Operational scripts
│   ├── worker.py                   # Background job processor (polls jobs table)
│   ├── utils.py                    # Shared data utilities (clean_*, load_and_merge)
│   ├── start-api.sh / start-api.bat       # Start API server
│   ├── start-ui.sh  / start-ui.bat        # Start Streamlit UI
│   ├── start-api-nodb.bat          # Start API with DISABLE_DB=true
│   ├── create_tables.py            # Standalone table creation
│   ├── migrate_db.py               # Run database migrations
│   ├── create_synthetic_data.py    # Generate synthetic test data
│   ├── generate_professional_presentation.py  # Slide deck generation
│   ├── local_ci.ps1                # Local CI check (PowerShell)
│   ├── dev/                        # Development / debug scripts (not for production)
│   │   ├── debug_api_call.py
│   │   ├── e2e_test.py
│   │   ├── enqueue_sample_report_job.py
│   │   └── ...
│   └── archive/                    # Old startup scripts
│
├── tests/                          # Automated test suite (205 tests)
│   ├── conftest.py                 # Fixtures: test client, mock DB, sample payloads
│   ├── test_core.py                # Data cleaning + feature engineering
│   ├── test_core_modules.py        # Additional core module tests
│   ├── test_database.py            # ORM models and schema
│   ├── test_database_integration.py  # DB integration via test client
│   ├── test_api_comprehensive.py   # All API endpoints
│   ├── test_api_coverage_boost.py  # Additional API coverage
│   ├── test_api_integration_advanced.py
│   ├── test_api_integration_complete.py
│   ├── test_api_debug.py
│   ├── test_model_loading.py       # Model file loads on startup
│   ├── test_preprocess_complete.py # Schema enforcement
│   ├── test_scripts_utils.py       # scripts/utils.py functions
│   ├── test_ui_auth.py             # Streamlit auth module
│   ├── test_ui_authenticated.py    # Authenticated UI flow
│   ├── test_ui_automation.py       # Playwright browser tests
│   ├── test_ui_functions.py        # UI helper functions
│   ├── test_automated_e2e.py       # End-to-end (requires live API)
│   ├── fixtures/                   # SQL and config test fixtures
│   ├── manual/                     # Manual test scripts (not auto-run)
│   └── archive/                    # Retired test files (not auto-run)
│
├── docker/
│   ├── Dockerfile.api              # FastAPI image (python:3.13-slim, multi-stage)
│   ├── Dockerfile.streamlit        # Streamlit image (python:3.13-slim, multi-stage)
│   └── Dockerfile.database         # DB initialisation container
│
├── docs/                           # Documentation
│   ├── README.md                   # Docs index
│   ├── QUICKSTART.md               # First-run guide
│   ├── DEVELOPMENT.md              # Local setup, testing, tooling
│   ├── DEPLOYMENT.md               # Docker Compose, production, CI/CD
│   ├── ARCHITECTURE.md             # System design, data flow, DB schema
│   ├── ER_DIAGRAM.md               # Entity-relationship diagram
│   ├── PROJECT_STRUCTURE.md        # This file
│   └── archive/                    # Historical documentation
│
├── .github/workflows/ci-cd.yml     # GitHub Actions CI/CD pipeline
├── .streamlit/config.toml          # Streamlit theme and server config
├── docker-compose.yml              # Orchestration: local + prod profiles
├── pyproject.toml                  # Poetry deps, pytest, mypy, ruff, black config
└── poetry.lock                     # Pinned dependency versions
```

---

## Key conventions

- **`core/`** is the only package imported by both `api/` and `ui/`. It has no web framework dependency.
- **`models/`** holds committed binary artifacts. `models/snapshots/` and `models/*.xlsx` are gitignored (large / generated).
- **`tests/manual/`** and **`tests/archive/`** are excluded from automatic pytest collection via `norecursedirs` in `pyproject.toml`.
- **`scripts/worker.py`** is invoked as a module (`python -m scripts.worker`) so Docker can call it without path manipulation.
- The `api/tests/` directory exists but is not collected by pytest (`testpaths = ["tests"]`); those tests are covered by the main `tests/` suite.
