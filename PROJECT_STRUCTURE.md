# Project Structure

This document describes the organization of the HR Attrition Rate project.

## Directory Structure

```
hr-attrition-rate/
│
├── .github/                    # GitHub Actions workflows
│   └── workflows/
│       └── ci-cd.yml          # CI/CD pipeline (tests, docker builds)
│
├── .streamlit/                 # Streamlit configuration
│   └── config.toml            # UI server settings
│
├── api/                        # FastAPI backend application
│   ├── app/                   # Main application code
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI app entrypoint
│   │   └── schemas.py        # Pydantic models for requests/responses
│   ├── tests/                 # API-specific tests
│   ├── security.py            # Authentication and authorization
│   ├── get_model_features.py  # Model feature extraction utilities
│   └── README.md              # API documentation
│
├── core/                       # Core business logic (model-agnostic)
│   ├── __init__.py
│   ├── data_processing.py     # Feature engineering and transformation
│   ├── preprocess.py          # Data preprocessing pipeline
│   ├── schema.py              # Data schema definitions
│   └── validation.py          # Input validation logic
│
├── database/                   # Database layer
│   ├── __init__.py
│   ├── models.py              # SQLAlchemy ORM models
│   ├── database.py            # Database connection and session management
│   ├── init_db.py             # Database initialization script
│   ├── seed_data.py           # Sample data seeding
│   └── schema.sql             # Raw SQL schema (reference)
│
├── ui/                         # Streamlit frontend
│   └── app.py                 # Main UI application
│
├── data/                       # Sample/seed data files
│   ├── extrait_eval.csv       # Employee evaluation data
│   ├── extrait_sirh.csv       # HR system data
│   └── extrait_sondage.csv    # Employee survey data
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md        # System architecture and design
│   ├── REFACTOR_SUMMARY.md    # Development history and changes
│   ├── deployment/            # Deployment-specific docs
│   │   └── HUGGINGFACE_DEPLOYMENT.md  # Unified HF Space deployment guide
│   └── archive/               # Historical/archived documentation
│
├── outputs/                    # Model artifacts and predictions
│   ├── employee_attrition_pipeline.pkl  # Trained ML model
│   └── snapshots/             # Model version snapshots
│
├── scripts/                    # Utility scripts
│   ├── utils.py               # Shared utility functions
│   ├── create_synthetic_data.py
│   ├── start-api.sh           # Development server scripts
│   ├── start-api.bat
│   ├── start-ui.sh
│   ├── start-ui.bat
│   └── README.md
│
├── tests/                      # Test suite
│   ├── conftest.py            # Pytest configuration and fixtures
│   ├── test_api_comprehensive.py
│   ├── test_core.py
│   ├── test_database.py
│   ├── test_coverage_85.py
│   └── TEST_README.md
│
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
│
├── docker/                     # Docker configuration files
│   ├── Dockerfile.api         # API container definition
│   ├── Dockerfile.streamlit   # UI container definition
│   ├── Dockerfile.database    # DB initialization container
│   └── Dockerfile.huggingface # Unified HF deployment (API+UI+SQLite+Supervisor)
├── docker-compose.yml          # Docker orchestration (dev/prod)
│
├── pyproject.toml              # Poetry dependencies and project config
├── poetry.lock                 # Locked dependency versions
│
├── README.md                   # Main project documentation
├── QUICKSTART.md               # Quick setup guide
├── DEVELOPMENT.md              # Development guidelines
├── DEPLOYMENT.md               # Production deployment guide
└── PROJECT_STRUCTURE.md        # This file

```

## Key Principles

### 1. **Separation of Concerns**
- **api/**: REST API layer - handles HTTP requests/responses
- **core/**: Business logic - reusable across different interfaces
- **database/**: Data persistence layer - ORM models and queries
- **ui/**: User interface - Streamlit frontend

### 2. **No Nested src/ Directory**
The project uses a flat structure with top-level packages (`api`, `core`, `database`, `ui`) instead of nesting them under `src/`. This is configured in `pyproject.toml` with `package-mode = false`.

### 3. **Environment-Specific Configurations**
- `.env.example`: Template for environment variables
- `docker-compose.yml`: Orchestrates services for development/production
- Multiple Dockerfiles for different deployment scenarios

### 4. **Documentation Organization**
- Root-level: User-facing docs (README, QUICKSTART, guides)
- `docs/`: Technical documentation and architecture
- `docs/deployment/`: Deployment-specific documentation
- `docs/archive/`: Historical/deprecated documentation

### 5. **Testing Strategy**
- `tests/`: Integration and end-to-end tests
- `api/tests/`: API-specific unit tests
- Coverage target: 76% (goal: 85%)

## Import Conventions

### From Scripts and Tests
```python
from api.app.main import app
from core.data_processing import clean_data
from database.models import Employee
```

### Within Modules (Relative Imports)
```python
# Within api/app/
from .schemas import PredictionRequest
from ..security import verify_token

# Within core/
from .validation import validate_input
```

## Data Flow

```
User → Streamlit UI (ui/app.py)
  ↓
  HTTP Request → FastAPI (api/app/main.py)
  ↓
  Validation → core/validation.py
  ↓
  Data Processing → core/data_processing.py
  ↓
  Model Prediction → outputs/employee_attrition_pipeline.pkl
  ↓
  Database Logging → database/models.py
  ↓
  Response → User
```

## Deployment Scenarios

### Local Development
- Use `scripts/start-api.sh` and `scripts/start-ui.sh`
- PostgreSQL via `docker-compose up db -d`
- Hot-reload enabled

### Docker Compose (Recommended)
- All services: `docker-compose up`
- Includes: PostgreSQL, API, UI, DB initialization

### Hugging Face Spaces
- **Unified**: `Dockerfile.huggingface` (API + UI + SQLite + Supervisor)
- **Separate**: `Dockerfile.huggingface.api` + `Dockerfile.huggingface.ui`

### Cloud Production
- See `DEPLOYMENT.md` for AWS, Azure, Kubernetes deployments

## Adding New Features

### New API Endpoint
1. Add route in `api/app/main.py`
2. Define schemas in `api/app/schemas.py`
3. Add business logic in `core/`
4. Add tests in `api/tests/`

### New Data Processing
1. Add function in `core/data_processing.py`
2. Add tests in `tests/test_core.py`
3. Update schema if needed in `core/schema.py`

### Database Changes
1. Update models in `database/models.py`
2. Update `database/schema.sql` (optional)
3. Add migration logic if using Alembic
4. Update `database/init_db.py` for initialization

## Configuration Files

- **pyproject.toml**: Python dependencies, project metadata
- **poetry.lock**: Locked dependency versions (committed)
- **.env**: Environment variables (NOT committed - in .gitignore)
- **.env.example**: Template for required environment variables
- **docker-compose.yml**: Service orchestration
- **.github/workflows/ci-cd.yml**: CI/CD pipeline

## Best Practices

1. **Never commit** `.env`, `*.db`, `__pycache__/`, or temporary test files
2. **Always run tests** before committing: `poetry run pytest`
3. **Use type hints** for better code quality (checked by mypy in CI)
4. **Format code** with black: `poetry run black .`
5. **Update documentation** when changing project structure
6. **Use Poetry** for dependency management: `poetry add <package>`
7. **Follow semantic versioning** for releases

## Maintenance

### Cleaning Up
```bash
# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove test artifacts
rm -rf htmlcov/ .coverage .pytest_cache/

# Remove temporary files
rm -f *.db temp_*.json test_*.json
```

### Updating Dependencies
```bash
# Update all dependencies
poetry update

# Update specific package
poetry update <package-name>

# Add new dependency
poetry add <package-name>

# Add dev dependency
poetry add --group dev <package-name>
```

## Contact & Support

For questions about project structure or contributing:
1. Read [DEVELOPMENT.md](DEVELOPMENT.md)
2. Check [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
3. Open an issue on GitHub
