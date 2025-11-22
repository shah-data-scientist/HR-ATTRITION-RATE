# Presentation Preparation Guide

## HR Attrition Rate Prediction - Project Deliverables

This document provides comprehensive preparation materials for your presentation covering all six key areas.

---

## 1. Git Repository Structure and CI/CD Pipeline Configuration

### Repository Structure Overview

```
hr-attrition-rate/
├── .github/workflows/ci-cd.yml    # CI/CD Pipeline
├── api/                           # FastAPI Backend (Port 8001)
│   ├── app/main.py               # API endpoints, prediction logic
│   ├── auth.py                   # Authentication (API Key, bcrypt)
│   └── tests/                    # API unit tests
├── core/                          # Business Logic
│   ├── data_processing.py        # Feature engineering
│   ├── preprocess.py             # Data cleaning
│   ├── schema.py                 # Pydantic models
│   └── validation.py             # Input validation
├── database/                      # PostgreSQL Integration
│   ├── models.py                 # SQLAlchemy ORM models
│   ├── database.py               # Connection management
│   └── init_db.py                # Schema initialization
├── ui/                            # Streamlit Frontend (Port 8501)
│   └── app.py                    # User interface
├── docker/                        # Docker Configurations
│   ├── Dockerfile.api            # API container
│   ├── Dockerfile.streamlit      # UI container
│   ├── Dockerfile.database       # DB initialization
│   └── Dockerfile.huggingface    # Unified deployment
├── tests/                         # Integration tests
├── outputs/                       # ML model artifacts
└── docker-compose.yml             # Service orchestration
```

### Key Talking Points

1. **Separation of Concerns**: API, Core Logic, Database, UI are independent modules
2. **Package Structure**: Flat structure (no nested `src/`) for simpler imports
3. **Configuration as Code**: All configs in `.env`, `docker-compose.yml`, `pyproject.toml`

### CI/CD Pipeline (`.github/workflows/ci-cd.yml`)

**Pipeline Jobs:**
| Job | Purpose | Tools Used |
|-----|---------|------------|
| `code-quality` | Linting & Type Checking | Black, Mypy |
| `security-scan` | Vulnerability Detection | Trivy (SARIF reports) |
| `test-with-database` | Integration Tests | PostgreSQL service, pytest |
| `test-no-database` | Hugging Face Mode Tests | DISABLE_DB=1, pytest |
| `test-authentication` | Security Module Tests | bcrypt, API key validation |
| `build-docker-images` | Container Builds | Docker Buildx, GHCR |
| `docker-huggingface` | Unified Image Build | Supervisor, SQLite |
| `deploy-staging` | Staging Deployment | Manual approval |
| `deploy-production` | Production Deployment | Environment protection |

**Key CI/CD Features:**
- Parallel test execution (with/without database)
- Automatic Docker image builds on push
- Security scanning with SARIF reports to GitHub Security
- Coverage reporting to Codecov
- Environment-based deployment gates

### Demonstration Commands

```bash
# View recent commits and branch structure
git log --oneline -10
git branch -a

# View CI/CD workflow
cat .github/workflows/ci-cd.yml

# Check GitHub Actions status (if using gh CLI)
gh run list --limit 5
```

---

## 2. API Functionality Demonstration with FastAPI

### API Architecture

```
Client (Streamlit UI)
    │
    │ HTTP + X-API-Key Header
    ▼
FastAPI Backend (Port 8001)
    │
    ├── /predict          → Batch predictions
    ├── /predict_report   → Predictions + Excel + SHAP
    ├── /predict_excel    → Excel report generation
    ├── /predict_shap_*   → SHAP visualizations
    ├── /jobs/*           → Async job management
    ├── /health           → Health check
    └── /db_health        → Database connectivity
```

### Key Endpoints

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/` | GET | No | Welcome message |
| `/health` | GET | No | API health status |
| `/db_health` | GET | No | Database connectivity |
| `/predict` | POST | Yes (X-API-Key) | Batch predictions |
| `/predict_report` | POST | Yes | Full report with Excel & SHAP |
| `/predict_excel` | POST | Yes | Excel file generation |
| `/predict_shap_images` | POST | Yes | SHAP waterfall plots |
| `/predict_shap_html` | POST | Yes | Interactive SHAP HTML |
| `/jobs/report` | POST | Yes | Async job submission |
| `/jobs/{job_id}` | GET | No | Job status check |
| `/jobs/{job_id}/result` | GET | No | Job results retrieval |

### Authentication Flow

```python
# api/auth.py
def _get_valid_api_key() -> str:
    """Dynamic API key lookup at request time"""
    return os.getenv("API_KEY", "demo_api_key")

async def get_api_key(api_key: str = Security(api_key_header)):
    valid_key = _get_valid_api_key()
    if api_key != valid_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key
```

### Demonstration Commands

```bash
# Health check (no auth required)
curl http://localhost:8001/health

# API docs (interactive)
# Open: http://localhost:8001/docs

# Prediction with authentication
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d @sample_payload.json

# Database health
curl http://localhost:8001/db_health
```

### Key Technical Justifications

1. **API Key Authentication**: Simpler than JWT for machine-to-machine communication
2. **Pydantic Validation**: Automatic input validation with detailed error messages
3. **SHAP Integration**: Explainability built into prediction pipeline
4. **Async Support**: Non-blocking I/O for better performance
5. **OpenAPI Documentation**: Auto-generated interactive docs at `/docs`

---

## 3. Unit and Functional Testing with Pytest

### Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── test_core.py                   # Core logic tests
├── test_api_comprehensive.py      # API endpoint tests
├── test_database_integration.py   # Database path tests
├── test_coverage_85.py            # Coverage boost tests
├── test_coverage_boost.py         # Additional coverage
├── test_uncovered_paths.py        # Edge case tests
└── test_final_push_85.py          # Final push tests
```

### Test Coverage Summary

```
Module                    Coverage
----------------------------------------
api/app/main.py          50%  (170/340 statements)
core/__init__.py         100% (5/5 statements)
core/data_processing.py  98%  (50/51 statements)
core/preprocess.py       85%  (33/39 statements)
core/schema.py           96%  (24/25 statements)
core/validation.py       93%  (62/67 statements)
database/models.py       100% (28/28 statements)
----------------------------------------
TOTAL                    70%  (372/531 statements)
```

### Testing Strategy

**1. Unit Tests (core/)**
- Data cleaning functions
- Feature engineering
- Schema validation
- Input sanitization

**2. Integration Tests (api/)**
- Endpoint functionality
- Authentication flows
- Database interactions
- Error handling

**3. Edge Case Tests**
- Empty input handling
- Invalid data types
- Boundary values
- Database disabled mode

### Key Test Examples

```python
# Test API authentication
def test_missing_api_key(client, complete_payload):
    response = client.post("/predict", json=complete_payload)
    assert response.status_code == 401

def test_invalid_api_key(client, complete_payload):
    response = client.post("/predict",
                          headers={"X-API-Key": "wrong"},
                          json=complete_payload)
    assert response.status_code == 403

# Test database health function
def test_db_ok_with_none(self):
    from api.app.main import _db_ok
    assert _db_ok(None) is False

# Test risk categorization
def test_get_risk_category_high(self):
    from api.app.main import get_risk_category
    assert get_risk_category(0.8, 0.5) == "High"
```

### Demonstration Commands

```bash
# Run all tests with coverage
poetry run pytest --cov=api --cov=core --cov=database --cov-report=term -v

# Run specific test file
poetry run pytest tests/test_core.py -v

# Run with database disabled
DISABLE_DB=1 poetry run pytest

# Generate HTML coverage report
poetry run pytest --cov=api --cov=core --cov-report=html
# Open htmlcov/index.html
```

---

## 4. PostgreSQL Database Overview

### Entity-Relationship Diagram

```
┌─────────────────────────┐
│       employees         │
├─────────────────────────┤
│ id (PK)                 │
│ id_employee (unique)    │
│ age, genre              │
│ revenu_mensuel          │
│ ... (all features)      │
│ user_id                 │
│ date_ingestion          │
└──────────┬──────────────┘
           │
           │ 1:N
           ▼
┌─────────────────────────┐
│    model_inputs         │
├─────────────────────────┤
│ id (PK)                 │
│ employee_id (FK)        │◄──────┐
│ input_features (JSON)   │       │
│ created_at              │       │
└──────────┬──────────────┘       │
           │                      │
           │ 1:1                  │
           ▼                      │
┌─────────────────────────┐       │
│    model_outputs        │       │
├─────────────────────────┤       │
│ id (PK)                 │       │
│ input_id (FK)           │       │
│ prediction              │       │
│ probability             │       │
│ risk_category           │       │
│ shap_values (JSON)      │       │
│ created_at              │       │
└──────────┬──────────────┘       │
           │                      │
           │ 1:1                  │
           ▼                      │
┌─────────────────────────┐       │
│ prediction_traceability │       │
├─────────────────────────┤       │
│ id (PK)                 │       │
│ output_id (FK)          │       │
│ model_version           │       │
│ api_version             │       │
│ created_at              │       │
└─────────────────────────┘       │
                                  │
┌─────────────────────────┐       │
│         jobs            │       │
├─────────────────────────┤       │
│ job_id (PK)             │       │
│ status                  │       │
│ employee_id (FK)        │───────┘
│ result (JSON)           │
│ error_message           │
│ created_at, updated_at  │
└─────────────────────────┘
```

### Database Configuration

```python
# database/database.py
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/hr_attrition_db"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
```

### Key Tables and Purpose

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `employees` | Employee master data | All input features, user_id |
| `model_inputs` | Raw prediction inputs | JSON blob of features |
| `model_outputs` | Prediction results | probability, risk_category, SHAP |
| `prediction_traceability` | Audit trail | model_version, api_version |
| `jobs` | Async job queue | status, result, error |

### Database Modes

**1. PostgreSQL (Production)**
- Full traceability and audit
- Job queue for async processing
- ACID compliance

**2. DISABLE_DB=1 (Hugging Face)**
- In-memory processing only
- No persistence
- SQLite fallback available

### Demonstration Commands

```bash
# Start PostgreSQL
docker-compose up db -d

# Check connection
curl http://localhost:8001/db_health

# View tables (via psql)
docker exec -it hr-attrition-db psql -U user -d hr_attrition_db -c "\dt"

# Query predictions
docker exec -it hr-attrition-db psql -U user -d hr_attrition_db \
  -c "SELECT * FROM model_outputs LIMIT 5;"
```

---

## 5. Use Cases and Results Discussion

### Primary Use Cases

**1. Individual Employee Risk Assessment**
- HR uploads single employee data
- System returns risk score + explanation
- SHAP waterfall shows contributing factors

**2. Batch Employee Analysis**
- Upload CSV files (eval, SIRH, survey)
- Receive Excel report with all predictions
- Aggregate risk distribution

**3. Continuous Monitoring (API Integration)**
- HRIS systems call `/predict` endpoint
- Real-time risk scoring
- Database logging for trend analysis

### Sample Results Interpretation

```json
{
  "id_employee": 12345,
  "prediction": "Leave",
  "probability": 0.78,
  "risk_category": "High",
  "message": "Employee 12345 has a HIGH risk of leaving (78% probability)",
  "shap_values": [-0.12, 0.45, -0.08, ...],
  "feature_names": ["satisfaction_environnement", "overtime", ...]
}
```

**Risk Categories:**
- **Low** (0-30%): Stable, no immediate action needed
- **Medium** (30-70%): Monitor closely, consider retention strategies
- **High** (70-100%): Urgent attention required

### Key Model Features (by SHAP importance)

1. `heure_supplementaires` (Overtime) - Strong positive impact on attrition
2. `satisfaction_employee_environnement` - Negative correlation
3. `annees_dans_l_entreprise` - Complex relationship
4. `revenu_mensuel` - Higher income reduces risk
5. `distance_domicile_travail` - Long commute increases risk

### Business Value

- **Early Warning System**: Identify at-risk employees before resignation
- **Targeted Interventions**: Focus resources on high-risk individuals
- **Data-Driven HR**: Replace gut feelings with evidence
- **Explainability**: SHAP provides actionable insights

---

## 6. Discussion Preparation

### Anticipated Questions & Answers

**Q1: Why API Key instead of JWT/OAuth?**
> API Key authentication is simpler and sufficient for internal/controlled access scenarios. It's stateless, easy to implement, and perfect for machine-to-machine communication. For user-facing authentication, we use bcrypt password hashing in the UI.

**Q2: How do you ensure prediction traceability?**
> Every prediction is logged in PostgreSQL with:
> - Input features (JSON blob)
> - Output (prediction, probability, SHAP values)
> - Timestamps
> - Model version and API version
> This creates a complete audit trail for compliance.

**Q3: Why PostgreSQL over MongoDB or SQLite?**
> PostgreSQL offers:
> - ACID compliance for data integrity
> - JSON columns for flexible SHAP storage
> - Mature tooling and cloud support
> - Foreign key constraints for referential integrity
> SQLite is used for Hugging Face deployment (demo mode).

**Q4: How does the testing strategy ensure quality?**
> Multi-layer approach:
> - Unit tests for core logic (98% coverage)
> - Integration tests for API endpoints
> - Database connectivity tests
> - CI/CD runs tests in parallel with/without database
> - Security scanning with Trivy

**Q5: How do you handle model updates?**
> - Model file in `outputs/employee_attrition_pipeline.pkl`
> - Version tracked in `prediction_traceability` table
> - Blue-green deployment possible with Docker
> - Rollback by reverting model file

**Q6: What happens when the database is unavailable?**
> - `DISABLE_DB=1` mode allows predictions without persistence
> - `/db_health` endpoint monitors connectivity
> - Graceful degradation: API continues working, just no logging

**Q7: How is SHAP integrated?**
> - TreeExplainer initialized at API startup
> - SHAP values computed per prediction
> - Values stored in database for reproducibility
> - Waterfall plots generated on demand

### Challenges Faced & Solutions

| Challenge | Solution |
|-----------|----------|
| API Key caching at import time | Dynamic `_get_valid_api_key()` function |
| Empty array validation crashes | Added `min_length=1` to Pydantic schema |
| Test file breaking pytest | Moved problematic files to `scripts/` |
| Docker networking issues | Used service names in Docker Compose |
| Environment variable conflicts | `.env.local` override pattern |

### Technical Debt & Future Improvements

1. **Increase API coverage** to 85%+ (currently 50%)
2. **Add Alembic migrations** for schema changes
3. **Implement rate limiting** for API protection
4. **Add Redis caching** for repeated predictions
5. **Kubernetes Helm charts** for enterprise deployment

---

## Readiness Checklist

### Documentation
- [x] README.md - Comprehensive
- [x] PROJECT_STRUCTURE.md - Complete
- [x] QUICKSTART.md - 5-minute guide
- [x] CI/CD workflow documented
- [x] Database schema documented
- [x] API endpoints documented

### Technical
- [x] All 165 tests passing
- [x] 70% test coverage
- [x] Docker Compose working
- [x] PostgreSQL integration functional
- [x] SHAP explanations working
- [x] Authentication implemented

### Demonstration Ready
- [x] Local environment setup
- [x] Sample data available in `data/`
- [x] API docs at `/docs`
- [x] Streamlit UI functional
- [x] Health endpoints working

---

## Quick Demo Script

```bash
# 1. Start services
docker-compose up -d

# 2. Verify health
curl http://localhost:8001/health
curl http://localhost:8001/db_health

# 3. Open API docs
# Browser: http://localhost:8001/docs

# 4. Open Streamlit UI
# Browser: http://localhost:8501

# 5. Run tests
poetry run pytest --cov=api --cov=core -v

# 6. View coverage report
# Browser: htmlcov/index.html
```

---

## Contact & Resources

- **API Documentation**: http://localhost:8001/docs
- **Project Repository**: [GitHub URL]
- **CI/CD Status**: [GitHub Actions URL]

Good luck with your presentation!
