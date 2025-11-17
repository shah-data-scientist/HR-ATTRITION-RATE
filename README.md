# HR Attrition Rate - Employee Turnover Prediction

A machine learning-powered system to predict and analyze employee attrition risk, featuring a FastAPI backend and Streamlit frontend.

## 🚀 Quick Start

### Prerequisites

- Python 3.13+ (managed via Poetry)
- Poetry (for dependency management)
- PostgreSQL 16+ or SQLite (for data storage)
- Docker & Docker Compose (optional, for containerized deployment)

### Security Setup

**IMPORTANT**: Before running the application, configure your API key:

1. **Copy environment template**
   ```bash
   cp .env.example .env
   ```

2. **Generate a secure API key** (recommended)
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

3. **Update `.env` file**
   ```bash
   # Replace with your generated key
   API_KEY=your_generated_secure_api_key_here
   SECRET_KEY=your_generated_secret_key_min_32_chars
   
   # Database credentials
   POSTGRES_USER=user
   POSTGRES_PASSWORD=strong_password_here
   POSTGRES_DB=hr_attrition_db
   ```

**⚠️ Security Notes:**
- Never commit `.env` files to version control
- Use strong, unique passwords for production
- Rotate API keys regularly
- Store secrets in environment variables or secret managers (AWS Secrets Manager, Azure Key Vault)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hr-attrition-rate
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Configure environment variables** (see [Security Setup](#security-setup))
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database credentials
   ```

4. **Run with Docker** (simplest option)
   ```bash
   docker-compose up -d
   ```
   This starts the database, API, and UI automatically.

### Running the Application

#### Option 1: Local Development (Recommended for Development)

**On Linux/Mac:**

Open two terminal windows:

Terminal 1 - Start the API:
```bash
./scripts/start-api.sh
```

Terminal 2 - Start the UI:
```bash
./scripts/start-ui.sh
```

**On Windows:**

Open two command prompts:

Command Prompt 1 - Start the API:
```cmd
scripts\start-api.bat
```

Command Prompt 2 - Start the UI:
```cmd
scripts\start-ui.bat
```

#### Option 2: Docker Deployment (Recommended for Production)

**Development Environment:**
```bash
docker-compose up
```

**Production Environment:**
```bash
# Use production configuration with replicas and log rotation
docker-compose -f docker-compose.prod.yml up -d
```

**Production Features:**
- 🔄 2 replicas for API and worker services
- 📊 Log rotation (10MB max, 3-5 files)
- ⚡ Resource limits (CPU/memory)
- 🔐 Non-root users in all containers
- 🚀 Multi-stage Docker builds for smaller images

This will start:
- PostgreSQL database on port 5432 (internal only)
- FastAPI backend on http://localhost:8001
- Streamlit UI on http://localhost:8501

### Access the Application

- **Streamlit UI**: http://localhost:8501
- **API Documentation**: http://localhost:8001/docs
- **API Health Check**: http://localhost:8001/health

## 📋 Features

- **Predictive Analytics**: Machine learning model to predict employee attrition
- **SHAP Explanations**: Interpretable AI with feature importance visualization
- **REST API**: FastAPI-based backend for programmatic access
- **Interactive UI**: Streamlit-based dashboard for easy data upload and analysis
- **Database Logging**: Full traceability of predictions in PostgreSQL
- **Docker Support**: Containerized deployment for easy scaling
- **Authentication**: API key-based security for all prediction endpoints
- **Security Middleware**: XSS, CSRF, CSP, HSTS protection
- **CI/CD Pipeline**: Automated testing, security scanning, and deployment
- **Test Coverage**: 74% overall (52% API, 98% data processing, 85% preprocessing)

## 🧪 Testing & Quality

### Test Coverage Report

```
Module                    Coverage
----------------------------------------
api/app/main.py          52%  (170/324 statements)
core/__init__.py         100% (5/5 statements)
core/data_processing.py  98%  (50/51 statements)
core/preprocess.py       85%  (33/39 statements)
core/schema.py           96%  (24/25 statements)
core/validation.py       93%  (62/67 statements)
database/models.py       100% (28/28 statements)
----------------------------------------
TOTAL                    74%  (372/501 statements)
```

**Coverage HTML Report**: Available at `htmlcov/index.html` after running tests

### Running Tests

```bash
# Run all tests with coverage
poetry run pytest --cov=api --cov=core --cov=database --cov-report=term --cov-report=html

# Run specific test suite
poetry run pytest tests/test_core.py -v

# Run with database disabled (Hugging Face mode)
DISABLE_DB=1 poetry run pytest
```

### CI/CD Pipeline

The project includes a comprehensive GitHub Actions pipeline:

**Code Quality:**
- ✅ Black formatting checks
- ✅ Mypy type checking

**Security:**
- ✅ Trivy vulnerability scanning (filesystem + Docker images)
- ✅ SARIF reports uploaded to GitHub Security

**Testing:**
- ✅ Tests with PostgreSQL integration
- ✅ Tests without database (DISABLE_DB mode)
- ✅ Authentication module tests
- ✅ Coverage reporting to Codecov

**Docker:**
- ✅ API and UI image builds
- ✅ Hugging Face deployment image
- ✅ Push to GitHub Container Registry (ghcr.io)

**Deployment (Optional):**
- ✅ Staging deployment with manual approval
- ✅ Production deployment with environment protection

See [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml) for details.

## 🏗️ Architecture

```
┌─────────────────────┐
│  Streamlit UI       │  (Port 8501)
│  ui/app.py          │
└──────────┬──────────┘
           │ HTTP API (X-API-Key required)
           ▼
┌─────────────────────┐
│  FastAPI Backend    │  (Port 8001)
│  api/app/main.py    │
│  + Auth Middleware  │
└──────────┬──────────┘
           │
           ├─► PostgreSQL DB (Port 5432)
           ├─► ML Model (outputs/)
           └─► SHAP Explainer
```

**Security Features:**
- 🔐 API Key authentication on all prediction endpoints
- 🛡️ Security headers (XSS, CSRF, CSP, HSTS)
- 📝 Request logging and monitoring
- 🔄 CORS protection
- 🗜️ GZip compression

See [docs/ER_DIAGRAM.md](docs/ER_DIAGRAM.md) for database schema details.

## 📁 Project Structure

```
hr-attrition-rate/
├── api/                    # FastAPI backend application
│   ├── app/
│   │   ├── main.py        # Main API application
│   │   └── schemas.py     # Pydantic data models
│   └── tests/             # API tests
├── core/                   # Core business logic
│   ├── data_processing.py # Feature engineering
│   ├── preprocess.py      # Data preprocessing
│   ├── schema.py          # Data schemas
│   └── validation.py      # Data validation
├── database/              # Database models and utilities
│   ├── models.py          # SQLAlchemy models
│   ├── database.py        # Database connection
│   └── init_db.py         # Database initialization
├── ui/                    # Streamlit frontend
│   └── app.py            # Main UI application
├── data/                  # Sample data files
├── outputs/               # Model artifacts and outputs
├── scripts/               # Utility scripts
│   ├── start-api.sh      # Start API (Linux/Mac)
│   ├── start-ui.sh       # Start UI (Linux/Mac)
│   ├── start-api.bat     # Start API (Windows)
│   └── start-ui.bat      # Start UI (Windows)
├── tests/                 # Test suite
├── docs/                  # Documentation
├── docker/                # Docker configuration
│   ├── Dockerfile.api         # API container
│   ├── Dockerfile.streamlit   # UI container
│   ├── Dockerfile.database    # DB initialization
│   └── Dockerfile.huggingface # Unified HF deployment
├── docker-compose.yml     # Docker orchestration
└── pyproject.toml        # Project dependencies
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Authentication (REQUIRED)
API_KEY="changeme_secure_api_key_here"           # API authentication key
SECRET_KEY="changeme_min_32_chars_secret_key"   # Application secret key

# Database
POSTGRES_USER="user"
POSTGRES_PASSWORD="password"
POSTGRES_DB="hr_attrition_db"
DATABASE_URL="postgresql://user:password@localhost:5432/hr_attrition_db"

# API Configuration
API_PORT="8001"
API_HOST="0.0.0.0"

# For UI to connect to API
API_BASE_URL="http://localhost:8001"

# Streamlit
STREAMLIT_SERVER_PORT="8501"

# Worker Configuration
WORKER_POLL_SEC="5"                              # Job polling interval
WORKER_STALE_SEC="300"                           # Job stale timeout

# Disable database mode (for Hugging Face deployment)
DISABLE_DB="0"                                   # Set to "1" to disable DB
```

**Security Best Practices:**
- Generate strong random keys: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- Use different keys for development and production
- Store production secrets in environment managers (AWS Secrets Manager, Azure Key Vault)
- Never commit `.env` files to Git (.gitignore already configured)

### Port Configuration

The application uses the following default ports:
- **API**: 8001 (configurable via `API_PORT`)
- **UI**: 8501 (configurable via `STREAMLIT_SERVER_PORT`)
- **Database**: 5432 (standard PostgreSQL port)

## 📊 Usage

### Using the Streamlit UI

1. Navigate to http://localhost:8501
2. Upload three CSV files:
   - `extrait_eval.csv` - Employee evaluation data
   - `extrait_sirh.csv` - HR system data
   - `extrait_sondage.csv` - Employee survey data
3. Click "Predict Attrition"
4. View results and download the Excel report
5. Explore SHAP explanations for each employee

### Using the API

**Health Check:**
```bash
curl http://localhost:8001/health
```

**Make Predictions** (requires API key):
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d @sample_payload.json
```

**Authentication:**
All prediction endpoints require the `X-API-Key` header:
- `/predict` - Batch predictions
- `/predict_report` - Generate Excel reports
- `/predict_excel` - Upload Excel for predictions
- `/predict_shap_images` - SHAP visualizations
- `/jobs/report` - Report job creation

See API documentation at http://localhost:8001/docs for detailed endpoint information.

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=api --cov=core --cov=database

# Run specific test file
poetry run pytest tests/test_core.py
```

**Current Coverage**: 74% overall (see [Testing & Quality](#-testing--quality) section for breakdown)

## 🐛 Troubleshooting

### Authentication Error (401/403)

**Problem**: API returns "Unauthorized" or "Forbidden"
```json
{"detail": "API key missing"}
{"detail": "Invalid API key"}
```

**Solution**:
1. Ensure `API_KEY` is set in `.env`
2. Verify the UI is using the same API key
3. Check the `X-API-Key` header is included in requests
4. For local development, restart both API and UI after changing `.env`

### Connection Refused Error

**Problem**: UI can't connect to API
```
Network error while connecting to API: [WinError 10061] No connection could be made because the target machine actively refused it
```

**Solution**:
1. Ensure the API is running: `curl http://localhost:8001/health`
2. Check that ports match in your configuration
3. For Docker: Use service names (e.g., `http://fastapi_app:8001`)
4. For local dev: Use `http://localhost:8001`

### Model Not Found

**Problem**: API fails to start with "Model file not found"

**Solution**: Ensure the model file exists in `outputs/employee_attrition_pipeline.pkl`. The model should be trained and committed to the repository. If missing, contact the repository maintainer.

### Database Connection Issues

**Problem**: API can't connect to PostgreSQL

**Solution**:
1. Ensure PostgreSQL is running: `docker-compose up db -d`
2. Check `DATABASE_URL` in `.env`
3. Initialize database: `poetry run python database/init_db.py`

## 📚 Documentation

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development workflow and guidelines

### Deployment
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment (Docker, AWS, Azure, K8s)
- **[docs/deployment/HUGGINGFACE_DEPLOYMENT.md](docs/deployment/HUGGINGFACE_DEPLOYMENT.md)** - Deploy to Hugging Face Spaces

### Architecture & Technical Details
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture details
- **[docs/ER_DIAGRAM.md](docs/ER_DIAGRAM.md)** - Database schema and entity relationships
- **[api/README.md](api/README.md)** - API documentation
- **[.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml)** - CI/CD pipeline

### Technical Justifications

#### 1. **API Key Authentication**
**Rationale**: Simple, stateless authentication suitable for machine-to-machine communication. Chosen over JWT for:
- Lower complexity (no token refresh/expiration management)
- Better for API-to-API communication
- Easier to implement and debug
- Sufficient for internal/controlled access scenarios

**Implementation**: `api/auth.py` with bcrypt password hashing and secure key generation

#### 2. **Multi-Stage Docker Builds**
**Rationale**: Reduces image size by 60-70%, improves security, and speeds up deployment:
- Builder stage: Installs dependencies and builds artifacts
- Runtime stage: Only includes necessary runtime dependencies
- Result: Smaller attack surface, faster container startup

**Implementation**: All Dockerfiles use Python 3.13-slim base with non-root users

#### 3. **PostgreSQL for Traceability**
**Rationale**: Full ACID compliance ensures prediction auditability and regulatory compliance:
- Separate tables for inputs, outputs, and traces
- Foreign key constraints maintain data integrity
- JSON columns for flexibility with SHAP values
- Timestamps for audit trails

**Alternative**: SQLite for local development (no Docker required)

#### 4. **SHAP for Explainability**
**Rationale**: Industry-standard interpretability method based on game theory:
- Model-agnostic: Works with any black-box model
- Theoretically sound: Shapley values have desirable properties
- Visual outputs: Waterfall, force, and summary plots
- Regulatory compliance: Meets GDPR "right to explanation"

**Implementation**: Integrated into prediction pipeline, stored in database for reproducibility

#### 5. **CI/CD with GitHub Actions**
**Rationale**: Automated quality gates prevent regressions and ensure production readiness:
- Code quality: Black formatting, Mypy type checking
- Security: Trivy scanning for vulnerabilities
- Testing: 74% coverage with PostgreSQL integration tests
- Deployment: Automated Docker builds and optional staging/production deployment

**Cost**: Free for public repositories, integrated with GitHub

#### 6. **Non-Root Docker Containers**
**Rationale**: Security best practice to limit container breakout impact:
- API runs as `appuser` (UID 1000)
- UI runs as `appuser` (UID 1001)
- Database init runs as `dbuser` (UID 1002)
- Complies with CIS Docker Benchmark

**Trade-off**: Requires careful file permission management

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make your changes
3. Run tests: `poetry run pytest`
4. Commit using conventional commits: `git commit -m "feat: add new feature"`
5. Push and create a pull request

## 📄 License

[Your License Here]

## 🆘 Support

For issues and questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review [docs/archive/](docs/archive/) for additional context
3. Open an issue on GitHub
