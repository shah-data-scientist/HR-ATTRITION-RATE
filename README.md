# HR Attrition Rate - Employee Turnover Prediction

A machine learning-powered system to predict and analyze employee attrition risk, featuring a FastAPI backend and Streamlit frontend.

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Poetry (for dependency management)
- PostgreSQL 16+ (for data storage)
- Docker & Docker Compose (optional, for containerized deployment)

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

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start PostgreSQL**
   ```bash
   docker-compose up db -d
   ```

5. **Initialize the database**
   ```bash
   poetry run python database/init_db.py
   ```

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

```bash
docker-compose up
```

This will start:
- PostgreSQL database on port 5432
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

## 🏗️ Architecture

```
┌─────────────────────┐
│  Streamlit UI       │  (Port 8501)
│  ui/app.py          │
└──────────┬──────────┘
           │ HTTP API
           ▼
┌─────────────────────┐
│  FastAPI Backend    │  (Port 8001)
│  api/app/main.py    │
└──────────┬──────────┘
           │
           ├─► PostgreSQL DB (Port 5432)
           ├─► ML Model (outputs/)
           └─► SHAP Explainer
```

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
├── docker-compose.yml     # Docker orchestration
├── Dockerfile.api         # API container definition
├── Dockerfile.streamlit   # UI container definition
└── pyproject.toml        # Project dependencies
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Database
DATABASE_URL="postgresql://user:password@localhost:5432/hr_attrition_db"

# API
API_PORT="8001"
API_HOST="0.0.0.0"

# For UI to connect to API
API_BASE_URL="http://localhost:8001"

# Streamlit
STREAMLIT_SERVER_PORT="8501"
```

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

**Make Predictions:**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```

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

## 🐛 Troubleshooting

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

**Solution**: Train the model first:
```bash
poetry run python train.py
```

### Database Connection Issues

**Problem**: API can't connect to PostgreSQL

**Solution**:
1. Ensure PostgreSQL is running: `docker-compose up db -d`
2. Check `DATABASE_URL` in `.env`
3. Initialize database: `poetry run python database/init_db.py`

## 📚 Documentation

- [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup and guidelines
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment instructions
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture details
- [docs/archive/](docs/archive/) - Historical documentation

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
