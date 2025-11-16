# Development Guide

This guide covers development setup, workflow, and best practices for the HR Attrition Rate project.

## 🛠️ Development Setup

### Prerequisites

- Python 3.12+
- Poetry 1.8+
- PostgreSQL 16+ (or use Docker)
- Git
- VS Code or PyCharm (recommended)

### Initial Setup

1. **Clone and enter the repository**
   ```bash
   git clone <repository-url>
   cd hr-attrition-rate
   ```

2. **Install Poetry** (if not already installed)
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**
   ```bash
   poetry install
   ```

4. **Activate the virtual environment**
   ```bash
   poetry shell
   ```

5. **Set up pre-commit hooks** (optional but recommended)
   ```bash
   poetry run pre-commit install
   ```

### Environment Configuration

1. **Copy the example environment file**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your local settings**
   ```bash
   # For local development
   DATABASE_URL="postgresql://user:password@localhost:5432/hr_attrition_db"
   API_BASE_URL="http://localhost:8001"
   API_PORT="8001"
   STREAMLIT_SERVER_PORT="8501"
   ```

### Database Setup

#### Option 1: Using Docker (Recommended)

```bash
# Start PostgreSQL container
docker-compose up db -d

# Initialize the database
poetry run python database/init_db.py
```

#### Option 2: Local PostgreSQL

```bash
# Create database
createdb hr_attrition_db

# Initialize the database
poetry run python database/init_db.py
```

## 🏃 Running the Application

### Development Mode

For development, run the API and UI in separate terminals with hot-reload enabled:

**Terminal 1 - API Server:**
```bash
./scripts/start-api.sh
# or on Windows: scripts\start-api.bat
```

**Terminal 2 - Streamlit UI:**
```bash
./scripts/start-ui.sh
# or on Windows: scripts\start-ui.bat
```

### Manual Commands

If you prefer manual control:

**Start API with hot-reload:**
```bash
poetry run uvicorn api.app.main:app --host 0.0.0.0 --port 8001 --reload
```

**Start UI with auto-reload:**
```bash
poetry run streamlit run ui/app.py --server.port 8501
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=api --cov=core --cov=database --cov-report=html

# Run specific test file
poetry run pytest tests/test_core.py

# Run tests matching a pattern
poetry run pytest -k "test_prediction"

# Run tests with verbose output
poetry run pytest -v

# Run tests and stop at first failure
poetry run pytest -x
```

### Writing Tests

Tests are organized in the `tests/` directory:

```
tests/
├── conftest.py                    # Shared fixtures
├── test_core.py                   # Core logic tests
├── test_database.py               # Database tests
├── test_streamlit_api_call.py     # UI API integration
├── test_e2e.py                    # End-to-end tests
└── ...
```

**Example test:**
```python
def test_prediction_api(client):
    """Test the prediction endpoint."""
    payload = {
        "eval_data": [...],
        "sirh_data": [...],
        "sondage_data": [...]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predictions" in response.json()
```

## 🔍 Code Quality

### Linting and Formatting

We use Ruff for both linting and formatting:

```bash
# Run linter
poetry run ruff check .

# Auto-fix issues
poetry run ruff check . --fix

# Format code
poetry run ruff format .
```

### Type Checking

```bash
# Run type checker
poetry run mypy api core database
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code before commits:

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

## 📦 Dependencies

### Adding Dependencies

```bash
# Add a production dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Lock dependencies without installing
poetry lock --no-update
```

### Dependency Groups

- **main**: Production dependencies (FastAPI, Streamlit, etc.)
- **dev**: Development tools (pytest, ruff, mypy, etc.)

## 🏗️ Project Structure

### Key Directories

- **`api/`**: FastAPI backend
  - `app/main.py`: API endpoints and startup logic
  - `app/schemas.py`: Pydantic models for API
  - `tests/`: API-specific tests

- **`core/`**: Business logic (shared between API and UI)
  - `data_processing.py`: Feature engineering
  - `preprocess.py`: Data preprocessing
  - `schema.py`: Data schemas
  - `validation.py`: Validation logic

- **`database/`**: Database layer
  - `models.py`: SQLAlchemy models
  - `database.py`: Connection management
  - `init_db.py`: Database initialization

- **`ui/`**: Streamlit frontend
  - `app.py`: Main UI application

- **`tests/`**: Test suite
- **`scripts/`**: Utility scripts
- **`data/`**: Sample data files
- **`outputs/`**: Model artifacts

### Configuration Files

- **`pyproject.toml`**: Project metadata and dependencies
- **`poetry.lock`**: Locked dependency versions
- **`docker-compose.yml`**: Docker orchestration
- **`.env`**: Environment variables (not committed)
- **`.env.example`**: Environment template

## 🔄 Development Workflow

### Feature Development

1. **Create a feature branch**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes**
   - Write code following existing patterns
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and quality checks**
   ```bash
   poetry run pytest
   poetry run ruff check . --fix
   poetry run mypy api core database
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `refactor:` Code refactoring
   - `test:` Test changes
   - `chore:` Maintenance tasks

5. **Push and create a pull request**
   ```bash
   git push origin feat/your-feature-name
   ```

### Bug Fixes

1. **Create a bugfix branch**
   ```bash
   git checkout -b fix/bug-description
   ```

2. **Write a failing test** that reproduces the bug

3. **Fix the bug** and ensure tests pass

4. **Commit and push**
   ```bash
   git commit -m "fix: description of bug fix"
   git push origin fix/bug-description
   ```

## 🐛 Debugging

### Debugging the API

**Using VS Code:**
1. Add a breakpoint in the code
2. Use the "Python: FastAPI" debug configuration
3. Start debugging (F5)

**Using print statements:**
```python
import logging
logger = logging.getLogger("uvicorn.error")
logger.debug(f"Debug info: {variable}")
```

### Debugging Streamlit

**Using print statements:**
```python
import streamlit as st
st.write("Debug:", variable)
print("Console debug:", variable)
```

**Check Streamlit logs:**
Look at the terminal where you started Streamlit for error messages.

### Database Debugging

**Connect to PostgreSQL:**
```bash
# Using Docker
docker exec -it hr-attrition-rate-db-1 psql -U user -d hr_attrition_db

# Local PostgreSQL
psql -U user -d hr_attrition_db
```

**Check tables:**
```sql
\dt                           -- List tables
SELECT * FROM employees LIMIT 5;  -- Query data
```

## 🚀 Performance

### Profiling

**Profile API endpoints:**
```bash
poetry add --group dev py-spy
py-spy top --pid <fastapi-pid>
```

**Profile code:**
```python
import cProfile
import pstats

with cProfile.Profile() as pr:
    # Your code here
    pass

stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

## 📝 Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Follow Google style for docstrings
- Keep comments up-to-date with code changes

**Example:**
```python
def predict_attrition(employee_data: pd.DataFrame) -> dict:
    """Predict attrition risk for employees.
    
    Args:
        employee_data: DataFrame containing employee features
        
    Returns:
        Dictionary with predictions and probabilities
        
    Raises:
        ValueError: If required columns are missing
    """
    pass
```

### Updating Documentation

When adding features:
1. Update relevant `.md` files
2. Update inline code comments
3. Update API docs (FastAPI auto-generates from docstrings)
4. Update README if user-facing

## 🔐 Security

### Best Practices

1. **Never commit secrets**
   - Use `.env` for sensitive data
   - Check `.gitignore` includes `.env`

2. **Validate all inputs**
   - Use Pydantic models for API validation
   - Sanitize user uploads

3. **Keep dependencies updated**
   ```bash
   poetry update
   poetry show --outdated
   ```

4. **Run security checks**
   ```bash
   poetry run bandit -r api core database
   ```

## 🆘 Common Issues

### Issue: Import errors

**Solution:** Ensure you're in the Poetry virtual environment
```bash
poetry shell
```

### Issue: Database connection errors

**Solution:** Check PostgreSQL is running and credentials are correct
```bash
docker-compose ps
psql -U user -h localhost -d hr_attrition_db
```

### Issue: Port already in use

**Solution:** Kill the process using the port
```bash
# Find process
lsof -i :8001  # Linux/Mac
netstat -ano | findstr :8001  # Windows

# Kill process
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows
```

### Issue: Model not found

**Solution:** Train the model
```bash
poetry run python train.py
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Pytest Documentation](https://docs.pytest.org/)
