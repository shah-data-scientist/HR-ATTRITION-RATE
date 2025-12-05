# Scripts Directory

This directory contains utility scripts for the HR Attrition Rate project.

## Startup Scripts

### `start-api.sh` / `start-api.bat`
Starts the FastAPI backend server on port 8001 with hot-reload enabled.

**Usage:**
```bash
# Linux/Mac
./scripts/start-api.sh

# Windows
scripts\start-api.bat
```

### `start-ui.sh` / `start-ui.bat`
Starts the Streamlit UI on port 8501.

**Usage:**
```bash
# Linux/Mac
./scripts/start-ui.sh

# Windows
scripts\start-ui.bat
```

### `start-api-nodb.bat`
Starts the API without a database connection (useful for lightweight testing).

**Usage:**
```cmd
scripts\start-api-nodb.bat
```

## Service Scripts

### `worker.py`
The background worker process that handles asynchronous tasks like report generation.

**Usage:**
```bash
poetry run python scripts/worker.py
```

## Database Scripts

### `create_tables.py`
Creates necessary database tables based on the SQLAlchemy models.

**Usage:**
```bash
poetry run python scripts/create_tables.py
```

### `migrate_db.py`
Handles database schema migrations, such as adding new columns or tables.

**Usage:**
```bash
poetry run python scripts/migrate_db.py
```

## Data & Testing Scripts

### `utils.py`
Common utility functions for data loading and merging. Used by `database/init_db.py` and other scripts.

**Key functions:**
- `load_and_merge_data()` - Loads and merges the three CSV data sources
- Data cleaning and preprocessing utilities

### `create_synthetic_data.py`
Generates synthetic employee data for testing purposes.

**Usage:**
```bash
poetry run python scripts/create_synthetic_data.py
```

## Development Scripts

The `scripts/dev/` directory contains scripts used primarily for debugging, diagnostics, and ad-hoc testing. See [scripts/dev/README.md](dev/README.md) for details.

## Notes

- All scripts should be run from the project root directory
- Use `poetry run` to ensure scripts execute in the correct Python environment
- Shell scripts (`.sh`) are for Linux/Mac, batch files (`.bat`) are for Windows
- For database initialization, use: `poetry run python -m database.init_db`