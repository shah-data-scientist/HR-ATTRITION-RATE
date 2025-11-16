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

## Utility Scripts

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

## Notes

- All scripts should be run from the project root directory
- Use `poetry run` to ensure scripts execute in the correct Python environment
- Shell scripts (`.sh`) are for Linux/Mac, batch files (`.bat`) are for Windows
- For database initialization, use: `poetry run python -m database.init_db`
