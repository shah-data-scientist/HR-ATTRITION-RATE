# Scripts Directory

This directory contains utility scripts for the HR Attrition Rate project.

## Startup Scripts

### `start-api.sh` / `start-api.bat`
Starts the FastAPI backend server on port 8001.

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

### `create_synthetic_data.py`
Generates synthetic employee data for testing purposes.

**Usage:**
```bash
poetry run python scripts/create_synthetic_data.py
```

### `utils.py`
Common utility functions shared across the project.

### `kill_pid.py`
Helper script to kill processes by PID (useful for freeing up ports).

**Usage:**
```bash
poetry run python scripts/kill_pid.py <PID>
```

## Debug Scripts

### `debug_api_call.py`
Debugging tool for testing API calls with detailed logging.

**Usage:**
```bash
poetry run python scripts/debug_api_call.py
```

### `debug_types.py`
Type checking and debugging utility for data structures.

**Usage:**
```bash
poetry run python scripts/debug_types.py
```

## Notes

- All scripts should be run from the project root directory
- Use `poetry run` to ensure scripts run in the correct environment
- Shell scripts (`.sh`) are for Linux/Mac, batch files (`.bat`) are for Windows
