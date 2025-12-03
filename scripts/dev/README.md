# Development Scripts

This directory contains development and debugging utilities that are not part of the production deployment.

## Purpose

These scripts help with:
- Database migrations and schema updates
- API debugging and testing
- Data alignment and validation
- Job queue testing
- Development diagnostics

## Files

### Database Tools
- **align_schema_with_data.py** - Align database schema with data files
- **migrate_db.py** - Database migration helper (note: also in parent `scripts/`)

### API Testing
- **call_predict_report.py** - Test the prediction report endpoint
- **debug_api_call.py** - Debug API calls with detailed logging
- **test_api.bat** - Windows batch script for API testing

### Job Queue Testing
- **enqueue_sample_report_job.py** - Add a sample job to the report queue for testing

### Debugging Utilities
- **debug_types.py** - Type checking and validation debugging
- **e2e_test.py** - End-to-end test script
- **quick_ui_diagnostics.py** - Quick UI health check and diagnostics

## Usage

Run scripts from the project root using Poetry:

```bash
# Database alignment
poetry run python scripts/dev/align_schema_with_data.py

# Test API call with debugging
poetry run python scripts/dev/debug_api_call.py

# UI diagnostics
poetry run python scripts/dev/quick_ui_diagnostics.py

# Enqueue test job
poetry run python scripts/dev/enqueue_sample_report_job.py
```

### Windows Batch Scripts

```cmd
# API testing (Windows)
scripts\dev\test_api.bat
```

## Production Scripts

For production-ready scripts, see the parent `scripts/` directory:
- **start-api.sh / start-api.bat** - Start the API server
- **start-ui.sh / start-ui.bat** - Start the Streamlit UI
- **worker.py** - Background worker for async jobs
- **create_synthetic_data.py** - Generate synthetic test data

## When to Use

### Development Scripts (This Directory)
Use these when:
- Debugging API issues
- Testing database migrations
- Validating data alignment
- Running diagnostic checks
- Testing job queue functionality

### Production Scripts (Parent Directory)
Use these for:
- Starting services in development or production
- Running the background worker
- Creating test data for demos

## Requirements

Most scripts require:
- Poetry environment activated
- Database connection (PostgreSQL or SQLite)
- API running (for API testing scripts)
- Environment variables configured (`.env` file)

Check individual scripts for specific requirements.

## Adding New Scripts

When adding new development scripts:
1. Place them in this directory if they're debug/dev-only
2. Use the parent `scripts/` directory for production utilities
3. Add descriptive docstrings explaining purpose and usage
4. Update this README with the new script description
5. Include error handling and helpful error messages

## Safety Notes

⚠️ **Database Scripts:**
- Always backup the database before running migration scripts
- Test on a development database first
- Review SQL queries before execution

⚠️ **API Scripts:**
- Use development/staging API endpoints, not production
- Be cautious with rate limits
- Never commit API keys in scripts

⚠️ **Job Queue Scripts:**
- Understand the job's impact before enqueuing
- Monitor worker logs when testing
- Clean up test jobs after validation

## Automated Testing

For automated test scripts, use the pytest suite:
```bash
poetry run pytest tests/
```

These development scripts are for manual debugging and one-off tasks.
