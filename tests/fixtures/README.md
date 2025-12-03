# Test Fixtures

This directory contains test data files used for manual testing, debugging, and development.

## Purpose

These files provide:
- Sample API request payloads
- Expected API responses
- SQL queries for database testing
- Error logs and debugging output

## Files

### JSON Payloads
- **test_payload.json** - Sample API request payload for predictions
- **test_response.json** - Expected API response format
- **streamlit_simulation_payload.json** - Simulated Streamlit UI request
- **streamlit_simulation_response.json** - Simulated Streamlit UI response
- **temp_api_response.json** - Temporary API response for debugging

### SQL Queries
- **query_employee_88888.sql** - Query to retrieve specific employee data
- **test_shap_insert.sql** - Test SQL for SHAP value insertion

### Debug Output
- **test_error.txt** - Error messages from test runs
- **test_output.txt** - Standard output from test runs

## Usage

### Using JSON Payloads

Test API endpoints with curl:
```bash
# Test prediction endpoint
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d @tests/fixtures/test_payload.json
```

### Using SQL Queries

Run SQL queries against the database:
```bash
# PostgreSQL
psql -U user -d hr_attrition_db -f tests/fixtures/query_employee_88888.sql

# SQLite (for local development)
sqlite3 hr_attrition.db < tests/fixtures/query_employee_88888.sql
```

### Examining Debug Output

Review error logs and test output:
```bash
# View error logs
cat tests/fixtures/test_error.txt

# View test output
cat tests/fixtures/test_output.txt
```

## .gitignore

Note: Generated JSON and TXT files in this directory are ignored by Git (see root `.gitignore`):
```gitignore
tests/fixtures/*.json
tests/fixtures/*.txt
```

Only committed fixtures are tracked. This prevents temporary test outputs from cluttering the repository.

## Automated Tests

For fixtures used in automated tests, see:
- **tests/conftest.py** - Pytest fixtures and test configuration
- **tests/test_*.py** - Test files that may generate temporary fixtures

## Adding New Fixtures

When adding new test data:
1. Use descriptive filenames
2. Add a comment in this README explaining the purpose
3. For JSON: validate the format matches the API schema
4. For SQL: ensure queries are safe and don't modify production data
5. Commit only fixtures needed for reproducible testing

## Sample Data

For CSV sample data used by the application, see the `data/` directory in the project root:
- `data/extrait_eval.csv` - Employee evaluation data
- `data/extrait_sirh.csv` - HR system data
- `data/extrait_sondage.csv` - Employee survey data

## Security Note

Never commit files containing:
- Real API keys
- Production database credentials
- Personally Identifiable Information (PII)
- Sensitive employee data

Use synthetic or anonymized data only.
