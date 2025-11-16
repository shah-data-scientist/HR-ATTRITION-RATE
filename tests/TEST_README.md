# Automated Testing Guide

This directory contains automated tests for the HR Attrition Rate system, including API tests and UI tests with screenshots.

## Quick Start - Automated E2E Test

### Option 1: API Tests (Simple Script - No pytest required)

```bash
# Make sure API is running first
./scripts/start-api.sh  # or scripts\start-api.bat on Windows

# Run automated API test in another terminal
python tests/run_automated_test.py
```

### Option 2: UI Tests with Screenshots (Playwright)

```bash
# Install Playwright first (one time)
pip install playwright
playwright install chromium

# Make sure both API and UI are running
./scripts/start-api.sh   # Terminal 1
./scripts/start-ui.sh    # Terminal 2

# Run UI automation test in another terminal
python tests/run_ui_test.py
```

This will:
- ✅ Open the Streamlit UI in a browser
- ✅ Upload test data from `data/` folder
- ✅ Click "Predict Attrition" button
- ✅ Capture screenshots at each step
- ✅ Validate results are displayed
- ✅ Save screenshots to `test_screenshots/` folder
- ✅ Check API health
- ✅ Load test data from `data/` folder
- ✅ Test with 5 sample employees
- ✅ Validate prediction structure and SHAP values
- ✅ Process full dataset and generate statistics
- ✅ Save results to `test_results_automated.json`

### Option 3: Pytest Suite (Full test coverage)

```bash
# Run all automated tests
poetry run pytest tests/test_automated_e2e.py -v

# Run with detailed output
poetry run pytest tests/test_automated_e2e.py -v -s

# Run only fast tests (skip full dataset)
poetry run pytest tests/test_automated_e2e.py -v -m "not slow"

# Run full dataset test
poetry run pytest tests/test_automated_e2e.py -v -m slow

# Run UI tests (requires Playwright)
poetry run pytest tests/test_ui_automation.py -v -s
```

## Test Files

### API Tests
- **`run_automated_test.py`**: Standalone API test script
  - No pytest dependency
  - Can be run directly
  - Generates detailed console output
  - Saves results to JSON

- **`test_automated_e2e.py`**: Complete pytest test suite
  - Health check
  - Sample data prediction (5 rows)
  - Prediction structure validation
  - SHAP values verification
  - Full dataset test (marked as slow)
  - Error handling tests

### UI Tests (NEW)
- **`run_ui_test.py`**: Standalone UI test with Playwright
  - No pytest dependency
  - Automated browser interaction
  - Screenshots at each step
  - Saves to `test_screenshots/` folder
  
- **`test_ui_automation.py`**: Pytest UI test suite
  - Full UI workflow testing
  - Screenshot capture
  - Result validation

### Existing Tests
- `test_core.py`: Core data processing tests
- `test_database.py`: Database operations tests
- `test_e2e.py`: Original interactive E2E test
- `test_streamlit_api_call.py`: Streamlit API integration
- `verify_setup.py`: Setup verification script

## Test Data

Tests use data from the `data/` folder:
- `data/extrait_eval.csv` - Employee evaluation data
- `data/extrait_sirh.csv` - HR system data
- `data/extrait_sondage.csv` - Employee survey data

## UI Test Screenshots

The UI automation test (`run_ui_test.py`) captures screenshots at each step:

1. **01_ui_initial_load.png**: Streamlit UI loaded
2. **02_files_uploaded.png**: After uploading CSV files
3. **03_prediction_initiated.png**: After clicking Predict button
4. **04_results_top.png**: Top of results page
5. **05_results_full_page.png**: Full scrolled page with results

Screenshots are saved to `test_screenshots/` directory (not committed to git).

### UI Test Output Example:

```
======================================================================
  STREAMLIT UI - AUTOMATED TESTING WITH SCREENSHOTS
======================================================================

Configuration:
  UI URL: http://localhost:8501
  Data folder: /path/to/data
  Screenshots: /path/to/test_screenshots

[STEP 1] Launching Browser
----------------------------------------------------------------------

[STEP 2] Opening Streamlit UI
----------------------------------------------------------------------
  Navigating to: http://localhost:8501
  ✓ UI loaded successfully
  📸 Screenshot: 01_ui_initial_load.png

[STEP 3] Uploading CSV Files
----------------------------------------------------------------------
  ✓ Uploaded: extrait_eval.csv
  ✓ Uploaded: extrait_sirh.csv
  ✓ Uploaded: extrait_sondage.csv
  📸 Screenshot: 02_files_uploaded.png

[STEP 4] Running Prediction
----------------------------------------------------------------------
  ✓ Found button: 'Predict Attrition'
  ⏳ Waiting for predictions to complete...
  📸 Screenshot: 03_prediction_initiated.png

[STEP 5] Capturing Results
----------------------------------------------------------------------
  Results indicators:
    ✓ predictions
    ✓ probabilities
    ✓ risk categories
    ✓ employee data

  📸 Screenshot: 04_results_top.png
  📸 Screenshot: 05_results_full_page.png
  ✓ Results displayed successfully!

======================================================================
  TEST COMPLETED
======================================================================

📸 Screenshots saved to:
  test_screenshots/

✓ UI automation test completed!
```

## Expected Output

### Successful Test Output:
```
======================================================================
  HR ATTRITION - AUTOMATED TEST SUITE
======================================================================

[TEST] API Health Check
----------------------------------------------------------------------
✓ API is healthy: API is healthy

[TEST] Loading Test Data
----------------------------------------------------------------------
✓ Loaded extrait_eval.csv: 147 rows
✓ Loaded extrait_sirh.csv: 147 rows
✓ Loaded extrait_sondage.csv: 147 rows

[TEST] Prediction with Sample Data (5 rows)
----------------------------------------------------------------------
✓ Received 5 predictions

  Sample Result (Employee 1):
    Prediction: Leave
    Probability: 97.33%
    Risk Category: High
    Trace ID: 123
    SHAP Values: 45 features
    Base Value: -1.0380

[TEST] Validating Prediction Structure
----------------------------------------------------------------------
✓ All required fields present and valid
✓ Prediction values within expected ranges
✓ SHAP explanations included

[TEST] Full Dataset Prediction
----------------------------------------------------------------------
  Processing 147 employees...
✓ Successfully processed 147 employees

  Prediction Distribution:
    Leave: 35 (23.8%)
    Stay:  112 (76.2%)

  Risk Distribution:
    High:   25 (17.0%)
    Medium: 45 (30.6%)
    Low:    77 (52.4%)

  Average Attrition Probability: 32.45%

✓ Full results saved to: test_results_automated.json

======================================================================
  ALL TESTS PASSED ✓
======================================================================
```

## Troubleshooting

### Playwright Installation

If UI tests fail with "Playwright not installed":

```bash
# Install Playwright Python package
pip install playwright

# Install browser binaries
playwright install chromium

# Or install all browsers
playwright install
```

### API Not Running
```
✗ Failed to connect to API: [Errno 111] Connection refused
  Make sure API is running at: http://localhost:8001
```
**Solution**: Start the API first with `./scripts/start-api.sh`

### Port Mismatch
```
✗ Health check failed with status 404
```
**Solution**: Check that `API_BASE_URL` environment variable is set correctly
```bash
export API_BASE_URL="http://localhost:8001"  # Linux/Mac
set API_BASE_URL=http://localhost:8001        # Windows
```

### Data Not Found
```
✗ Failed to load test data: FileNotFoundError
```
**Solution**: Ensure you're running the test from the project root or that `data/` folder exists

## Running Tests in CI/CD

```bash
# Start services
docker-compose up -d

# Wait for API to be ready
sleep 10

# Run tests
poetry run pytest tests/test_automated_e2e.py -v --tb=short

# Cleanup
docker-compose down
```

## Test Markers

Tests are marked for selective execution:

- `@pytest.mark.slow`: Full dataset tests (can be skipped for faster CI)

Run specific markers:
```bash
# Skip slow tests
pytest tests/test_automated_e2e.py -m "not slow"

# Only slow tests
pytest tests/test_automated_e2e.py -m slow
```

## Output Files

Tests generate the following files:

- `test_results_automated.json`: Full prediction results from automated test
- `test_results_full.json`: Full dataset results from pytest
- `test_payload.json`: Sample request payload (from original e2e test)
- `test_response.json`: Sample response (from original e2e test)

These files can be used for debugging or analysis.

## Development

When adding new tests:

1. Add test to `test_automated_e2e.py` as a new test method
2. Follow naming convention: `test_<description>`
3. Use descriptive assertions with custom messages
4. Add print statements for detailed output
5. Mark slow tests with `@pytest.mark.slow`

Example:
```python
def test_new_feature(self, api_client, test_data):
    """Test description."""
    # Test implementation
    response = api_client.post("/endpoint", json=data)
    assert response.status_code == 200, "Custom error message"
    print("✓ Test passed with details")
```
