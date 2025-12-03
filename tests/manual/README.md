# Manual Test Scripts

This directory contains one-off test scripts and utilities used for manual testing and debugging during development.

## Purpose

These scripts are **not part of the automated test suite** (which uses pytest in the parent `tests/` directory). Instead, they are:
- Interactive debugging tools
- Manual verification scripts
- Data exploration utilities
- One-time test scenarios

## Files

### Employee Testing Scripts
- **test_four_employees.py** - Test predictions for a specific set of 4 employees
- **test_multiple_new.py** - Test batch predictions for multiple new employees
- **test_new_employee.py** - Test single new employee prediction
- **test_single_employee_prediction.py** - Alternative single prediction test

### UI Testing
- **test_ui_manual.py** - Manual UI testing script
- **test_streamlit_interaction.py** - Streamlit app flow integration test (requires updated testing API)

### API Testing
- **test_api_with_core.py** - API schema validation and integration test
- **test_streamlit_api_call.py** - Test Streamlit app's API call function (requires API running)

### Debug Utilities
- **quick_test.py** - Quick ad-hoc testing script
- **show_employee_records.py** - Display employee records from the database
- **show_shap_details.py** - Display SHAP explanation details for predictions

## Usage

These scripts are designed to be run manually from the project root:

```bash
# Example: Test a single employee prediction
poetry run python tests/manual/test_new_employee.py

# Example: Display employee records
poetry run python tests/manual/show_employee_records.py

# Example: Show SHAP details
poetry run python tests/manual/show_shap_details.py
```

## When to Use

Use these scripts when you need to:
- Manually verify specific edge cases
- Debug prediction behavior for specific employees
- Explore database contents
- Test API endpoints interactively
- Investigate SHAP explanations

## Automated Tests

For automated testing, use the pytest suite in the parent directory:
```bash
poetry run pytest tests/
```

## Adding New Manual Tests

When creating new manual test scripts:
1. Place them in this directory
2. Use descriptive names (e.g., `test_specific_scenario.py`)
3. Add docstrings explaining the purpose
4. Document any dependencies or setup requirements

## Note

These scripts may:
- Require the API to be running
- Access the database directly
- Make assumptions about data state
- Not be idempotent (safe to run multiple times)

Always check the script contents before running to understand what it does.
