# Manual UI Test Results
**Date:** November 17, 2025  
**Test Type:** Manual API testing simulating UI workflow  
**Test Employees:** 5 from training dataset (IDs: 1, 2, 4, 5, 7)

## Test Execution Summary

### ✅ Successful Operations

1. **API Predictions**: All 5 employee predictions completed successfully
   - Employee 1: 89.72% Leave probability (High Risk) - Trace ID 22
   - Employee 2: 4.23% Leave probability (Low Risk) - Trace ID 23
   - Employee 4: 90.93% Leave probability (High Risk) - Trace ID 24
   - Employee 5: 28.92% Leave probability (Low Risk) - Trace ID 25
   - Employee 7: 63.94% Leave probability (High Risk) - Trace ID 26

2. **Database Records**: All core tables populated correctly
   - ✅ `employees` table: 5 records created
   - ✅ `model_inputs` table: 5 records with full feature JSON
   - ✅ `model_outputs` table: 5 records with predictions & probabilities
   - ✅ `predictions_traceability` table: 5 records with trace IDs & metadata

3. **SHAP Computation**: All 5 predictions returned SHAP analysis in API response
   - 66 features per employee
   - Base value: -1.0380 for all
   - Feature names and SHAP values included in response JSON

### ❌ Failed Operations

1. **SHAP Database Storage**: NO SHAP records saved despite computation
   ```sql
   SELECT has_shap FROM predictions WHERE trace_id IN (22,23,24,25,26)
   -- Result: has_shap = 'NO' for all 5 records
   
   SELECT COUNT(*) FROM shap_analysis;
   -- Result: 0 rows (empty table)
   ```

2. **User ID Headers**: Custom headers not being applied
   - Sent: X-User-ID values ("test_user_1" through "test_user_5")
   - Stored: All records show "demo1" (default value)
   - Indicates HTTP header not being read correctly

## Database Verification Query Results

```sql
SELECT 
    e.id_employee,
    e.user_id,
    e.age,
    e.genre,
    e.departement,
    t.trace_id,
    o.risk_category,
    o.prediction_proba,
    CASE WHEN s.shap_id IS NOT NULL THEN 'YES' ELSE 'NO' END as has_shap
FROM employees e
JOIN model_inputs mi ON e.id_employee = mi.id_employee
JOIN predictions_traceability t ON mi.input_id = t.input_id
JOIN model_outputs o ON t.output_id = o.output_id
LEFT JOIN shap_analysis s ON t.trace_id = s.trace_id
WHERE e.id_employee IN (1, 2, 4, 5, 7)
ORDER BY t.trace_id DESC;
```

**Results:**
| id_employee | user_id | age | genre | departement | trace_id | risk_category | prediction_proba | has_shap |
|-------------|---------|-----|-------|-------------|----------|---------------|------------------|----------|
| 7 | demo1 | 27 | M | Consulting | 26 | High | 0.6394 | NO |
| 5 | demo1 | 33 | F | Consulting | 25 | Low | 0.2892 | NO |
| 4 | demo1 | 37 | M | Consulting | 24 | High | 0.9093 | NO |
| 2 | demo1 | 49 | M | Consulting | 23 | Low | 0.0423 | NO |
| 1 | demo1 | 41 | F | Commercial | 22 | High | 0.8972 | NO |

## Root Cause Analysis

### SHAP Storage Bug
**Symptom:** SHAP values computed but not saved to `shap_analysis` table  
**Evidence:**
- API response includes SHAP values (66 features)
- Database table `shap_analysis` remains empty (0 rows)
- Logging statements in SHAP save code block don't appear in logs

**Hypothesis:**
The code path containing the SHAP storage logic (lines 408-424 in `api/app/main.py`) is **not being executed**. This could be due to:
1. `compute_shap` parameter not reaching `generate_predictions()` correctly
2. `shap_values_instance` being reset to None before storage attempt
3. Database session management issue preventing `db.add()` execution
4. Silent exception occurring before logging statements

**Code Location:** `api/app/main.py`, lines 408-424:
```python
if compute_shap and shap_values_instance is not None:
    try:
        logger.error(f"🔴 ATTEMPTING SHAP SAVE...")  # Not appearing in logs!
        new_shap_analysis = ShapAnalysis(...)
        db.add(new_shap_analysis)
    except Exception as e:
        logger.error(f"🔴 EXCEPTION: {e}")  # Not appearing in logs!
```

### User ID Header Bug
**Symptom:** X-User-ID header values not being stored  
**Evidence:**
- Test script sent headers: `{"X-User-ID": "test_user_1"}` through `{"X-User-ID": "test_user_5"}`
- All database records show `user_id = "demo1"` (default value)

**Code Location:** `api/app/main.py`, line 344:
```python
user_id = request.headers.get('X-User-ID', 'demo1')
```

**Hypothesis:**
- Headers may not be passed correctly through Docker network
- FastAPI Request object not receiving custom headers
- Header name case sensitivity issue ("X-User-ID" vs "x-user-id")

## Test Environment

- **API**: FastAPI container (hrattritionrate-fastapi_app-1), port 8001
- **Database**: PostgreSQL container (hrattritionrate-db-1), port 5432
- **UI**: Streamlit container (hrattritionrate-ui-1), port 8501
- **Test Method**: Direct HTTP POST to `/predict_report` endpoint via httpx
- **Python Version**: 3.13
- **Docker Compose**: Running with all services healthy

## Recommendations

1. **Immediate Fix Required:** Debug why SHAP storage code block doesn't execute
   - Add logging before `if not _is_db_disabled()` block
   - Verify `compute_shap` parameter value at storage point
   - Check if database session is valid when `db.add()` is called

2. **Header Fix:** Investigate HTTP header transmission
   - Test with lowercase header name: "x-user-id"
   - Verify headers in FastAPI middleware
   - Add logging to confirm header reception

3. **Testing:** After fixes, re-run test with:
   ```bash
   poetry run python test_ui_manual.py
   ```

## Test Script

The test script `test_ui_manual.py` successfully:
- Loaded 5 diverse employees from training CSVs
- Sent properly formatted API requests
- Verified database records via SQL query
- Provided comprehensive output for debugging

**Test Duration:** ~10 seconds total for 5 predictions  
**Success Rate:** 100% for predictions, 0% for SHAP storage  
**Data Quality:** All predictions reasonable given employee profiles
