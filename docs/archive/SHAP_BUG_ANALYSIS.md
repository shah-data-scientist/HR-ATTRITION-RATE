# SHAP Storage Bug - Root Cause Analysis

## Executive Summary
**STATUS:** Bug identified and partially working  
**SHAP Storage:** Working for FIRST prediction only, fails for all subsequent predictions

## Key Findings

### 1. SHAP Storage Works Once
- **Evidence:** Database query shows trace_id 22 HAS SHAP data
- **First prediction:** Employee ID 1, trace_id 22, SHAP saved successfully ✅
- **All subsequent:** Trace_ids 23-33 (11 predictions), NO SHAP data ❌

```sql
SELECT t.trace_id, e.id_employee, 
       CASE WHEN s.shap_id IS NOT NULL THEN 'YES' ELSE 'NO' END as has_shap
FROM predictions_traceability t
LEFT JOIN shap_analysis s ON t.trace_id=s.trace_id
WHERE t.trace_id >= 22;

Result:
 trace_id | id_employee | has_shap
----------+-------------+----------
       22 |           1 | YES      ← ONLY ONE WITH SHAP!
       23 |           2 | NO
       24 |           4 | NO
       25 |           5 | NO
       26 |           7 | NO
       27 |           1 | NO
       ...
       33 |           1 | NO
```

### 2. Code Changes Not Applied Without Rebuild
**Critical Discovery:** Docker image does NOT have volume mount for `/api` code

- Changed files in `api/app/main.py` → Changes ignored by running container
- Must rebuild image: `docker-compose build fastapi_app`
- **Impact:** All debugging attempts (logging, file writes) were using OLD code
- **Time Lost:** ~2 hours debugging with stale code

**Solution Applied:** Identified need to rebuild Docker images for code changes

### 3. SHAP Computation vs Storage
**SHAP computation:** ✅ Working perfectly (API returns 66 features every time)  
**SHAP storage:** ❌ Fails after first success

**Code Flow:**
```python
# Lines 328-334: SHAP computation (WORKS)
if compute_shap:
    shap_values_instance = explainer.shap_values(...)  # ✅ Returns 66 values
    base_value_instance = float(explainer.expected_value)  # ✅ Returns -1.0379
    feature_names_for_instance = list(...)  # ✅ Returns 66 names

# Lines 408-420: SHAP storage (FAILS after first)
if compute_shap and shap_values_instance is not None:
    new_shap_analysis = ShapAnalysis(...)
    db.add(new_shap_analysis)  # ❌ Only works once
db.commit()
```

### 4. Possible Root Causes

#### Hypothesis A: Database Session Issue
- First prediction in app lifecycle: Session fresh, SHAP saves
- Subsequent predictions: Session state corrupted/not flushed
- **Test:** Check if `db.flush()` before SHAP add would help

#### Hypothesis B: Silent Exception After First Success
- Exception occurs in SHAP save block
- Try-except at line 495 catches it: `db.rollback()` called
- But API returns 200 OK (prediction succeeded, just SHAP failed)
- **Test:** Add explicit try-except around SHAP save only

#### Hypothesis C: Unique Constraint Violation
- **Unlikely:** trace_id is unique per prediction
- Each prediction gets new trace_id (22, 23, 24...)
- Table constraint: `UNIQUE (trace_id)`
- **Counter-evidence:** Should raise IntegrityError, not silent fail

#### Hypothesis D: Object Identity Issue
- ShapAnalysis object stays in session after commit
- SQLAlchemy trying to re-use same object
- **Test:** Call `db.expunge_all()` or create new object each time

### 5. Test Results Summary

**Manual UI Test (5 employees):**
- Employee 1: Prediction ✅, SHAP ✅ (trace_id 22)
- Employee 2: Prediction ✅, SHAP ❌ (trace_id 23)
- Employee 3: Prediction ✅, SHAP ❌ (trace_id 24)
- Employee 4: Prediction ✅, SHAP ❌ (trace_id 25)
- Employee 5: Prediction ✅, SHAP ❌ (trace_id 26)

**Quick Tests (7 attempts):**
- All predictions successful (trace_ids 27-33)
- All SHAP storage failed
- All using same employee (ID 1)

**Pattern:** Single-employee predictions, processed sequentially. First one works, rest fail.

## Recommended Fix Strategy

### Option 1: Add Explicit Flush (SIMPLE)
```python
# After line 407: db.flush()
if compute_shap and shap_values_instance is not None:
    new_shap_analysis = ShapAnalysis(
        trace_id=new_trace.trace_id,
        shap_values=shap_values_instance.tolist(),
        base_value=float(base_value_instance),
        feature_names=feature_names_for_instance,
        created_at=datetime.now(),
    )
    db.add(new_shap_analysis)
    db.flush()  # ← ADD THIS

db.commit()
```

### Option 2: Separate SHAP Transaction (ROBUST)
```python
if compute_shap and shap_values_instance is not None:
    try:
        new_shap_analysis = ShapAnalysis(...)
        db.add(new_shap_analysis)
        db.flush()
    except Exception as e:
        logger.error(f"SHAP save failed: {e}")
        # Don't fail the whole prediction
```

### Option 3: Check for Existing SHAP (DEFENSIVE)
```python
if compute_shap and shap_values_instance is not None:
    # Check if SHAP already exists for this trace_id
    existing = db.query(ShapAnalysis).filter(
        ShapAnalysis.trace_id == new_trace.trace_id
    ).first()
    
    if not existing:
        new_shap_analysis = ShapAnalysis(...)
        db.add(new_shap_analysis)
```

## Next Steps

1. **Revert debugging code** - Clean up all debug statements added
2. **Rebuild Docker image** - Apply clean fix
3. **Test with single employee** - Verify first prediction saves SHAP
4. **Test with 5 employees** - Verify all 5 save SHAP
5. **Check database** - Query shap_analysis table for all 5 records

## Files Modified During Investigation
- `api/app/main.py` - Multiple debugging attempts (REVERT NEEDED)
- `test_ui_manual.py` - ✅ Working test script (KEEP)
- `quick_test.py` - ✅ Single employee test (KEEP)
- `MANUAL_UI_TEST_RESULTS.md` - ✅ Test documentation (KEEP)

## Lesson Learned
**Always check if code changes require Docker image rebuild!**  
Volume mounts are NOT configured for API code in this project.
