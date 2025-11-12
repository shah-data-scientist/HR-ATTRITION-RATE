# UI Testing Guide

## What Was Fixed

### 1. ✅ Threshold Control Enhancement
**Location**: `app_improved.py` lines 70-103

**Changes Made:**
- Threshold slider now in a **prominent boxed container** with:
  - Green border highlight
  - Light gray background
  - Bold title with icon
  - Visual metrics showing Low/Medium/High risk ranges
- Maintains keyboard accessibility
- Clear visual feedback

**Before**: Simple horizontal slider
**After**: Boxed, highlighted control with visual risk indicators

---

### 2. ✅ Confusion Matrix (Already Correct!)
**Location**: `app_improved.py` lines 106-166

**Confirmed Implementation:**
- ✅ **Row-percentages**: Each row sums to 100% (lines 116-120)
- ✅ **Responsive**: Uses `st.pyplot(fig, use_container_width=True)` (line 153)
- ✅ **Reduced figure size**: `figsize=(5, 4)` (line 133)
- ✅ **Clean layout**: Bordered container with proper styling

The existing confusion matrix implementation was already correct per requirements!

---

### 3. ✅ UI→API Integration
**Location**: `app_improved.py` lines 169-202, `ui_api_client.py`

**Changes Made:**
- **NO local model inference** - removed all `model.predict_proba()` calls
- UI now calls FastAPI `/predict` endpoint via `httpx`
- API client module (`ui_api_client.py`) provides:
  - `call_prediction_api()` - Send data to API
  - `check_api_health()` - Verify API is running
  - `extract_probabilities_from_api_response()` - Parse results
- Uses environment variables for configuration:
  - `API_URL` - API endpoint (default: http://localhost:8000)
  - `API_TOKEN` - Authentication token

**Before**: `predictions_proba = st.session_state.model.predict_proba(data)[:, 1]`
**After**: `result = call_prediction_api(employee_data)`

---

## How to Test

### Step 1: Start the API Server
```bash
# Terminal 1: Start the API
poetry run uvicorn api.app.main:app --host 0.0.0.0 --port 8000
```

**Verify API is running:**
- Open http://localhost:8000/docs in your browser
- You should see the Swagger UI

---

### Step 2: Test the API Client
```bash
# Terminal 2: Test API client standalone
poetry run python ui_api_client.py
```

**Expected output:**
```
Checking API at http://localhost:8000...
✓ API is healthy!

Calling prediction API...
✓ Prediction successful!
  Employee 99999:
    Prediction: Stay
    Probability: 35.50%
    Risk: Low
```

---

### Step 3: Launch the Improved UI
```bash
# Terminal 2 (or new terminal): Start Streamlit
poetry run streamlit run app_improved.py
```

**The UI should open in your browser at http://localhost:8501**

---

### Step 4: Test the UI Features

#### A. Test Threshold Control
1. Look for the **green-bordered box** labeled "High Risk Threshold Control"
2. **Move the slider** - observe the risk metrics update (Low/Medium/High ranges)
3. **Verify keyboard accessibility**:
   - Click the slider
   - Use arrow keys ←→ to adjust
   - Should work smoothly

#### B. Test Predictions via API
1. Click **"Load Sample Data"** button
2. Verify employee data loads in the preview table
3. Click **"Predict Attrition"** button
4. **Watch for**:
   - Spinner: "Calling prediction API..."
   - Success message with count
   - Results table appears with employee predictions
   - Risk distribution metrics (Low/Medium/High counts)

#### C. Test Confusion Matrix (if you add training data)
If you extend the UI to show confusion matrix:
1. Should display in a bordered container
2. Percentages should be **row-normalized** (each row sums to 100%)
3. Should be **responsive** (resizes with window)
4. Figure size should be reasonable (not too large)

---

## Common Issues & Solutions

### Issue 1: API Connection Error
**Symptom**: "Connection Error: ..." in UI

**Solution**:
```bash
# Check if API is running
curl http://localhost:8000/health

# If not running, start it:
poetry run uvicorn api.app.main:app --host 0.0.0.0 --port 8000
```

---

### Issue 2: Authentication Error
**Symptom**: "API Error (401): ..." or "API Error (403): ..."

**Solution**:
```bash
# Check your .env file has the correct token
cat .env | grep API_TOKEN

# Should match what the API expects
# Default: API_TOKEN="your_super_secret_api_token"
```

---

### Issue 3: Model Not Found Error from API
**Symptom**: API returns "Model file not found"

**Solution**:
```bash
# Train the model to create the artifact
poetry run python train.py

# Verify model exists
ls -lh outputs/employee_attrition_pipeline.pkl
```

---

### Issue 4: Sample Data Not Found
**Symptom**: "Sample data files not found in 'data/' directory"

**Solution**:
```bash
# Check data files exist
ls -lh data/*.csv

# Should show:
# data/extrait_eval.csv
# data/extrait_sirh.csv
# data/extrait_sondage.csv
```

---

## Files Created

| File | Purpose |
|------|---------|
| `app_improved.py` | New UI with all fixes (threshold control, API calls) |
| `ui_api_client.py` | API client module for UI→API communication |
| `UI_TESTING_GUIDE.md` | This testing guide |

---

## Comparison: Old vs New

### Old `app.py` (Local Inference)
```python
# Line 206
predictions_proba = st.session_state.model.predict_proba(processed_df_aligned)[:, 1]
```

### New `app_improved.py` (API Calls)
```python
# Lines 290-295
result = call_prediction_api(employee_records)
if result:
    predictions = result["predictions"]
    # Use API predictions instead of local model
```

---

## Next Steps for User Testing

1. **Start both servers** (API and UI)
2. **Test threshold control** - verify boxed design and responsiveness
3. **Test predictions** - click "Load Sample Data" → "Predict Attrition"
4. **Verify API integration** - check that predictions come from API (watch terminal logs)
5. **Report any issues** - note any errors or unexpected behavior

---

## Architecture Flow

```
┌─────────────────┐
│  Streamlit UI   │
│ (app_improved)  │
└────────┬────────┘
         │ httpx.post()
         │ /predict
         ▼
┌─────────────────┐
│   FastAPI       │
│  (api/app)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────┐
│  Core Modules   │───→│  ML Model    │
│ (cleaning, etc) │    │  Pipeline    │
└─────────────────┘    └──────────────┘
         │
         ▼
┌─────────────────┐
│   PostgreSQL    │
│   (logging)     │
└─────────────────┘
```

**Key**: UI never touches the model directly - only through API!

---

## Ready to Test?

Run these commands in order:

```bash
# Terminal 1
poetry run uvicorn api.app.main:app --reload

# Terminal 2
poetry run streamlit run app_improved.py
```

Then follow the testing steps above and report your findings!
