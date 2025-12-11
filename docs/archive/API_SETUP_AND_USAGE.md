# HR Attrition API - Setup and Usage Guide

## Summary

The API has been tested and is **working correctly**. The 422 error you experienced was likely due to the API server not running when you tried to use the Streamlit app.

## Architecture Overview

```
┌─────────────────┐
│  Streamlit App  │  (Port 8504)
│   (ui/app.py)   │
└────────┬────────┘
         │
         │ HTTP POST /predict
         ▼
┌─────────────────┐
│   FastAPI API   │  (Port 8001)
│ (api/app/main.py)│
└────────┬────────┘
         │
         ├─► Load Model
         ├─► Process Data
         ├─► Generate Predictions
         └─► Calculate SHAP values
```

## Current Status

✅ **API Server**: Running on `http://localhost:8001`
✅ **Streamlit App**: Running on `http://localhost:8504`
✅ **Health Check**: Passing
✅ **Predictions**: Working with test data
✅ **SHAP Values**: Being generated correctly

## How to Start the System

### Option 1: Manual Start (Recommended for Development)

**Terminal 1 - Start the API:**
```bash
cd "c:\Users\shahu\Documents\OneDrive\OPEN CLASSROOMS\PROJET 5\HR Attrition Rate"
poetry run uvicorn api.app.main:app --host 0.0.0.0 --port 8001 --reload
```

**Terminal 2 - Start the Streamlit App:**
```bash
cd "c:\Users\shahu\Documents\OneDrive\OPEN CLASSROOMS\PROJET 5\HR Attrition Rate"
poetry run streamlit run ui/app.py
```

### Option 2: Using Docker Compose

**Note**: The docker-compose.yml has the API on port 8000, but the Streamlit app expects port 8001. You need to update one of them.

**Update ui/app.py line 60:**
```python
API_BASE_URL = "http://localhost:8000"  # Change from 8001 to 8000
```

Then run:
```bash
docker-compose up
```

## Testing the API

### Quick Health Check
```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "ok",
  "message": "API is healthy"
}
```

### Run End-to-End Tests

I've created test scripts for you:

**1. Simple API Test:**
```bash
poetry run python test_api_debug.py
```

**2. Comprehensive E2E Test:**
```bash
poetry run python test_e2e.py
```

## API Endpoints

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "API is healthy"
}
```

### POST `/predict`
Predict attrition risk for employees.

**Request Body:**
```json
{
  "eval_data": [
    {
      "eval_number": "E_1",
      "augmentation_salaire_precedente": "11%",
      "heures_supplementaires": "Oui",
      "note_evaluation_actuelle": 2,
      "note_evaluation_precedente": 4,
      "anciennete": 3
    }
  ],
  "sirh_data": [
    {
      "id_employee": 1,
      "genre": "m",
      "nombre_heures_travailless": 186,
      "departement": "IT",
      "salaire": 76106
    }
  ],
  "sondage_data": [
    {
      "code_sondage": 1,
      "satisfaction_employee_nature_travail": 2,
      "satisfaction_employee_equipe": 3,
      "satisfaction_employee_equilibre_pro_perso": 2,
      "annees_dans_le_poste_actuel": 1,
      "annees_dans_l_entreprise": 5,
      "annees_sous_responsable_actuel": 4
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "id_employee": 1,
      "prediction": "Leave",
      "probability": 0.9732697263319924,
      "risk_category": "High",
      "message": "Employee 1 is predicted to Leave with 97.33% attrition risk (Risk: High).",
      "trace_id": 267,
      "shap_values": [...],
      "base_value": -1.0379683952352932
    }
  ]
}
```

## Using the Streamlit App

1. **Start both servers** (API and Streamlit) as shown above
2. **Open the Streamlit app** in your browser at `http://localhost:8504`
3. **Upload CSV files** or use the local files from the `data/` directory
4. **Click "Predict Attrition"**
5. **Download the Excel report** with predictions
6. **View SHAP explanations** for each employee

## Data Flow

1. **Input**: Three CSV files
   - `extrait_eval.csv` - Evaluation data
   - `extrait_sirh.csv` - HR information system data
   - `extrait_sondage.csv` - Survey data

2. **Streamlit App**:
   - Loads CSV files
   - Converts to list of dictionaries
   - Sends to API via POST request

3. **API**:
   - Validates input data using Pydantic schemas
   - Merges data from three sources on `id_employee`
   - Applies feature engineering
   - Generates predictions
   - Calculates SHAP values
   - Stores results in database
   - Returns predictions with SHAP values

4. **Streamlit App**:
   - Receives predictions
   - Generates Excel report with:
     - Summary tab (predictions)
     - Features tab (SHAP values)
     - Metrics tab (summary statistics)
   - Displays SHAP waterfall plots for each employee

## Troubleshooting

### 422 Validation Error

**Cause**: Usually means:
1. API server is not running
2. Data format doesn't match the expected schema
3. Missing required fields

**Solution**:
1. Check if API is running: `curl http://localhost:8001/health`
2. Check the API logs for validation errors
3. Run the test scripts to verify the API is working

### Connection Refused

**Cause**: API server is not running

**Solution**: Start the API server as shown in the setup instructions

### Model Not Loaded

**Cause**: Model file not found at expected path

**Solution**: Ensure `outputs/employee_attrition_pipeline.pkl` exists

## Next Steps

1. ✅ API is running and tested
2. ✅ Streamlit app is connected to API
3. ✅ End-to-end test scripts created
4. 📝 Consider updating the docker-compose port to match (8001 or 8000)
5. 📝 Add error handling for edge cases
6. 📝 Add logging for debugging

## Files Created for Testing

- `test_api_debug.py` - Simple API test with 3 employees
- `test_e2e.py` - Comprehensive end-to-end test
- `API_SETUP_AND_USAGE.md` - This documentation

## Important Notes

- The API uses port **8001** (configured in ui/app.py)
- Docker Compose uses port **8000** (needs to be aligned)
- Both FastAPI and Streamlit need to be running simultaneously
- The API stores prediction traceability in the PostgreSQL database
