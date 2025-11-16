# Employee Attrition Prediction API

FastAPI backend for predicting employee attrition risk using a machine learning model with SHAP explanations.

## Features

- **Prediction Endpoint**: `POST /predict` - Get attrition risk predictions for employees
- **Health Check**: `GET /health` - API health status
- **Root Info**: `GET /` - Basic API information
- **Auto Documentation**: Interactive docs at `/docs` (Swagger UI) and `/redoc` (ReDoc)
- **Data Validation**: Pydantic models ensure data integrity
- **Database Logging**: All predictions stored in PostgreSQL for traceability
- **SHAP Values**: Feature importance explanations included in predictions

## Quick Start

### Prerequisites

- Python 3.12+
- Poetry (dependency management)
- PostgreSQL (for prediction logging)

### Running the API

**Option 1: Using startup script**
```bash
./scripts/start-api.sh  # Linux/Mac
scripts\start-api.bat   # Windows
```

**Option 2: Manual start**
```bash
poetry run uvicorn api.app.main:app --host 0.0.0.0 --port 8001 --reload
```

**Option 3: Docker**
```bash
docker-compose up fastapi_app
```

The API will be available at: http://localhost:8001

## API Documentation

Once running, access interactive documentation:

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## Endpoints

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "API is healthy"
}
```

### `POST /predict`

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
      "probability": 0.97,
      "risk_category": "High",
      "message": "Employee 1 is predicted to Leave with 97.00% attrition risk (Risk: High).",
      "trace_id": 267,
      "shap_values": [-0.015, 0.234, ...],
      "base_value": -1.038,
      "feature_names": ["feature1", "feature2", ...]
    }
  ]
}
```

## Configuration

Set via environment variables (see `.env.example`):

- `API_PORT`: Port number (default: 8001)
- `API_HOST`: Host address (default: 0.0.0.0)
- `DATABASE_URL`: PostgreSQL connection string

## Architecture

```
Client → FastAPI (/predict) → Data Processing → ML Model → SHAP → PostgreSQL
                                                    ↓
                                            Return Predictions
```

The API:
1. Receives raw employee data (3 separate datasets)
2. Merges and engineers features
3. Makes predictions using trained model
4. Calculates SHAP explanations
5. Stores everything in database
6. Returns predictions with SHAP values

## Database Tables

All predictions are logged in PostgreSQL:

- `employees` - Employee master data
- `model_inputs` - Raw input features for each prediction
- `model_outputs` - Prediction results
- `predictions_traceability` - Links inputs to outputs with metadata

## Development

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest api/tests/

# Run with auto-reload
poetry run uvicorn api.app.main:app --reload --port 8001

# Check code quality
poetry run ruff check api/
```

## See Also

- [QUICKSTART.md](../QUICKSTART.md) - Fast setup guide
- [DEVELOPMENT.md](../DEVELOPMENT.md) - Development workflow
- [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) - System architecture
