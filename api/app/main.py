import json  # For JSON column
import logging  # Add logging import
import os
from contextlib import asynccontextmanager
from datetime import datetime  # For timestamps

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status, Request
from sqlalchemy.orm import Session  # Explicitly import Session for type hinting

from api.app.schemas import (
    BatchPredictionInput,
    BatchPredictionOutput,
    PredictionOutput,
)
from api.security import get_api_key, API_TOKEN

# Database imports
from database.database import get_db
from database.models import Employee, ModelInput, ModelOutput, PredictionTraceability

logger = logging.getLogger("uvicorn.error")

from core.data_processing import clean_and_engineer_features
from core.preprocess import enforce_schema
from core.validation import NUMERIC_COLS, CATEGORICAL_COLS

# --- Configuration ---
# Define risk categories for Excel report and HTML visualization
RISK_THRESHOLDS = {"Low": (0.0, 0.3), "Medium": (0.3, 0.7), "High": (0.7, 1.0)}

# --- Global Model and Preprocessor ---
model = None
expected_model_columns = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model, expected_model_columns
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "outputs",
        "employee_attrition_pipeline.pkl",
    )
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model file not found at {model_path}. Please ensure the model is trained and saved."
        )

    model = joblib.load(model_path)
    expected_model_columns = get_expected_columns_from_pipeline(model)
    print(f"Model loaded successfully from {model_path}")
    print(f"Expected model columns: {expected_model_columns}")

    # --- DEBUG: Inspect ColumnTransformer's internal column lists ---
    if "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        for name, transformer, columns in preprocessor.transformers_:
            if name == "num":
                print(f"Preprocessor Numeric Columns (from model): {columns}")
            elif name == "cat":
                print(f"Preprocessor Categorical Columns (from model): {columns}")
    # --- END DEBUG ---

    yield
    # Clean up (optional)
    print("FastAPI app shutting down.")


app = FastAPI(
    title="Employee Attrition Prediction API",
    description="API for predicting employee attrition risk based on various features.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_expected_columns_from_pipeline(pipeline):
    """Gets the list of columns the model was trained on."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.feature_names_in_)


def get_risk_category(probability: float, threshold: float = 0.5) -> str:
    """Categorizes attrition risk based on the difference between probability and a dynamic threshold,
    with a minimum absolute probability for "Medium" risk.
    - High Risk: probability >= threshold + buffer
    - Low Risk: probability < threshold - buffer
    - Medium Risk: (threshold - buffer <= probability < threshold + buffer) AND (probability >= min_medium_prob)
    - Otherwise: Low Risk
    """
    buffer = 0.05  # How close to the threshold to be considered "Medium" (e.g., 5 percentage points)
    min_medium_prob = (
        0.20  # Minimum absolute probability to be considered "Medium" risk (e.g., 20%)
    )

    if probability >= threshold + buffer:  # Clearly above threshold
        return "High"
    if probability < threshold - buffer:  # Clearly below threshold
        return "Low"
    if (
        probability >= min_medium_prob
    ):  # Close to threshold AND above minimum for Medium
        return "Medium"
    # Close to threshold but below min_medium_prob
    return "Low"  # Default to Low if not High, not clearly Low, and below min_medium_prob


@app.get("/", summary="Root endpoint", response_model=dict[str, str])
async def read_root():
    """Provides basic information about the API.
    """
    return {
        "message": "Welcome to the Employee Attrition Prediction API!",
        "version": app.version,
        "documentation_url": "/docs",
    }


@app.get("/token", summary="Get a temporary API token", response_model=dict[str, str])
async def get_token():
    """Provides a temporary API token for testing purposes.
    """
    return {"token": API_TOKEN}


@app.get("/health", summary="Health check endpoint", response_model=dict[str, str])
async def health_check():
    """
    Health check endpoint to verify API status.
    Returns a simple message indicating the API is healthy.
    """
    return {"status": "ok", "message": "API is healthy"}


@app.post(
    "/predict",
    response_model=BatchPredictionOutput,
    summary="Predict attrition risk for a batch of employees",
    dependencies=[Depends(get_api_key)],
)
async def predict_attrition(
    batch_input: BatchPredictionInput, request: Request, db: Session = Depends(get_db)
):
    """Predicts the attrition risk for a list of employees based on their features.
    All model inputs, outputs, and prediction traceability are recorded in the database.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet.",
        )

    predictions_output: list[PredictionOutput] = []

    # Convert list of Pydantic models to a list of dictionaries
    employees_data_list = [employee.model_dump() for employee in batch_input.employees]

    # Convert to DataFrame for processing
    input_df = pd.DataFrame(employees_data_list)

    # Apply feature engineering
    processed_data = clean_and_engineer_features(input_df.copy())

    # ✨ enforce schema and coerce types
    feature_order = NUMERIC_COLS + CATEGORICAL_COLS
    data_for_prediction = enforce_schema(processed_data, feature_order)

    try:
        # Make predictions
        prediction_proba_raw = model.predict_proba(data_for_prediction)[:, 1]
        predictions_binary = (prediction_proba_raw >= 0.5).astype(
            int
        )  # Using default threshold 0.5 for API

        # Calculate log-odds for SHAP
        # log_odds = np.log(prediction_proba_raw / (1 - prediction_proba_raw))
        # Handle cases where probability is 0 or 1 to avoid log(0) or division by zero
        log_odds = np.log(
            np.clip(prediction_proba_raw, 1e-10, 1 - 1e-10)
            / (1 - np.clip(prediction_proba_raw, 1e-10, 1 - 1e-10))
        )

        for i, employee_input_data in enumerate(batch_input.employees):
            employee_id = int(input_df.loc[i, "id_employee"])
            prob = prediction_proba_raw[i]
            pred_label = "Leave" if predictions_binary[i] == 1 else "Stay"
            risk_cat = get_risk_category(
                prob, threshold=0.5
            )  # Using default threshold 0.5 for API
            current_log_odds = log_odds[i]

            # 1. Fetch or Create Employee Record
            employee_db = (
                db.query(Employee).filter(Employee.id_employee == employee_id).first()
            )
            if not employee_db:
                # If employee doesn't exist, create a new one with available data
                employee_data_for_db = {
                    k: v
                    for k, v in employee_input_data.model_dump(
                        exclude_unset=True
                    ).items()
                    if k != "id_employee"  # id_employee is handled separately
                }
                employee_db = Employee(
                    id_employee=employee_id,
                    **employee_data_for_db,
                    date_ingestion=datetime.now(),
                )
                db.add(employee_db)
                db.flush()  # Flush to ensure employee_db gets an ID if newly created

            # 2. Record Model Input
            # Use the features that were actually sent to the model (after preprocessing/engineering)
            # Convert processed_data row to a dictionary for JSON storage
            features_for_db = (
                processed_data.loc[i].drop("id_employee", errors="ignore").to_dict()
            )
            new_model_input = ModelInput(
                id_employee=employee_id,
                features=json.dumps(features_for_db),
                prediction_timestamp=datetime.now(),
            )
            db.add(new_model_input)
            db.flush()  # Flush to get input_id

            # 3. Record Model Output
            new_model_output = ModelOutput(
                prediction_proba=float(prob),
                risk_category=risk_cat,
                prediction_label=pred_label,
                log_odds=float(current_log_odds),
                prediction_timestamp=datetime.now(),
            )
            db.add(new_model_output)
            db.flush()  # Flush to get output_id

            # 4. Record Traceability
            new_trace = PredictionTraceability(
                input_id=new_model_input.input_id,
                output_id=new_model_output.output_id,
                model_version=app.version,
                prediction_source="API",
                request_metadata={
                    "user_agent": request.headers.get("user-agent"),
                    "client_host": request.client.host,
                },
                created_at=datetime.now(),
            )
            db.add(new_trace)
            db.commit()  # Commit all changes for this prediction

            # Refresh objects to get latest state, especially IDs
            db.refresh(new_model_input)
            db.refresh(new_model_output)
            db.refresh(new_trace)

            predictions_output.append(
                PredictionOutput(
                    id_employee=employee_id,
                    prediction=pred_label,
                    probability=float(prob),
                    risk_category=risk_cat,
                    message=f"Employee {employee_id} is predicted to {pred_label} with {prob:.2%} attrition risk (Risk: {risk_cat}).",
                    trace_id=new_trace.trace_id,
                )
            )
    except Exception as e:
        db.rollback()  # Rollback in case of error
        logger.exception(
            "Prediction failed with an unexpected error."
        )  # Log the full exception traceback
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )

    return BatchPredictionOutput(predictions=predictions_output)
