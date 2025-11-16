import json  # For JSON column
import logging  # Add logging import
import os
from contextlib import asynccontextmanager
from datetime import datetime  # For timestamps

import joblib
import numpy as np
import pandas as pd
import shap # Import shap
from fastapi import Depends, FastAPI, HTTPException, status, Request
from sqlalchemy.orm import Session  # Explicitly import Session for type hinting

# from api.security import get_api_key, API_TOKEN

# Database imports
from database.database import get_db
from database.models import Employee, ModelInput, ModelOutput, PredictionTraceability

# Core imports - using canonical schemas and processing
from core.schema import (
    ProcessedBatchPredictionInput, # Renamed from BatchPredictionInput
    BatchPredictionOutput,
    PredictionOutput,
    EvalInputSchema,
    SirhInputSchema,
    SondageInputSchema,
    RawBatchPredictionInput,
)
from core.data_processing import clean_and_engineer_features
from core.preprocess import enforce_schema
from core.validation import NUMERIC_COLS, CATEGORICAL_COLS

logger = logging.getLogger("uvicorn.error")

# --- Configuration ---
# Define risk categories for Excel report and HTML visualization
RISK_THRESHOLDS = {"Low": (0.0, 0.3), "Medium": (0.3, 0.7), "High": (0.7, 1.0)}

# --- Global Model and Preprocessor ---
model = None
expected_model_columns = None
explainer = None
expected_model_columns_for_shap = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model, expected_model_columns, explainer, expected_model_columns_for_shap
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

    # Load X_train for SHAP explainer
    x_train_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "outputs", "X_train.parquet"
    )
    if not os.path.exists(x_train_path):
        raise RuntimeError(
            f"X_train file not found at {x_train_path}. SHAP explainer cannot be initialized."
        )
    x_train_for_shap = pd.read_parquet(x_train_path)

    # Initialize SHAP explainer
    # Assuming the model is a pipeline with a 'preprocessor' and a 'model' step
    preprocessor = model.named_steps["preprocessor"]
    ml_model = model.named_steps["model"]

    # Transform x_train_for_shap to get the data in the format the ML model expects
    x_train_transformed = preprocessor.transform(x_train_for_shap)
    expected_model_columns_for_shap = preprocessor.get_feature_names_out()

    # Ensure x_train_transformed is a DataFrame for explainer
    if not isinstance(x_train_transformed, pd.DataFrame):
        x_train_transformed = pd.DataFrame(
            x_train_transformed, columns=expected_model_columns_for_shap
        )

    explainer = shap.LinearExplainer(ml_model, x_train_transformed)
    print("SHAP Explainer initialized successfully.")

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
)
async def predict_attrition(
    batch_input: RawBatchPredictionInput, request: Request, db: Session = Depends(get_db)
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

    # Convert raw input data to DataFrames
    eval_df = pd.DataFrame([e.model_dump() for e in batch_input.eval_data])
    sirh_df = pd.DataFrame([s.model_dump() for s in batch_input.sirh_data])
    sondage_df = pd.DataFrame([s.model_dump() for s in batch_input.sondage_data])

    # --- Data Merging Logic ---
    # 1. Create id_employee for eval_df from eval_number
    eval_df["id_employee"] = eval_df["eval_number"].str.extract(r'E_(\d+)').astype(int)
    print(f"eval_df id_employee dtype: {eval_df['id_employee'].dtype}")
    print(f"eval_df id_employee null count: {eval_df['id_employee'].isnull().sum()}")

    # 2. Merge sirh_df and sondage_df
    # Rename 'code_sondage' in sondage_df to 'id_employee' for merging
    sondage_df_renamed = sondage_df.rename(columns={"code_sondage": "id_employee"})
    merged_sirh_sondage = sirh_df.merge(sondage_df_renamed, on="id_employee", how="outer")
    print(f"merged_sirh_sondage id_employee dtype: {merged_sirh_sondage['id_employee'].dtype}")
    print(f"merged_sirh_sondage id_employee null count: {merged_sirh_sondage['id_employee'].isnull().sum()}")

    # 3. Merge with eval_df
    # Use a full outer merge to ensure all employees are kept, handling potential missing data
    input_df = merged_sirh_sondage.merge(eval_df, on="id_employee", how="outer", suffixes=('_sirh_sondage', '_eval'))
    print(f"input_df id_employee dtype after all merges: {input_df['id_employee'].dtype}")
    print(f"input_df id_employee null count after all merges: {input_df['id_employee'].isnull().sum()}")
    print(f"input_df id_employee head after all merges:\n{input_df['id_employee'].head()}")

    # Drop duplicates if any (e.g., if an id_employee appears multiple times across sources)
    input_df.drop_duplicates(subset=["id_employee"], inplace=True)

    # Ensure id_employee is integer type
    input_df["id_employee"] = input_df["id_employee"].astype(int)

    # Apply feature engineering
    processed_data = clean_and_engineer_features(input_df.copy())

    # ✨ enforce schema and coerce types
    feature_order = NUMERIC_COLS + CATEGORICAL_COLS
    data_for_prediction = enforce_schema(processed_data, feature_order)

    print("Data for prediction dtypes:")
    print(data_for_prediction.dtypes)
    print("Data for prediction head:")
    print(data_for_prediction.head())

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

        for i, employee_input_data_row in input_df.iterrows(): # Iterate through the merged input_df
            employee_id = int(employee_input_data_row["id_employee"])
            prob = prediction_proba_raw[i]
            pred_label = "Leave" if predictions_binary[i] == 1 else "Stay"
            risk_cat = get_risk_category(
                prob, threshold=0.5
            )  # Using default threshold 0.5 for API
            current_log_odds = log_odds[i]

            # Calculate SHAP values for the current employee
            # Ensure data_for_prediction is a DataFrame for consistent indexing
            if not isinstance(data_for_prediction, pd.DataFrame):
                data_for_prediction = pd.DataFrame(
                    data_for_prediction, columns=expected_model_columns
                )

            # Transform the single row for SHAP explanation
            single_employee_data_transformed = model.named_steps["preprocessor"].transform(
                data_for_prediction.iloc[[i]]
            )
            
            # Ensure it's a DataFrame for SHAP explainer
            if not isinstance(single_employee_data_transformed, pd.DataFrame):
                single_employee_data_transformed = pd.DataFrame(
                    single_employee_data_transformed, columns=expected_model_columns_for_shap
                )

            shap_values_instance = explainer.shap_values(single_employee_data_transformed)[0]
            base_value_instance = explainer.expected_value

            # 1. Fetch or Create/Update Employee Record
            # Get the cleaned data from processed_data DataFrame (row i)
            cleaned_employee_data = processed_data.loc[i].to_dict()

            # Remove id_employee as it's handled separately
            employee_data_for_db = {
                k: v for k, v in cleaned_employee_data.items()
                if k != "id_employee" and k in [col.name for col in Employee.__table__.columns]
            }

            employee_db = (
                db.query(Employee).filter(Employee.id_employee == employee_id).first()
            )

            if not employee_db:
                # Employee doesn't exist - CREATE new record
                employee_db = Employee(
                    id_employee=employee_id,
                    **employee_data_for_db,
                    date_ingestion=datetime.now(),
                )
                db.add(employee_db)
            else:
                # Employee exists - UPDATE with latest data (Option A: Latest Snapshot)
                for key, value in employee_data_for_db.items():
                    setattr(employee_db, key, value)
                # Update ingestion timestamp to reflect latest update
                employee_db.date_ingestion = datetime.now()

            db.flush()  # Flush to ensure employee_db is persisted

            # 2. Record Model Input
            # Store the RAW input data for audit/traceability (what the user sent)
            # Need to reconstruct the raw input for this specific employee from the original batch_input
            # This is complex as the raw inputs are separate. For now, we'll store the merged input_df row.
            # A more robust solution would involve linking raw inputs to the merged employee_id.
            raw_features_for_db = input_df.loc[i].to_dict() # Store the merged raw input for now
            new_model_input = ModelInput(
                id_employee=employee_id,
                features=json.dumps(raw_features_for_db),
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
                threshold=0.5, # Pass the threshold value
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
                    shap_values=shap_values_instance.tolist(),  # Convert numpy array to list
                    base_value=base_value_instance,
                    feature_names=list(expected_model_columns_for_shap),  # Add feature names
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
