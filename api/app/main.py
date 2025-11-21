import json  # For JSON column
import logging  # Add logging import
import os
from contextlib import asynccontextmanager
from datetime import datetime  # For timestamps
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()  # Load .env
load_dotenv(".env.local", override=True)  # Load .env.local and override

import base64
import io
import zipfile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import shap  # Import shap
from fastapi import Depends, FastAPI, HTTPException, status, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session  # Explicitly import Session for type hinting
from sqlalchemy import text
from typing import Optional

# Authentication and security
from api.auth import get_api_key, get_optional_api_key
from api.middleware import setup_security_middleware

# Database imports
from database.database import get_db
from database.models import (
    Employee,
    ModelInput,
    ModelOutput,
    PredictionTraceability,
    ShapAnalysis,
    Job,
    User,
)

# Core imports - using canonical schemas and processing
from core.schema import (
    ProcessedBatchPredictionInput,  # Renamed from BatchPredictionInput
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


def filter_id_employee_from_shap(shap_values: list, feature_names: list) -> tuple[list, list]:
    """Filter out id_employee from SHAP values and feature names.

    Args:
        shap_values: List of SHAP values for all features
        feature_names: List of feature names corresponding to SHAP values

    Returns:
        Tuple of (filtered_shap_values, filtered_feature_names) with id_employee removed
    """
    if not feature_names or len(feature_names) != len(shap_values):
        return shap_values, feature_names

    # Employee ID variations to filter out
    employee_id_variations = {
        "id_employee",
        "num_id_employee",
        "num__id_employee",  # double underscore variant
        "employee_id",
        "id employee",
        "employeeid",
        "emp_id",
        "empid"
    }

    # Find indices where feature is NOT an employee ID (case-insensitive)
    indices_to_keep = [
        i for i, name in enumerate(feature_names)
        if name.lower().replace("_", "").replace(" ", "") not in
           {var.lower().replace("_", "").replace(" ", "") for var in employee_id_variations}
    ]

    # Filter both lists
    filtered_shap_values = [shap_values[i] for i in indices_to_keep]
    filtered_feature_names = [feature_names[i] for i in indices_to_keep]

    return filtered_shap_values, filtered_feature_names


# Optional local toggle to skip DB writes (useful when Postgres isn't available)
def _is_db_disabled() -> bool:
    return os.getenv("DISABLE_DB", "0") == "1"


# --- Health endpoints ---
def _db_ok(db: Optional[Session]) -> bool:
    try:
        if db is None:
            return False
        db.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# --- Global Model and Preprocessor ---
model = None
expected_model_columns = None
explainer = None
expected_model_columns_for_shap = None


def init_model_for_cli():
    """Initialize model and SHAP explainer when running outside FastAPI lifespan."""
    global model, expected_model_columns, explainer, expected_model_columns_for_shap
    if model is not None and explainer is not None:
        return
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

    x_train_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "outputs", "X_train.parquet"
    )
    if not os.path.exists(x_train_path):
        raise RuntimeError(
            f"X_train file not found at {x_train_path}. SHAP explainer cannot be initialized."
        )
    x_train_for_shap = pd.read_parquet(x_train_path)
    preprocessor = model.named_steps["preprocessor"]
    ml_model = model.named_steps["model"]
    x_train_transformed = preprocessor.transform(x_train_for_shap)
    expected_model_columns_for_shap = preprocessor.get_feature_names_out()
    if not isinstance(x_train_transformed, pd.DataFrame):
        x_train_transformed = pd.DataFrame(
            x_train_transformed, columns=expected_model_columns_for_shap
        )
    explainer = shap.LinearExplainer(ml_model, x_train_transformed)


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
    description="API for predicting employee attrition risk based on various features. Requires API key authentication.",
    version="1.0.0",
    lifespan=lifespan,
)

# Setup security middleware (CORS, security headers, logging, etc.)
setup_security_middleware(app)


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
    return (
        "Low"  # Default to Low if not High, not clearly Low, and below min_medium_prob
    )


def generate_predictions(
    batch_input: RawBatchPredictionInput,
    request: Request,
    db: Optional[Session],
    compute_shap: bool = True,
) -> list[PredictionOutput]:
    """Core prediction routine shared by endpoints. Computes predictions, SHAP values,
    and persists traceability depending on DB toggle.
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
    eval_df["id_employee"] = eval_df["eval_number"].str.extract(r"E_(\d+)").astype(int)

    # 2. Merge sirh_df and sondage_df
    sondage_df_renamed = sondage_df.rename(columns={"code_sondage": "id_employee"})
    merged_sirh_sondage = sirh_df.merge(
        sondage_df_renamed, on="id_employee", how="outer"
    )

    # 3. Merge with eval_df
    input_df = merged_sirh_sondage.merge(
        eval_df, on="id_employee", how="outer", suffixes=("_sirh_sondage", "_eval")
    )

    # Drop duplicates if any and ensure id_employee is int
    input_df.drop_duplicates(subset=["id_employee"], inplace=True)
    input_df["id_employee"] = input_df["id_employee"].astype(int)

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
        )  # default threshold 0.5

        # Calculate log-odds for SHAP
        log_odds = np.log(
            np.clip(prediction_proba_raw, 1e-10, 1 - 1e-10)
            / (1 - np.clip(prediction_proba_raw, 1e-10, 1 - 1e-10))
        )

        for i, employee_input_data_row in input_df.iterrows():
            employee_id = int(employee_input_data_row["id_employee"])
            prob = float(prediction_proba_raw[i])
            pred_label = "Leave" if predictions_binary[i] == 1 else "Stay"
            risk_cat = get_risk_category(prob, threshold=0.5)
            current_log_odds = float(log_odds[i])

            # Ensure data_for_prediction is a DataFrame for consistent indexing
            if not isinstance(data_for_prediction, pd.DataFrame):
                data_for_prediction = pd.DataFrame(
                    data_for_prediction, columns=expected_model_columns
                )

            shap_values_instance = None
            base_value_instance = None
            feature_names_for_instance = None

            if compute_shap:
                # Transform the single row for SHAP explanation
                single_employee_data_transformed = model.named_steps[
                    "preprocessor"
                ].transform(data_for_prediction.iloc[[i]])

                # Ensure it's a DataFrame for SHAP explainer
                if not isinstance(single_employee_data_transformed, pd.DataFrame):
                    single_employee_data_transformed = pd.DataFrame(
                        single_employee_data_transformed,
                        columns=expected_model_columns_for_shap,
                    )

                shap_values_instance = explainer.shap_values(
                    single_employee_data_transformed
                )[0]
                base_value_instance = float(explainer.expected_value)
                feature_names_for_instance = list(expected_model_columns_for_shap)

            if not _is_db_disabled():
                # 1. Fetch or Create/Update Employee Record — store RAW merged row
                raw_employee_data = input_df.loc[i].to_dict()
                employee_data_for_db = {
                    k: v
                    for k, v in raw_employee_data.items()
                    if k != "id_employee"
                    and k in [col.name for col in Employee.__table__.columns]
                }

                # Set user_id from request header (who is making the prediction request)
                user_id = request.headers.get("X-User-ID", "demo1")
                employee_data_for_db["user_id"] = user_id

                employee_db = (
                    db.query(Employee)
                    .filter(Employee.id_employee == employee_id)
                    .first()
                )

                if not employee_db:
                    employee_db = Employee(
                        id_employee=employee_id,
                        **employee_data_for_db,
                        date_ingestion=datetime.now(),
                    )
                    db.add(employee_db)
                else:
                    for key, value in employee_data_for_db.items():
                        setattr(employee_db, key, value)
                    employee_db.date_ingestion = datetime.now()

                db.flush()

                # 2. Record Model Input (store merged raw input)
                raw_features_for_db = input_df.loc[i].to_dict()
                new_model_input = ModelInput(
                    id_employee=employee_id,
                    features=json.dumps(raw_features_for_db),
                    prediction_timestamp=datetime.now(),
                )
                db.add(new_model_input)
                db.flush()

                # 3. Record Model Output
                new_model_output = ModelOutput(
                    prediction_proba=float(prob),
                    risk_category=risk_cat,
                    prediction_label=pred_label,
                    log_odds=current_log_odds,
                    threshold=0.5,
                    prediction_timestamp=datetime.now(),
                )
                db.add(new_model_output)
                db.flush()

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
                db.flush()  # Flush to get trace_id

                # 5. Record SHAP Analysis (if computed)
                if compute_shap and shap_values_instance is not None:
                    try:
                        new_shap_analysis = ShapAnalysis(
                            trace_id=new_trace.trace_id,
                            shap_values=shap_values_instance.tolist(),
                            base_value=float(base_value_instance),
                            feature_names=feature_names_for_instance,
                            created_at=datetime.now(),
                        )
                        db.add(new_shap_analysis)
                        db.flush()  # Flush SHAP separately before commit
                    except Exception as e:
                        logger.error(
                            f"Failed to save SHAP for employee {employee_id}: {type(e).__name__}: {str(e)}"
                        )
                        # Don't fail the entire prediction if SHAP save fails

                db.commit()

                db.refresh(new_model_input)
                db.refresh(new_model_output)
                db.refresh(new_trace)

                trace_id = new_trace.trace_id
            else:
                trace_id = None

            predictions_output.append(
                PredictionOutput(
                    id_employee=employee_id,
                    prediction=pred_label,
                    probability=prob,
                    risk_category=risk_cat,
                    message=f"Employee {employee_id} is predicted to {pred_label} with {prob:.2%} attrition risk (Risk: {risk_cat}).",
                    trace_id=trace_id,
                    shap_values=(
                        shap_values_instance.tolist()
                        if shap_values_instance is not None
                        else None
                    ),
                    base_value=base_value_instance,
                    feature_names=feature_names_for_instance,
                )
            )
    except Exception as e:
        if not _is_db_disabled() and db is not None:
            db.rollback()
        logger.exception("Prediction failed with an unexpected error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )

    return predictions_output


@app.get("/", summary="Root endpoint", response_model=dict[str, str])
async def read_root():
    """Provides basic information about the API."""
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
    return {
        "status": "ok",
        "message": "API is healthy",
        "db_disabled": str(_is_db_disabled()),
    }


@app.post(
    "/jobs/report",
    summary="Enqueue report-generation job",
    response_model=dict[str, str],
)
async def create_report_job(
    batch_input: RawBatchPredictionInput,
    request: Request,
    db: Optional[Session] = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    """Create an async job to generate reports (Excel + SHAP images).

    **Authentication Required:** Provide X-API-Key header with valid API key.
    """
    if _is_db_disabled() or db is None:
        raise HTTPException(
            status_code=503, detail="Database is disabled; jobs are unavailable."
        )
    try:
        user_id = request.headers.get("X-User-ID", "demo1")
        job = Job(
            job_type="report",
            status="queued",
            payload_json=batch_input.model_dump(),
            user_id=user_id,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return {"job_id": job.job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {e}")


@app.get("/jobs/{job_id}", summary="Get job status", response_model=dict)
async def get_job_status(job_id: str, db: Optional[Session] = Depends(get_db)):
    if _is_db_disabled() or db is None:
        raise HTTPException(
            status_code=503, detail="Database is disabled; jobs are unavailable."
        )
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "job_type": job.job_type,
        "status": job.status,
        "updated_at": str(job.updated_at) if job.updated_at else None,
        "error": job.error,
    }


@app.get("/jobs/{job_id}/result", summary="Get job result", response_model=dict)
async def get_job_result(job_id: str, db: Optional[Session] = Depends(get_db)):
    if _is_db_disabled() or db is None:
        raise HTTPException(
            status_code=503, detail="Database is disabled; jobs are unavailable."
        )
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed" or not job.result_json:
        raise HTTPException(status_code=202, detail="Job not completed yet")
    return job.result_json


@app.get("/db_health", summary="Database health check", response_model=dict[str, str])
async def db_health(db: Optional[Session] = Depends(get_db)):
    """Check database connectivity and return status."""
    if _is_db_disabled() or db is None:
        return {
            "status": "disabled",
            "message": "Database usage is disabled via DISABLE_DB=1",
        }
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "message": "Database connection successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")


@app.post("/auth/login", summary="Authenticate user", response_model=dict)
async def login(
    username: str,
    password: str,
    db: Optional[Session] = Depends(get_db),
):
    """
    Authenticate a user with username and password.

    Returns user data if authentication is successful.

    **No API key required** - this endpoint is public for UI authentication.
    """
    if _is_db_disabled() or db is None:
        raise HTTPException(
            status_code=503,
            detail="Database is disabled; authentication unavailable."
        )

    try:
        # Query user by username
        user = db.query(User).filter(User.username == username).first()

        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Verify password
        if not User.verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Update last login timestamp
        from datetime import datetime, timezone
        user.last_login = datetime.now(timezone.utc)
        db.commit()
        db.refresh(user)

        # Return user data (excluding password hash)
        return {
            "success": True,
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role,
            "last_login": user.last_login.isoformat() if user.last_login else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Login failed with unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication error: {str(e)}"
        )


@app.get("/auth/user/{username}", summary="Get user info", response_model=dict)
async def get_user_info(
    username: str,
    db: Optional[Session] = Depends(get_db),
):
    """
    Get user information by username (for session validation).

    **No API key required** - this endpoint is public for UI session management.
    """
    if _is_db_disabled() or db is None:
        raise HTTPException(
            status_code=503,
            detail="Database is disabled; user info unavailable."
        )

    try:
        user = db.query(User).filter(User.username == username).first()

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role,
            "is_active": bool(user.is_active),
            "last_login": user.last_login.isoformat() if user.last_login else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Get user info failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user info: {str(e)}"
        )


@app.post(
    "/predict",
    response_model=BatchPredictionOutput,
    summary="Predict attrition risk for a batch of employees",
)
async def predict_attrition(
    batch_input: RawBatchPredictionInput,
    request: Request,
    db: Optional[Session] = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    """Predicts the attrition risk for a list of employees based on their features.
    All model inputs, outputs, and prediction traceability are recorded in the database.

    **Authentication Required:** Provide X-API-Key header with valid API key.
    """
    # Fast path: skip SHAP for quick predictions and DB writes
    predictions_output = generate_predictions(
        batch_input, request, db, compute_shap=False
    )
    return BatchPredictionOutput(predictions=predictions_output)


@app.post(
    "/predict_report",
    summary="Predict and generate report artifacts (Excel, SHAP images)",
)
async def predict_attrition_report(
    batch_input: RawBatchPredictionInput,
    request: Request,
    db: Optional[Session] = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    """Runs prediction and returns:

    **Authentication Required:** Provide X-API-Key header with valid API key.
    - predictions: same as /predict
    - excel_base64: Excel workbook (Summary, Features, Metrics) in base64
    - shap_images: list of base64-encoded waterfall plots per employee
    """
    predictions_output = generate_predictions(
        batch_input, request, db, compute_shap=True
    )

    # Build Summary sheet
    summary_rows = []
    for p in predictions_output:
        summary_rows.append(
            {
                "Employee_ID": p.id_employee,
                "Risk_Attrition": p.risk_category,
                "Attrition_Risk_Percentage": round(float(p.probability) * 100, 1),
                "Prediction": p.prediction,
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    # Build Features sheet (stacked SHAP values)
    features_frames = []
    for p in predictions_output:
        if p.shap_values is not None and p.feature_names is not None:
            # Use feature names from API; fallback to generic names if needed
            feature_names = (
                p.feature_names
                if len(p.feature_names) == len(p.shap_values)
                else [f"Feature {i}" for i in range(len(p.shap_values))]
            )
            df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Coefficient": p.shap_values,
                }
            )
            df["Employee_ID"] = p.id_employee
            df["Prediction"] = p.prediction
            features_frames.append(df)
    features_df = (
        pd.concat(features_frames, ignore_index=True)
        if features_frames
        else pd.DataFrame()
    )

    # Build Metrics sheet
    total = len(predictions_output)
    predicted_leave = sum(1 for p in predictions_output if p.prediction == "Leave")
    predicted_stay = sum(1 for p in predictions_output if p.prediction == "Stay")
    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Total Employees Processed",
                "Predicted to Leave",
                "Predicted to Stay",
            ],
            "Value": [total, predicted_leave, predicted_stay],
        }
    )

    # Write Excel to bytes
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        if not features_df.empty:
            features_df[
                ["Employee_ID", "Feature", "Coefficient", "Prediction"]
            ].to_excel(writer, sheet_name="Features", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
    excel_buffer.seek(0)
    excel_b64 = base64.b64encode(excel_buffer.read()).decode("utf-8")

    # Generate SHAP waterfall images per employee
    shap_images = []
    for p in predictions_output:
        if p.shap_values is None or p.base_value is None:
            continue
        feature_names = (
            p.feature_names
            if p.feature_names and len(p.feature_names) == len(p.shap_values)
            else [f"Feature {i}" for i in range(len(p.shap_values))]
        )

        # Filter out id_employee from SHAP visualization
        filtered_shap_values, filtered_feature_names = filter_id_employee_from_shap(
            p.shap_values, feature_names
        )

        explanation = shap.Explanation(
            values=np.array(filtered_shap_values),
            base_values=p.base_value,
            data=np.zeros(len(filtered_shap_values)),
            feature_names=filtered_feature_names,
        )
        shap.plots.waterfall(explanation, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        shap_images.append(
            {
                "employee_id": p.id_employee,
                "risk_category": p.risk_category,
                "attrition_prob": float(p.probability),
                "prediction_type": p.prediction,
                "img_base64": img_str,
            }
        )

    return {
        "predictions": [p.model_dump() for p in predictions_output],
        "excel_base64": excel_b64,
        "shap_images": shap_images,
    }


@app.post(
    "/predict_excel",
    summary="Predict and return Excel report only",
)
async def predict_excel(
    batch_input: RawBatchPredictionInput,
    request: Request,
    db: Optional[Session] = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    """Generate predictions and return Excel report.

    **Authentication Required:** Provide X-API-Key header with valid API key.
    """
    predictions_output = generate_predictions(
        batch_input, request, db, compute_shap=True
    )

    summary_rows = [
        {
            "Employee_ID": p.id_employee,
            "Risk_Attrition": p.risk_category,
            "Attrition_Risk_Percentage": round(float(p.probability) * 100, 1),
            "Prediction": p.prediction,
        }
        for p in predictions_output
    ]
    summary_df = pd.DataFrame(summary_rows)

    features_frames = []
    for p in predictions_output:
        if p.shap_values is not None and p.feature_names is not None:
            feature_names = (
                p.feature_names
                if len(p.feature_names) == len(p.shap_values)
                else [f"Feature {i}" for i in range(len(p.shap_values))]
            )
            df = pd.DataFrame({"Feature": feature_names, "Coefficient": p.shap_values})
            df["Employee_ID"] = p.id_employee
            df["Prediction"] = p.prediction
            features_frames.append(df)
    features_df = (
        pd.concat(features_frames, ignore_index=True)
        if features_frames
        else pd.DataFrame()
    )

    total = len(predictions_output)
    predicted_leave = sum(1 for p in predictions_output if p.prediction == "Leave")
    predicted_stay = sum(1 for p in predictions_output if p.prediction == "Stay")
    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Total Employees Processed",
                "Predicted to Leave",
                "Predicted to Stay",
            ],
            "Value": [total, predicted_leave, predicted_stay],
        }
    )

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        if not features_df.empty:
            features_df[
                ["Employee_ID", "Feature", "Coefficient", "Prediction"]
            ].to_excel(writer, sheet_name="Features", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
    excel_buffer.seek(0)
    excel_b64 = base64.b64encode(excel_buffer.read()).decode("utf-8")

    return {"excel_base64": excel_b64}


@app.post(
    "/predict_shap_images",
    summary="Predict and return SHAP waterfall plots only",
)
async def predict_shap_images(
    batch_input: RawBatchPredictionInput,
    request: Request,
    db: Optional[Session] = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    """Runs prediction and returns only SHAP waterfall images as base64 list.

    **Authentication Required:** Provide X-API-Key header with valid API key.
    """
    predictions_output = generate_predictions(
        batch_input, request, db, compute_shap=True
    )

    shap_images = []
    for p in predictions_output:
        if p.shap_values is None or p.base_value is None:
            continue
        feature_names = (
            p.feature_names
            if p.feature_names and len(p.feature_names) == len(p.shap_values)
            else [f"Feature {i}" for i in range(len(p.shap_values))]
        )

        # Filter out id_employee from SHAP visualization
        filtered_shap_values, filtered_feature_names = filter_id_employee_from_shap(
            p.shap_values, feature_names
        )

        explanation = shap.Explanation(
            values=np.array(filtered_shap_values),
            base_values=p.base_value,
            data=np.zeros(len(filtered_shap_values)),
            feature_names=filtered_feature_names,
        )
        shap.plots.waterfall(explanation, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        shap_images.append(
            {
                "employee_id": p.id_employee,
                "risk_category": p.risk_category,
                "attrition_prob": float(p.probability),
                "prediction_type": p.prediction,
                "img_base64": img_str,
            }
        )

    return {"shap_images": shap_images}


@app.post(
    "/predict_shap_html",
    summary="Generate HTML page with SHAP analysis for all employees",
    response_class=HTMLResponse,
)
async def predict_shap_html(
    batch_input: RawBatchPredictionInput,
    request: Request,
    db: Optional[Session] = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    """Generates an interactive HTML page with expandable employee sections showing SHAP analysis.

    **Authentication Required:** Provide X-API-Key header with valid API key.

    Returns an HTML page with:
    - Expandable sections for each employee
    - SHAP waterfall diagrams
    - SHAP values table
    - Download button for the HTML page
    """
    predictions_output = generate_predictions(
        batch_input, request, db, compute_shap=True
    )

    # Generate SHAP images and create employee sections
    employee_sections = []

    for idx, p in enumerate(predictions_output, 1):
        if p.shap_values is None or p.base_value is None:
            continue

        # Generate SHAP waterfall plot
        feature_names = (
            p.feature_names
            if p.feature_names and len(p.feature_names) == len(p.shap_values)
            else [f"Feature {i}" for i in range(len(p.shap_values))]
        )

        # Filter out id_employee from SHAP visualization
        filtered_shap_values, filtered_feature_names = filter_id_employee_from_shap(
            p.shap_values, feature_names
        )

        explanation = shap.Explanation(
            values=np.array(filtered_shap_values),
            base_values=p.base_value,
            data=np.zeros(len(filtered_shap_values)),
            feature_names=filtered_feature_names,
        )

        shap.plots.waterfall(explanation, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Determine risk color
        risk_color = {
            "High": "#dc3545",
            "Medium": "#ffc107",
            "Low": "#28a745"
        }.get(p.risk_category, "#6c757d")

        # Create employee section HTML
        employee_section = f"""
        <div class="employee-card">
            <div class="employee-header" onclick="toggleEmployee({idx})">
                <h3>Employee {p.id_employee}</h3>
                <div class="employee-summary">
                    <span class="risk-badge" style="background-color: {risk_color};">{p.risk_category} Risk</span>
                    <span class="prob-badge">{p.probability:.1%} Attrition Probability</span>
                    <span class="pred-badge">Prediction: {p.prediction}</span>
                </div>
                <span class="toggle-icon" id="icon-{idx}">▼</span>
            </div>
            <div class="employee-content" id="content-{idx}">
                <div class="shap-section">
                    <h4>SHAP Waterfall Diagram</h4>
                    <img src="data:image/png;base64,{img_base64}" alt="SHAP Waterfall for Employee {p.id_employee}" class="shap-image">
                </div>
            </div>
        </div>
        """
        employee_sections.append(employee_section)

    # Build complete HTML page
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Employee Risk Analysis - SHAP Report</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                min-height: 100vh;
            }}

            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }}

            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}

            .header h1 {{
                font-size: 2rem;
                margin-bottom: 10px;
            }}

            .header p {{
                opacity: 0.9;
                font-size: 1rem;
            }}

            .download-section {{
                background: #f8f9fa;
                padding: 20px 30px;
                border-bottom: 1px solid #dee2e6;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            .download-btn {{
                background: #28a745;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 1rem;
                cursor: pointer;
                transition: background 0.3s;
            }}

            .download-btn:hover {{
                background: #218838;
            }}

            .content {{
                padding: 30px;
            }}

            .employee-card {{
                margin-bottom: 20px;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                overflow: hidden;
                transition: box-shadow 0.3s;
            }}

            .employee-card:hover {{
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}

            .employee-header {{
                background: #f8f9fa;
                padding: 20px;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: background 0.3s;
            }}

            .employee-header:hover {{
                background: #e9ecef;
            }}

            .employee-header h3 {{
                font-size: 1.25rem;
                color: #333;
            }}

            .employee-summary {{
                display: flex;
                gap: 15px;
                align-items: center;
            }}

            .risk-badge, .prob-badge, .pred-badge {{
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 0.875rem;
                font-weight: 500;
            }}

            .risk-badge {{
                color: white;
            }}

            .prob-badge {{
                background: #e3f2fd;
                color: #1976d2;
            }}

            .pred-badge {{
                background: #f3e5f5;
                color: #7b1fa2;
            }}

            .toggle-icon {{
                font-size: 1.5rem;
                transition: transform 0.3s;
            }}

            .toggle-icon.rotated {{
                transform: rotate(-180deg);
            }}

            .employee-content {{
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease;
            }}

            .employee-content.expanded {{
                max-height: 5000px;
                padding: 20px;
                border-top: 1px solid #dee2e6;
            }}

            .shap-section {{
                margin-bottom: 30px;
            }}

            .shap-section h4, .shap-values-section h4 {{
                font-size: 1.1rem;
                margin-bottom: 15px;
                color: #495057;
            }}

            .shap-image {{
                max-width: 100%;
                height: auto;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }}

            .shap-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}

            .shap-table th {{
                background: #f8f9fa;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                border-bottom: 2px solid #dee2e6;
            }}

            .shap-table td {{
                padding: 10px 12px;
                border-bottom: 1px solid #dee2e6;
            }}

            .shap-table tbody tr:hover {{
                background: #f8f9fa;
            }}

            .stats-summary {{
                background: #e3f2fd;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
            }}

            .stats-summary h3 {{
                margin-bottom: 15px;
                color: #1976d2;
            }}

            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }}

            .stat-item {{
                background: white;
                padding: 15px;
                border-radius: 6px;
                text-align: center;
            }}

            .stat-value {{
                font-size: 2rem;
                font-weight: bold;
                color: #1976d2;
            }}

            .stat-label {{
                color: #666;
                font-size: 0.875rem;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Employee Risk Analysis in Detail</h1>
                <p>SHAP (SHapley Additive exPlanations) Analysis Report</p>
            </div>

            <div class="download-section">
                <div>
                    <strong>Total Employees Analyzed:</strong> {len(predictions_output)}
                </div>
                <button class="download-btn" onclick="downloadHTML()">
                    📥 Download Full Report
                </button>
            </div>

            <div class="content">
                <div class="stats-summary">
                    <h3>Summary Statistics</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value">{len(predictions_output)}</div>
                            <div class="stat-label">Total Employees</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{sum(1 for p in predictions_output if p.prediction == 'Leave')}</div>
                            <div class="stat-label">Predicted to Leave</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{sum(1 for p in predictions_output if p.prediction == 'Stay')}</div>
                            <div class="stat-label">Predicted to Stay</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{sum(1 for p in predictions_output if p.risk_category == 'High')}</div>
                            <div class="stat-label">High Risk</div>
                        </div>
                    </div>
                </div>

                {"".join(employee_sections)}
            </div>
        </div>

        <script>
            function toggleEmployee(id) {{
                const content = document.getElementById('content-' + id);
                const icon = document.getElementById('icon-' + id);

                content.classList.toggle('expanded');
                icon.classList.toggle('rotated');
            }}

            function downloadHTML() {{
                const htmlContent = document.documentElement.outerHTML;
                const blob = new Blob([htmlContent], {{ type: 'text/html' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'employee_risk_analysis_shap_report.html';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }}
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)
