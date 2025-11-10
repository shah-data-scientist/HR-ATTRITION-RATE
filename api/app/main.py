import os
from fastapi import FastAPI, HTTPException, status, Depends
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
import json  # For JSON column
from datetime import datetime  # For timestamps
import logging  # Add logging import

from sqlalchemy.orm import Session  # Explicitly import Session for type hinting

from api.app.schemas import (
    EmployeeFeatures,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
)

from contextlib import asynccontextmanager

# Database imports
from database.database import get_db
from database.models import Employee, ModelInput, ModelOutput, PredictionTraceability

logger = logging.getLogger("uvicorn.error")

# exactly the model's expected columns you printed at startup
EXPECTED_COLS = [
    "satisfaction_employee_environnement",
    "note_evaluation_precedente",
    "niveau_hierarchique_poste",
    "satisfaction_employee_nature_travail",
    "satisfaction_employee_equipe",
    "satisfaction_employee_equilibre_pro_perso",
    "note_evaluation_actuelle",
    "heures_supplementaires",
    "augementation_salaire_precedente",
    "id_employee",
    "age",
    "genre",
    "revenu_mensuel",
    "statut_marital",
    "departement",
    "poste",
    "nombre_experiences_precedentes",
    "annee_experience_totale",
    "annees_dans_l_entreprise",
    "annees_dans_le_poste_actuel",
    "nombre_participation_pee",
    "nb_formations_suivies",
    "nombre_employee_sous_responsabilite",
    "distance_domicile_travail",
    "niveau_education",
    "domaine_etude",
    "ayant_enfants",
    "frequence_deplacement",
    "annees_depuis_la_derniere_promotion",
    "annes_sous_responsable_actuel",
    "improvement_evaluation",
    "total_satisfaction",
    "work_mobility",
]

# categorical vs numeric (adjust if your training said otherwise)
CATEGORICAL = [
    "statut_marital",
    "departement",
    "poste",
    "domaine_etude",
    "frequence_deplacement",
    "augementation_salaire_precedente",
    "ayant_enfants",
]
NUMERIC = [c for c in EXPECTED_COLS if c not in CATEGORICAL]


def coerce_and_align(df: pd.DataFrame) -> pd.DataFrame:
    # keep only expected columns and create missing ones
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[EXPECTED_COLS]

    # normalize whitespace/NaNs
    df.replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan}, inplace=True)

    # force numeric
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # force categorical/text
    for col in CATEGORICAL:
        df[col] = df[col].astype("string").str.strip().where(df[col].notna(), None)

    # helpful one-time log
    logger.info("Incoming dtypes for prediction:\n%s", df.dtypes.to_string())
    # quick scan for mixed types (objects that should be numeric)
    bad = [c for c in NUMERIC if not np.issubdtype(df[c].dtype, np.number)]
    if bad:
        logger.error("Non-numeric columns among NUMERIC: %s", bad)

    return df


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


# --- Helper Functions (copied/adapted from main app.py) ---
def _clean_extrait_eval(df):
    df = df.copy()
    if "augmentation_salaire_precedente" in df.columns:
        df["augmentation_salaire_precedente"] = (
            df["augmentation_salaire_precedente"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df["augmentation_salaire_precedente"] = (
            pd.to_numeric(df["augmentation_salaire_precedente"], errors="coerce")
            / 100.0
        )
    for col in [
        "heures_supplementaires",
        "heure_supplementaires",
        "heures_supplémentaires",
    ]:
        if col in df.columns:
            df[col] = (
                df[col]
                .replace({"Oui": 1, "Non": 0, "oui": 1, "non": 0, True: 1, False: 0})
                .astype("Int64")
            )
            if col != "heures_supplementaires":
                df.rename(columns={col: "heures_supplementaires"}, inplace=True)
    if "eval_number" in df.columns:
        df["id_employee"] = (
            df["eval_number"].astype(str).str.replace("E_", "", regex=False)
        )
        df["id_employee"] = pd.to_numeric(df["id_employee"], errors="coerce").astype(
            "Int64"
        )
        df.drop(columns=["eval_number"], inplace=True, errors="ignore")
    return df


def _clean_extrait_sirh(df):
    df = df.copy()
    if "genre" in df.columns:
        df["genre"] = (
            df["genre"].replace({"M": 1, "F": 0, "m": 1, "f": 0}).astype("Int64")
        )
    for col in [
        "nombre_heures_travailless",
        "...",
    ]:  # Removed '...' as it's a placeholder
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df


def _clean_extrait_sondage(df):
    df = df.copy()
    if "code_sondage" in df.columns:
        df.rename(columns={"code_sondage": "id_employee"}, inplace=True)
    if "id_employee" in df.columns:
        df["id_employee"] = pd.to_numeric(df["id_employee"], errors="coerce").astype(
            "Int64"
        )
    return df


def load_and_merge_data(eval_df, sirh_df, sond_df):
    eval_df = _clean_extrait_eval(eval_df)
    sirh_df = _clean_extrait_sirh(sirh_df)
    sond_df = _clean_extrait_sondage(sond_df)

    # Ensure 'id_employee' column exists in all dataframes before merging
    for df in [eval_df, sirh_df, sond_df]:
        if "id_employee" not in df.columns:
            df["id_employee"] = pd.Series(dtype="Int64")

    merged = eval_df.merge(
        sirh_df, on="id_employee", how="outer", suffixes=("_eval", "_sirh")
    )
    merged = merged.merge(sond_df, on="id_employee", how="outer")
    # if '...' in merged.columns: # Removed '...' as it's a placeholder
    #     merged.drop(columns=['...'], inplace=True, errors='ignore')
    merged.drop_duplicates(inplace=True)
    return merged


def clean_and_engineer_features(df):
    """Applies the same cleaning and feature engineering steps as in the notebook."""
    df = df.copy()
    # Feature Engineering steps
    if {"note_evaluation_actuelle", "note_evaluation_precedente"}.issubset(df.columns):
        df["improvement_evaluation"] = (
            df["note_evaluation_actuelle"] - df["note_evaluation_precedente"]
        )

    sat_cols = [
        "satisfaction_employee_nature_travail",
        "satisfaction_employee_equipe",
        "satisfaction_employee_equilibre_pro_perso",
    ]
    if set(sat_cols).issubset(df.columns):
        df["total_satisfaction"] = df[sat_cols[0]] * df[sat_cols[1]] * df[sat_cols[2]]

    if {"annees_dans_le_poste_actuel", "annees_dans_l_entreprise"}.issubset(df.columns):
        denom = df["annees_dans_l_entreprise"].replace(0, np.nan)
        df["work_mobility"] = (df["annees_dans_le_poste_actuel"] / denom).fillna(0)

    return df


def get_expected_columns_from_pipeline(pipeline):
    """Gets the list of columns the model was trained on."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.feature_names_in_)


def get_risk_category(probability: float, threshold: float = 0.5) -> str:
    """
    Categorizes attrition risk based on the difference between probability and a dynamic threshold,
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
    elif probability < threshold - buffer:  # Clearly below threshold
        return "Low"
    elif (
        probability >= min_medium_prob
    ):  # Close to threshold AND above minimum for Medium
        return "Medium"
    else:  # Close to threshold but below min_medium_prob
        return "Low"  # Default to Low if not High, not clearly Low, and below min_medium_prob


@app.get("/", summary="Root endpoint", response_model=Dict[str, str])
async def read_root():
    """
    Provides basic information about the API.
    """
    return {
        "message": "Welcome to the Employee Attrition Prediction API!",
        "version": app.version,
        "documentation_url": "/docs",
    }


@app.post(
    "/predict",
    response_model=BatchPredictionOutput,
    summary="Predict attrition risk for a batch of employees",
)
async def predict_attrition(
    batch_input: BatchPredictionInput, db: Session = Depends(get_db)
):
    """
    Predicts the attrition risk for a list of employees based on their features.
    All model inputs, outputs, and prediction traceability are recorded in the database.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet.",
        )

    predictions_output: List[PredictionOutput] = []

    # Convert list of Pydantic models to a list of dictionaries
    employees_data_list = [employee.model_dump() for employee in batch_input.employees]

    # Convert to DataFrame for processing
    input_df = pd.DataFrame(employees_data_list)

    # Explicitly cast relevant columns to numeric types to prevent type mismatch errors
    numeric_cols_to_cast = [
        "age",
        "genre",
        "revenu_mensuel",
        "nombre_experiences_precedentes",
        "annee_experience_totale",
        "annees_dans_l_entreprise",
        "annees_dans_le_poste_actuel",
        "nombre_participation_pee",
        "nb_formations_suivies",
        "nombre_employee_sous_responsabilite",
        "distance_domicile_travail",
        "niveau_education",
        "ayant_enfants",
        "annees_depuis_la_derniere_promotion",
        "annes_sous_responsable_actuel",
        "satisfaction_employee_environnement",
        "note_evaluation_precedente",
        "niveau_hierarchique_poste",
        "satisfaction_employee_nature_travail",
        "satisfaction_employee_equipe",
        "satisfaction_employee_equilibre_pro_perso",
        "note_evaluation_actuelle",
        "heures_supplementaires",
        "augementation_salaire_precedente",
    ]
    for col in numeric_cols_to_cast:
        if col in input_df.columns:
            # Use errors='coerce' to turn unparseable values into NaN, then fill NaN with 0
            input_df[col] = (
                pd.to_numeric(input_df[col], errors="coerce").fillna(0).astype(float)
            )

    # Ensure 'id_employee' is present and filled with integers
    if "id_employee" not in input_df.columns or input_df["id_employee"].isnull().all():
        # If id_employee is entirely missing or all null, assign temporary IDs
        input_df["id_employee"] = range(
            1, len(input_df) + 1
        )  # Start from 1 for new entries
    else:
        # Fill any individual missing id_employee values with temporary IDs
        missing_id_mask = input_df["id_employee"].isna()
        if missing_id_mask.any():
            max_existing_id = (
                db.query(Employee.id_employee)
                .order_by(Employee.id_employee.desc())
                .first()
            )
            start_id = (max_existing_id[0] + 1) if max_existing_id else 1
            new_ids = range(start_id, start_id + missing_id_mask.sum())
            input_df.loc[missing_id_mask, "id_employee"] = list(new_ids)
    input_df["id_employee"] = input_df["id_employee"].astype(
        int
    )  # Ensure it's integer type

    # Apply feature engineering
    processed_data = clean_and_engineer_features(input_df.copy())

    # ✨ enforce schema and coerce types
    data_for_prediction = coerce_and_align(processed_data)

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
                model_version="v1.0",  # TODO: Make this dynamic from config
                prediction_source="API",
                request_metadata={
                    "user_agent": "FastAPI client"
                },  # TODO: Extract actual request metadata
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
