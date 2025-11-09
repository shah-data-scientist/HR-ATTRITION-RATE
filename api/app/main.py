import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np

from api.app.schemas import EmployeeFeatures, PredictionOutput, BatchPredictionInput, BatchPredictionOutput

from contextlib import asynccontextmanager

# --- Configuration ---
# Define risk categories for Excel report and HTML visualization
RISK_THRESHOLDS = {
    "Low": (0.0, 0.3),
    "Medium": (0.3, 0.7),
    "High": (0.7, 1.0)
}

# --- Global Model and Preprocessor ---
model = None
expected_model_columns = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model, expected_model_columns
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'employee_attrition_pipeline.pkl')
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}. Please ensure the model is trained and saved.")
    
    model = joblib.load(model_path)
    expected_model_columns = get_expected_columns_from_pipeline(model)
    print(f"Model loaded successfully from {model_path}")
    print(f"Expected model columns: {expected_model_columns}")
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
    if 'augmentation_salaire_precedente' in df.columns:
        df['augmentation_salaire_precedente'] = (
            df['augmentation_salaire_precedente'].astype(str)
            .str.replace('%','', regex=False).str.replace(',','.', regex=False).str.strip()
        )
        df['augmentation_salaire_precedente'] = pd.to_numeric(df['augmentation_salaire_precedente'], errors='coerce')/100.0
    for col in ['heures_supplementaires','heure_supplementaires','heures_supplémentaires']:
        if col in df.columns:
            df[col] = df[col].replace({'Oui':1,'Non':0,'oui':1,'non':0, True:1, False:0}).astype('Int64')
            if col != 'heures_supplementaires':
                df.rename(columns={col:'heures_supplementaires'}, inplace=True)
    if 'eval_number' in df.columns:
        df['id_employee'] = df['eval_number'].astype(str).str.replace('E_','', regex=False)
        df['id_employee'] = pd.to_numeric(df['id_employee'], errors='coerce').astype('Int64')
        df.drop(columns=['eval_number'], inplace=True, errors='ignore')
    return df

def _clean_extrait_sirh(df):
    df = df.copy()
    if 'genre' in df.columns:
        df['genre'] = df['genre'].replace({'M':1,'F':0,'m':1,'f':0}).astype('Int64')
    for col in ['nombre_heures_travailless', '...']: # Removed '...' as it's a placeholder
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df

def _clean_extrait_sondage(df):
    df = df.copy()
    if 'code_sondage' in df.columns:
        df.rename(columns={'code_sondage':'id_employee'}, inplace=True)
    if 'id_employee' in df.columns:
        df['id_employee'] = pd.to_numeric(df['id_employee'], errors='coerce').astype('Int64')
    return df

def load_and_merge_data(eval_df, sirh_df, sond_df):
    eval_df = _clean_extrait_eval(eval_df)
    sirh_df = _clean_extrait_sirh(sirh_df)
    sond_df = _clean_extrait_sondage(sond_df)

    # Ensure 'id_employee' column exists in all dataframes before merging
    for df in [eval_df, sirh_df, sond_df]:
        if 'id_employee' not in df.columns:
            df['id_employee'] = pd.Series(dtype='Int64')

    merged = eval_df.merge(sirh_df, on='id_employee', how='outer', suffixes=('_eval','_sirh'))
    merged = merged.merge(sond_df, on='id_employee', how='outer')
    # if '...' in merged.columns: # Removed '...' as it's a placeholder
    #     merged.drop(columns=['...'], inplace=True, errors='ignore')
    merged.drop_duplicates(inplace=True)
    return merged

def clean_and_engineer_features(df):
    """Applies the same cleaning and feature engineering steps as in the notebook."""
    df = df.copy()
    # Feature Engineering steps
    if {"note_evaluation_actuelle","note_evaluation_precedente"}.issubset(df.columns):
        df["improvement_evaluation"] = df["note_evaluation_actuelle"] - df["note_evaluation_precedente"]

    sat_cols = ["satisfaction_employee_nature_travail","satisfaction_employee_equipe","satisfaction_employee_equilibre_pro_perso"]
    if set(sat_cols).issubset(df.columns):
        df["total_satisfaction"] = df[sat_cols[0]] * df[sat_cols[1]] * df[sat_cols[2]]

    if {"annees_dans_le_poste_actuel","annees_dans_l_entreprise"}.issubset(df.columns):
        denom = df["annees_dans_l_entreprise"].replace(0, np.nan)
        df["work_mobility"] = (df["annees_dans_le_poste_actuel"] / denom).fillna(0)
        
    return df

def get_expected_columns_from_pipeline(pipeline):
    """Gets the list of columns the model was trained on."""
    preprocessor = pipeline.named_steps['preprocessor']
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
    buffer = 0.05 # How close to the threshold to be considered "Medium" (e.g., 5 percentage points)
    min_medium_prob = 0.20 # Minimum absolute probability to be considered "Medium" risk (e.g., 20%)

    if probability >= threshold + buffer: # Clearly above threshold
        return "High"
    elif probability < threshold - buffer: # Clearly below threshold
        return "Low"
    elif probability >= min_medium_prob: # Close to threshold AND above minimum for Medium
        return "Medium"
    else: # Close to threshold but below min_medium_prob
        return "Low" # Default to Low if not High, not clearly Low, and below min_medium_prob

@app.get("/", summary="Root endpoint", response_model=Dict[str, str])
async def read_root():
    """
    Provides basic information about the API.
    """
    return {"message": "Welcome to the Employee Attrition Prediction API!",
            "version": app.version,
            "documentation_url": "/docs"}

@app.post("/predict", response_model=BatchPredictionOutput, summary="Predict attrition risk for a batch of employees")
async def predict_attrition(batch_input: BatchPredictionInput):
    """
    Predicts the attrition risk for a list of employees based on their features.

    The input should be a list of employee feature sets. Each set should contain
    the raw features as defined in the `EmployeeFeatures` schema.

    The API will preprocess the data, apply feature engineering, and then use
    the loaded machine learning model to predict the probability of attrition
    and categorize the risk (Low, Medium, High).
    """
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded yet.")

    predictions_output: List[PredictionOutput] = []

    # Convert list of Pydantic models to a list of dictionaries
    employees_data = [employee.model_dump() for employee in batch_input.employees]
    
    # Convert to DataFrame
    input_df = pd.DataFrame(employees_data)

    # Ensure 'id_employee' is present for merging, even if it's just a placeholder
    if 'id_employee' not in input_df.columns:
        input_df['id_employee'] = range(len(input_df)) # Assign temporary IDs
    
    # Apply feature engineering
    processed_data = clean_and_engineer_features(input_df.copy())

    # Align columns with the model's expected columns
    data_for_prediction = pd.DataFrame(columns=expected_model_columns)
    for col in expected_model_columns:
        if col in processed_data.columns:
            data_for_prediction[col] = processed_data[col]
        else:
            data_for_prediction[col] = 0 # Placeholder, ideally use preprocessor's default

    # Ensure order of columns
    data_for_prediction = data_for_prediction[expected_model_columns]

    try:
        # Make predictions
        prediction_proba = model.predict_proba(data_for_prediction)[:, 1]
        predictions = (prediction_proba >= 0.5).astype(int) # Using default threshold 0.5 for API

        for i, employee_data in enumerate(batch_input.employees):
            prob = prediction_proba[i]
            pred_label = "Leave" if predictions[i] == 1 else "Stay"
            risk_cat = get_risk_category(prob, threshold=0.5) # Using default threshold 0.5 for API

            predictions_output.append(
                PredictionOutput(
                    id_employee=employee_data.id_employee,
                    prediction=pred_label,
                    probability=float(prob),
                    risk_category=risk_cat,
                    message=f"Employee {employee_data.id_employee} is predicted to {pred_label} with {prob:.2%} attrition risk (Risk: {risk_cat})."
                )
            )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}")

    return BatchPredictionOutput(predictions=predictions_output)
