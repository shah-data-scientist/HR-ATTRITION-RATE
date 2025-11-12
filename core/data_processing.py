"""Data cleaning, transformation, and feature engineering.

This module replicates the exact cleaning and feature engineering
logic from train.py to ensure consistency between training and inference.
"""

import numpy as np
import pandas as pd


def clean_raw_input(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw input data (matching train.py cleaning logic).

    Transformations:
    1. Convert genre: "M"/"Homme" → 1, "F"/"Femme" → 0
    2. Convert heures_supplementaires: "Oui" → 1, "Non" → 0
    3. Extract id_employee from eval_number if present
    4. Convert augmentation_salaire_precedente from "X%" to float

    Args:
        df: DataFrame with raw input features

    Returns:
        DataFrame with cleaned features
    """
    df = df.copy()

    # 1. Clean genre: convert string to 0/1
    if "genre" in df.columns:
        df["genre"] = (
            df["genre"]
            .astype(str)
            .str.lower()
            .replace({"m": 1, "f": 0, "homme": 1, "femme": 0, "male": 1, "female": 0})
            .infer_objects(copy=False)
            .astype("Int64")
        )

    # 2. Clean heures_supplementaires: "Oui"/"Non" → 1/0
    # Handle multiple possible column names (train.py checks for variants)
    for col in ["heures_supplementaires", "heure_supplementaires", "heures_supplémentaires"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .replace({"oui": 1, "non": 0, "yes": 1, "no": 0, "true": 1, "false": 0})
                .infer_objects(copy=False)
                .astype("Int64")
            )
            # Standardize column name
            if col != "heures_supplementaires":
                df.rename(columns={col: "heures_supplementaires"}, inplace=True)

    # 3. Clean augmentation_salaire_precedente: handle "X%" format
    # Note: column has typo "augementation" in some places
    for col in ["augmentation_salaire_precedente", "augementation_salaire_precedente"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Standardize to the typo version (matches train.py output)
            if col == "augmentation_salaire_precedente":
                df.rename(columns={col: "augementation_salaire_precedente"}, inplace=True)

    # 4. Extract id_employee from eval_number if needed
    if "eval_number" in df.columns and "id_employee" not in df.columns:
        df["id_employee"] = (
            df["eval_number"]
            .astype(str)
            .str.replace("E_", "", regex=False)
        )
        df["id_employee"] = pd.to_numeric(df["id_employee"], errors="coerce").astype("Int64")

    # 5. Clean ayant_enfants: convert to consistent format if needed
    if "ayant_enfants" in df.columns:
        # If it's "Oui"/"Non", keep as string; if 0/1, convert to string
        df["ayant_enfants"] = df["ayant_enfants"].astype(str)

    # 6. Drop unnecessary columns
    cols_to_drop = ["eval_number", "code_sondage", "nombre_heures_travailless", "..."]
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True, errors="ignore")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering (matching train.py logic).

    Engineered features:
    1. improvement_evaluation: change in evaluation score
    2. total_satisfaction: product of three satisfaction metrics
    3. work_mobility: ratio of years in current role to years in company

    Args:
        df: DataFrame with cleaned features

    Returns:
        DataFrame with engineered features added
    """
    df = df.copy()

    # Feature 1: Improvement in evaluation score
    if {"note_evaluation_actuelle", "note_evaluation_precedente"}.issubset(df.columns):
        df["improvement_evaluation"] = (
            df["note_evaluation_actuelle"] - df["note_evaluation_precedente"]
        )
    else:
        df["improvement_evaluation"] = 0

    # Feature 2: Total satisfaction (product of three satisfaction scores)
    sat_cols = [
        "satisfaction_employee_nature_travail",
        "satisfaction_employee_equipe",
        "satisfaction_employee_equilibre_pro_perso",
    ]
    if set(sat_cols).issubset(df.columns):
        df["total_satisfaction"] = (
            df[sat_cols[0]] * df[sat_cols[1]] * df[sat_cols[2]]
        )
    else:
        df["total_satisfaction"] = 0

    # Feature 3: Work mobility (years in role / years in company)
    if {"annees_dans_le_poste_actuel", "annees_dans_l_entreprise"}.issubset(
        df.columns
    ):
        denom = df["annees_dans_l_entreprise"].replace(0, np.nan)
        df["work_mobility"] = (df["annees_dans_le_poste_actuel"] / denom).fillna(0)
    else:
        df["work_mobility"] = 0.0

    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: clean raw input and apply feature engineering.

    This is the main entry point for data processing.

    Args:
        df: DataFrame with raw input features

    Returns:
        DataFrame ready for model input (after cleaning and engineering)
    """
    df = clean_raw_input(df)
    df = engineer_features(df)
    return df
