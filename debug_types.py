import pandas as pd
import numpy as np
import logging
import json
import os

# Configure logging to print to console
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("debug_types")


# --- Helper Functions (copied from api/app/main.py and utils.py) ---
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
    for df_item in [
        eval_df,
        sirh_df,
        sond_df,
    ]:  # Renamed df to df_item to avoid conflict
        if "id_employee" not in df_item.columns:
            df_item["id_employee"] = pd.Series(dtype="Int64")

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


# --- Definitions from api/app/main.py ---
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
        logger.warning("Non-numeric columns among NUMERIC: %s", bad)

    return df


# --- Simulate Input Payload ---
json_payload = {
    "employees": [
        {
            "id_employee": 1,
            "age": 35,
            "genre": 1,
            "revenu_mensuel": 6000.0,
            "statut_marital": "Marié",
            "departement": "R&D",
            "poste": "Développeur",
            "nombre_experiences_precedentes": 2,
            "annee_experience_totale": 10,
            "annees_dans_l_entreprise": 5,
            "annees_dans_le_poste_actuel": 3,
            "nombre_participation_pee": 1,
            "nb_formations_suivies": 2,
            "nombre_employee_sous_responsabilite": 0,
            "distance_domicile_travail": 15,
            "niveau_education": 3,
            "domaine_etude": "Informatique",
            "ayant_enfants": 1,
            "frequence_deplacement": "Rarement",
            "annees_depuis_la_derniere_promotion": 2,
            "annes_sous_responsable_actuel": 2,
            "satisfaction_employee_environnement": 3,
            "note_evaluation_precedente": 3.5,
            "niveau_hierarchique_poste": 2,
            "satisfaction_employee_nature_travail": 4,
            "satisfaction_employee_equipe": 3,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "note_evaluation_actuelle": 4.0,
            "heures_supplementaires": 0,
            "augementation_salaire_precedente": 0.07,
        },
        {
            "id_employee": 9999,
            "age": 28,
            "genre": 0,
            "revenu_mensuel": 4500.0,
            "statut_marital": "Célibataire",
            "departement": "Ventes",
            "poste": "Commercial",
            "nombre_experiences_precedentes": 1,
            "annee_experience_totale": 4,
            "annees_dans_l_entreprise": 2,
            "annees_dans_le_poste_actuel": 2,
            "nombre_participation_pee": 0,
            "nb_formations_suivies": 1,
            "nombre_employee_sous_responsabilite": 0,
            "distance_domicile_travail": 5,
            "niveau_education": 2,
            "domaine_etude": "Marketing",
            "ayant_enfants": 0,
            "frequence_deplacement": "Fréquemment",
            "annees_depuis_la_derniere_promotion": 1,
            "annes_sous_responsable_actuel": 1,
            "satisfaction_employee_environnement": 4,
            "note_evaluation_precedente": 3.0,
            "niveau_hierarchique_poste": 1,
            "satisfaction_employee_nature_travail": 3,
            "satisfaction_employee_equipe": 4,
            "satisfaction_employee_equilibre_pro_perso": 4,
            "note_evaluation_actuelle": 3.5,
            "heures_supplementaires": 1,
            "augementation_salaire_precedente": 0.05,
        },
    ]
}

if __name__ == "__main__":
    logger.info("Starting end-to-end type debugging...")

    # Simulate FastAPI's initial DataFrame creation
    employees_data_list = json_payload["employees"]
    input_df = pd.DataFrame(employees_data_list)
    logger.info("--- dtypes after initial DataFrame creation (input_df) ---")
    logger.info(input_df.dtypes.to_string())

    # Simulate id_employee handling
    if "id_employee" not in input_df.columns or input_df["id_employee"].isnull().all():
        input_df["id_employee"] = range(1, len(input_df) + 1)
    else:
        missing_id_mask = input_df["id_employee"].isna()
        if missing_id_mask.any():
            # For this debug script, we'll just fill with a placeholder if missing
            # In API, it would query DB for max_existing_id
            input_df.loc[missing_id_mask, "id_employee"] = -1  # Placeholder
    input_df["id_employee"] = input_df["id_employee"].astype(int)
    logger.info("--- dtypes after id_employee handling ---")
    logger.info(input_df.dtypes.to_string())

    # Simulate clean_and_engineer_features
    processed_data = clean_and_engineer_features(input_df.copy())
    logger.info("--- dtypes after clean_and_engineer_features (processed_data) ---")
    logger.info(processed_data.dtypes.to_string())

    # Simulate coerce_and_align
    final_data_for_prediction = coerce_and_align(processed_data)
    logger.info("--- dtypes after coerce_and_align (final_data_for_prediction) ---")
    logger.info(final_data_for_prediction.dtypes.to_string())

    logger.info("End of debugging script.")
