import pandas as pd
import numpy as np


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
    for col in ["nombre_heures_travailless", "..."]:
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

    # Clean 'ayant_enfants' column
    if "ayant_enfants" in merged.columns:
        merged["ayant_enfants"] = (
            merged["ayant_enfants"]
            .replace({"Y": 1, "N": 0, "y": 1, "n": 0})
            .astype("Int64")
        )

    # Ensure 'augementation_salaire_precedente' is numeric (handling potential typo)
    if "augementation_salaire_precedente" in merged.columns:
        merged["augementation_salaire_precedente"] = (
            merged["augementation_salaire_precedente"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        merged["augementation_salaire_precedente"] = (
            pd.to_numeric(merged["augementation_salaire_precedente"], errors="coerce")
            / 100.0
        )

    if "..." in merged.columns:
        merged.drop(columns=["..."], inplace=True, errors="ignore")
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


def get_expected_columns(pipeline):
    """Gets the list of columns the model was trained on."""
    preprocessor = pipeline.named_steps["preprocessor"]
    # The feature_names_in_ attribute stores the names of features seen during fit
    return list(preprocessor.feature_names_in_)
