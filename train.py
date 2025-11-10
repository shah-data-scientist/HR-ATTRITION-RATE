import hashlib
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform, uniform
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# --- Global configuration ---
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
FILE_EVAL = DATA_DIR / "extrait_eval.csv"
FILE_SIRH = DATA_DIR / "extrait_sirh.csv"
FILE_SONDAGE = DATA_DIR / "extrait_sondage.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Load & Prepare Data ---
def _clean_extrait_eval(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the 'extrait_eval' DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing evaluation data.

    Returns:
        pd.DataFrame: The cleaned DataFrame.

    """
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


def _clean_extrait_sirh(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the 'extrait_sirh' DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing SIRH data.





    Returns:
        pd.DataFrame: The cleaned DataFrame.


    """
    df = df.copy()
    if "genre" in df.columns:
        df["genre"] = (
            df["genre"].replace({"M": 1, "F": 0, "m": 1, "f": 0}).astype("Int64")
        )
    for col in ["nombre_heures_travailless", "..."]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df


def _clean_extrait_sondage(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the 'extrait_sondage' DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing survey data.





    Returns:
        pd.DataFrame: The cleaned DataFrame.


    """
    df = df.copy()
    if "code_sondage" in df.columns:
        df.rename(columns={"code_sondage": "id_employee"}, inplace=True)
    if "id_employee" in df.columns:
        df["id_employee"] = pd.to_numeric(df["id_employee"], errors="coerce").astype(
            "Int64"
        )
    return df


def load_and_merge_data(


    eval_path: Path, sirh_path: Path, sond_path: Path


) -> pd.DataFrame:
    """Load and merge data from evaluation, SIRH, and survey CSV files.

    Args:
        eval_path (Path): Path to the evaluation CSV file.


        sirh_path (Path): Path to the SIRH CSV file.


        sond_path (Path): Path to the survey CSV file.





    Returns:
        pd.DataFrame: The merged and cleaned DataFrame.


    """
    eval_df = _clean_extrait_eval(pd.read_csv(eval_path))
    sirh_df = _clean_extrait_sirh(pd.read_csv(sirh_path))
    sond_df = _clean_extrait_sondage(pd.read_csv(sond_path))

    merged = eval_df.merge(
        sirh_df, on="id_employee", how="outer", suffixes=("_eval", "_sirh")
    )
    merged = merged.merge(sond_df, on="id_employee", how="outer")
    if "..." in merged.columns:
        merged.drop(columns=["..."], inplace=True, errors="ignore")
    merged.drop_duplicates(inplace=True)
    print("Shape after merge:", merged.shape)
    return merged


consolidated_modelisation = load_and_merge_data(FILE_EVAL, FILE_SIRH, FILE_SONDAGE)

# --- Feature Engineering & Preprocessing ---
df = consolidated_modelisation.copy()

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

target_col = "a_quitte_l_entreprise"
if target_col in df.columns:
    df[target_col] = (
        df[target_col]
        .astype(str)
        .str.strip()
        .replace({"Oui": 1, "Non": 0, "oui": 1, "non": 0, "OUI": 1, "NON": 0})
    )
    df[target_col] = (
        pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    )
else:
    raise ValueError(f"Target column '{target_col}' not found.")

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

num_cols = X.select_dtypes(
    include=["int64", "float64", "Int64", "Float64"]
).columns.tolist()
cat_cols = X.select_dtypes(
    include=["object", "category", "string", "bool"]
).columns.tolist()

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), num_cols), ("cat", ohe, cat_cols)],
    remainder="drop",
)

# --- Modeling ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

logreg_en_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        (
            "model",
            LogisticRegression(solver="saga", max_iter=5000, penalty="elasticnet"),
        ),
    ]
)

param_distributions = {
    "model__C": loguniform(1e-3, 1e2),
    "model__l1_ratio": uniform(0, 1),
    "model__class_weight": [None, "balanced"],
    "smote__k_neighbors": [3, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    estimator=logreg_en_pipe,
    param_distributions=param_distributions,
    n_iter=20,
    scoring={
        "recall": "recall",
        "precision": "precision",
        "pr_auc": "average_precision",
    },
    refit="recall",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    return_train_score=True,
    verbose=0,
)
search.fit(X_train, y_train)
best_pipe = search.best_estimator_
best_params = search.best_params_
print("Best params:", best_params)

# --- Outputs ---
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

snapshot_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + "_p4_attrition"
snap_path_x = SNAPSHOT_DIR / f"{snapshot_id}_X_train.parquet"
snap_path_y = SNAPSHOT_DIR / f"{snapshot_id}_y_train.parquet"
model_path = OUTPUT_DIR / "employee_attrition_pipeline.pkl"
manifest_path = SNAPSHOT_DIR / f"{snapshot_id}_manifest.json"

X_train.to_parquet(snap_path_x, index=False)
y_train.to_frame(name=target_col).to_parquet(snap_path_y, index=False)

joblib.dump(best_pipe, model_path)

schema = {col: str(dtype) for col, dtype in X_train.dtypes.items()}
data_hash = hashlib.md5(
    pd.util.hash_pandas_object(X_train, index=True).values
).hexdigest()

manifest = {
    "snapshot_id": snapshot_id,
    "row_count": int(len(X_train)),
    "feature_count": int(X_train.shape[1]),
    "target": target_col,
    "schema": schema,
    "hash_md5": data_hash,
    "best_params": {
        k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()
    },
    "model_artifact": str(model_path),
}
with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("Saved snapshot + model + manifest.")
print(snap_path_x)
print(snap_path_y)
print(model_path)
print(manifest_path)

# Save the test set for app evaluation
x_test_path = OUTPUT_DIR / "X_test.parquet"
y_test_path = OUTPUT_DIR / "y_test.parquet"
X_test.to_parquet(x_test_path, index=False)
y_test.to_frame(name=target_col).to_parquet(y_test_path, index=False)
print("Saved test set for app evaluation.")
print(x_test_path)
print(y_test_path)

# Save the train set for app evaluation
x_train_path = OUTPUT_DIR / "X_train.parquet"
y_train_path = OUTPUT_DIR / "y_train.parquet"
X_train.to_parquet(x_train_path, index=False)
y_train.to_frame(name=target_col).to_parquet(y_train_path, index=False)
print("Saved train set for app evaluation.")
print(x_train_path)
print(y_train_path)
