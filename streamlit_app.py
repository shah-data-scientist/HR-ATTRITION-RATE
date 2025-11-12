import matplotlib

matplotlib.use('Agg') # Use non-interactive backend for matplotlib
import pandas as pd
import streamlit as st

pd.set_option("future.no_silent_downcasting", True)  # This was inserted here
import base64  # Import base64 for embedding images
import io
import sys  # Import sys
from datetime import UTC, datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Import seaborn for enhanced plotting
import shap
import streamlit.components.v1 as components
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: float) -> float:
    """Inverse of sigmoid function."""
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    return np.log(p / (1 - p))


# --- Configuration ---
# Define risk categories for Excel/HTML reports (probability-based, for confusion matrix)
RISK_THRESHOLDS = {"Low": (0.0, 0.3), "Medium": (0.3, 0.7), "High": (0.7, 1.0)}

# Define risk categories for log-odds (f(x))
# Corresponding log-odds for probabilities 0.3 and 0.7
# logit(0.3) approx -0.847
# logit(0.7) approx 0.847
LOG_ODDS_RISK_THRESHOLDS = {
    "Low": (-np.inf, _logit(0.3)),  # f(x) < -0.847
    "Medium": (_logit(0.3), _logit(0.7)),  # -0.847 <= f(x) < 0.847
    "High": (_logit(0.7), np.inf),  # f(x) >= 0.847
}

REQUIRED_FILES = ["extrait_eval.csv", "extrait_sirh.csv", "extrait_sondage.csv"]

# --- Session State Initialization ---
if "prediction_triggered" not in st.session_state:
    st.session_state.prediction_triggered = False
if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "shap_html_content" not in st.session_state:
    st.session_state.shap_html_content = None
if "excel_report_data" not in st.session_state:
    st.session_state.excel_report_data = None
if "processed_data_for_shap" not in st.session_state:
    st.session_state.processed_data_for_shap = None
if "explainer" not in st.session_state:
    st.session_state.explainer = None
if "all_features" not in st.session_state:
    st.session_state.all_features = None


def _load_local_csv_files():
    """Loads the required CSV files from the local 'data' directory."""
    try:
        eval_file_path = "data/extrait_eval.csv"
        sirh_file_path = "data/extrait_sirh.csv"
        sondage_file_path = "data/extrait_sondage.csv"

        # Read files into BytesIO objects to mimic uploaded files
        with open(eval_file_path, "rb") as f:
            eval_file = io.BytesIO(f.read())
            eval_file.name = "extrait_eval.csv"
        with open(sirh_file_path, "rb") as f:
            sirh_file = io.BytesIO(f.read())
            sirh_file.name = "extrait_sirh.csv"
        with open(sondage_file_path, "rb") as f:
            sondage_file = io.BytesIO(f.read())
            sondage_file.name = "extrait_sondage.csv"

        return eval_file, sirh_file, sondage_file
    except FileNotFoundError as e:
        st.error(f"Required data file not found: {e}. Please ensure 'data' directory "
                 "contains 'extrait_eval.csv', 'extrait_sirh.csv', and 'extrait_sondage.csv'.")
        return None, None, None


def _handle_file_uploads_and_predict(main_threshold: float) -> None:
    st.subheader("Upload Employee Data for Prediction")
    uploaded_files = st.file_uploader(
        "Upload three CSV files: `extrait_eval.csv`, "
        "`extrait_sirh.csv`, `extrait_sondage.csv`",
        type=["csv"],
        accept_multiple_files=True,
    )

    eval_file, sirh_file, sondage_file = None, None, None
    files_source = "uploaded" # To track if files came from uploader or local

    if uploaded_files:
        if len(uploaded_files) != len(REQUIRED_FILES):
            st.warning("Please upload all three required CSV files.")
            return

        uploaded_file_names = [file.name for file in uploaded_files]
        if not all(name in uploaded_file_names for name in REQUIRED_FILES):
            st.warning(
                "Please make sure to upload the three required files: "
                "`extrait_eval.csv`, `extrait_sirh.csv`, and `extrait_sondage.csv`."
            )
            return

        file_map = {file.name: file for file in uploaded_files}
        eval_file = file_map.get("extrait_eval.csv")
        sirh_file = file_map.get("extrait_sirh.csv")
        sondage_file = file_map.get("extrait_sondage.csv")
    else:
        # If no files uploaded, try to load from local data directory for testing
        st.info("No files uploaded. Attempting to load from local 'data' directory for testing.")
        eval_file, sirh_file, sondage_file = _load_local_csv_files()
        files_source = "local"

    if eval_file and sirh_file and sondage_file:
        predict_button = st.button("Predict Attrition")

        if predict_button:
            processed_df, merged_df = _load_and_process_data(
                eval_file, sirh_file, sondage_file
            )
            _process_predictions_and_reports(
                processed_df, merged_df, main_threshold
            )
        elif files_source == "uploaded":
            st.info("Please upload the CSV files to get started.")
        elif files_source == "local":
            st.info("Local files loaded. Click 'Predict Attrition' to proceed.")
    else:
        st.error("Could not load required CSV files.")


def _load_and_process_data(
    eval_file: io.BytesIO,
    sirh_file: io.BytesIO,
    sondage_file: io.BytesIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, merge, and clean the data."""
    eval_df = pd.read_csv(eval_file)
    sirh_df = pd.read_csv(sirh_file)
    sondage_df = pd.read_csv(sondage_file)
    merged_df = load_and_merge_data(eval_df, sirh_df, sondage_df)
    processed_df = clean_and_engineer_features(merged_df.copy())
    return processed_df, merged_df


def _process_predictions_and_reports(
    processed_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    main_threshold: float,
) -> None:
    """Process uploaded files, make predictions, and generate reports."""
    # Ensure all expected columns are present; fill missing with 0.
    expected_cols = get_expected_columns(st.session_state.model)
    missing_cols = set(expected_cols) - set(processed_df.columns)
    for c in missing_cols:
        processed_df[c] = 0  # Or other appropriate default value

    # Align columns and order
    processed_df_aligned = processed_df[expected_cols]

    # Get categorical columns from the preprocessor
    preprocessor = st.session_state.model.named_steps["preprocessor"]
    cat_cols_from_model = preprocessor.transformers_[1][2]

    # Convert categorical columns to string
    for col in cat_cols_from_model:
        if col in processed_df_aligned.columns:
            processed_df_aligned.loc[:, col] = processed_df_aligned[col].astype(str)

    # Transform data for SHAP
    x_transformed_for_shap = st.session_state.model.named_steps["preprocessor"].transform(
        processed_df_aligned
    )

    # Store processed data for SHAP
    st.session_state.processed_data_for_shap = x_transformed_for_shap

    # Ensure x_transformed_for_shap is a DataFrame for consistent indexing
    if not isinstance(x_transformed_for_shap, pd.DataFrame):
        x_transformed_for_shap = pd.DataFrame(
            x_transformed_for_shap, columns=st.session_state.all_features
        )

    # Make predictions
    predictions_proba = st.session_state.model.predict_proba(processed_df_aligned)[:, 1]
    predictions_class = (predictions_proba >= main_threshold).astype(int)

    # Generate SHAP explainer and values
    explainer = shap.LinearExplainer(st.session_state.model.named_steps["model"], x_transformed_for_shap)
    st.session_state.explainer = explainer

    # Create report data
    report_data = merged_df[["id_employee"]].copy()
    report_data["Attrition_Risk_Percentage"] = predictions_proba
    report_data["Prediction"] = np.where(predictions_class == 1, "Leave", "Stay")
    report_data["Risk_Attrition"] = report_data["Attrition_Risk_Percentage"].apply(
        lambda x: get_risk_category(x, main_threshold)
    )

    st.session_state.report_data = report_data

    # Generate SHAP report data
    shap_report_data = generate_shap_report_data(
        report_data,
        x_transformed_for_shap,
        explainer,
        st.session_state.all_features,
    )
    st.session_state.shap_report_data = shap_report_data

    # Prepare data for Excel Tab 2 (Features)
    excel_tab2_data = []
    for idx, employee_id in enumerate(report_data["id_employee"]):
        shap_values_row = explainer.shap_values(x_transformed_for_shap.iloc[idx])
        # Ensure shap_values_row is 1-dimensional
        if isinstance(shap_values_row, np.ndarray) and shap_values_row.ndim > 1:
            shap_values_row = shap_values_row.flatten()
        elif (
            isinstance(shap_values_row, list)
            and len(shap_values_row) == 1
            and isinstance(shap_values_row[0], np.ndarray)
        ):
            shap_values_row = shap_values_row[0].flatten()

        # Create a DataFrame for SHAP values and features for this employee
        employee_shap_df = pd.DataFrame(
            {
                "Feature": st.session_state.all_features,
                "Coefficient": shap_values_row,
            }
        )
        employee_shap_df["Employee_ID"] = employee_id
        employee_shap_df["Prediction"] = report_data.loc[
            report_data["id_employee"] == employee_id, "Prediction"
        ].iloc[0]
        excel_tab2_data.append(employee_shap_df)

    st.session_state.excel_report_data = pd.concat(excel_tab2_data)

    st.session_state.prediction_triggered = True


def clear_prediction_results() -> None:
    """Clear all prediction results and reset session state."""
    st.session_state.prediction_triggered = False
    st.session_state.report_data = None
    st.session_state.shap_report_data = None
    st.session_state.excel_report_data = None
    st.session_state.processed_data_for_shap = None
    st.session_state.explainer = None
    st.session_state.all_features = None


# --- Load Model and Data ---
@st.cache_resource
def load_model_and_data():
    """Load the trained model and test/train data."""
    model = joblib.load("outputs/employee_attrition_pipeline.pkl")
    x_train_loaded = pd.read_parquet("outputs/X_train.parquet")
    y_train_loaded = pd.read_parquet("outputs/y_train.parquet").squeeze()
    x_test_loaded = pd.read_parquet("outputs/X_test.parquet")
    y_test_loaded = pd.read_parquet("outputs/y_test.parquet").squeeze()
    return model, x_train_loaded, y_train_loaded, x_test_loaded, y_test_loaded


# Function to set up the app's global resources
def _setup_app() -> None:
    if "model" not in st.session_state:
        (
            st.session_state.model,
            st.session_state.x_train_loaded,
            st.session_state.y_train_loaded,
            st.session_state.x_test_loaded,
            st.session_state.y_test_loaded,
        ) = load_model_and_data()
        # Initialize all_features here, after the model is loaded
        st.session_state.all_features = st.session_state.model.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    if "train_prediction_proba" not in st.session_state:
        # Pre-calculate probabilities for training data (for confusion matrix)
        st.session_state.train_prediction_proba = st.session_state.model.predict_proba(
            st.session_state.x_train_loaded
        )[:, 1]


# Call setup function only if not in a test environment (or if Streamlit is running)
# This check is a common pattern to prevent global execution during imports in tests
if "streamlit" in sys.modules and "pytest" not in sys.modules:
    _setup_app()


# --- Helper Functions (from train.py) ---
def _clean_extrait_eval(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "augementation_salaire_precedente" in df.columns:
        df.rename(
            columns={
                "augementation_salaire_precedente": "augmentation_salaire_precedente"
            },
            inplace=True,
        )
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
    # Harmonize different column names for "heures_supplementaires"
    heures_sup_cols = [
        "heures_supplementaires",
        "heure_supplementaires",
        "heures_supplémentaires",
    ]
    for col in heures_sup_cols:
        if col in df.columns and col != "heures_supplementaires":
            df.rename(columns={col: "heures_supplementaires"}, inplace=True)

    if "heures_supplementaires" in df.columns:
        df["heures_supplementaires"] = (
            df["heures_supplementaires"]
            .replace({"Oui": 1, "Non": 0, "oui": 1, "non": 0, True: 1, False: 0})
            .astype("Int64")
        )
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
    df = df.copy()
    if "genre" in df.columns:
        df["genre"] = df["genre"].str.lower()
        df["genre"] = (
            df["genre"].replace({"m": 1, "f": 0}).astype("Int64")
        )
    for col in ["nombre_heures_travailless"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df


def _clean_extrait_sondage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "code_sondage" in df.columns:
        df.rename(columns={"code_sondage": "id_employee"}, inplace=True)
    if "id_employee" in df.columns:
        df["id_employee"] = pd.to_numeric(df["id_employee"], errors="coerce").astype(
            "Int64"
        )
    if "annes_sous_responsable_actuel" in df.columns:
        df.rename(
            columns={"annes_sous_responsable_actuel": "annees_sous_responsable_actuel"},
            inplace=True,
        )
    return df


def load_and_merge_data(
    eval_df: pd.DataFrame, sirh_df: pd.DataFrame, sond_df: pd.DataFrame
) -> pd.DataFrame:
    """Load and merge employee data from evaluation, SIRH, and survey dataframes."""
    eval_df = _clean_extrait_eval(eval_df)
    sirh_df = _clean_extrait_sirh(sirh_df)
    sond_df = _clean_extrait_sondage(sond_df)

    # The _clean_ functions are expected to ensure 'id_employee' is present.
    # If not, the merge will handle missing keys.

    merged = eval_df.merge(
        sirh_df, on="id_employee", how="outer", suffixes=("_eval", "_sirh")
    )
    merged = merged.merge(sond_df, on="id_employee", how="outer")
    if "..." in merged.columns:
        merged.drop(columns=["..."], inplace=True, errors="ignore")
    merged.drop_duplicates(inplace=True)
    return merged


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning and feature engineering steps as in the notebook."""
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

    # Dynamically get categorical columns from the fitted preprocessor and convert them to string type
    # This prevents TypeError in OneHotEncoder due to mixed types (int/str)
    preprocessor = st.session_state.model.named_steps["preprocessor"]
    # The categorical transformer is usually the second one in the ColumnTransformer
    # (name, transformer_object, column_names_list)
    cat_cols_from_model = preprocessor.transformers_[1][2]

    for col in df.columns:
        if col in cat_cols_from_model:
            df[col] = df[col].astype(str)

    return df


def get_expected_columns(pipeline: Pipeline) -> list[str]:
    """Get the list of columns the model was trained on."""
    # Use the columns from the loaded X_train to ensure consistency
    return st.session_state.x_train_loaded.columns.tolist()


def _get_risk_category_from_log_odds(log_odds: float) -> str:
    """Categorize attrition risk based on log-odds (f(x)) using predefined thresholds."""
    low_threshold = LOG_ODDS_RISK_THRESHOLDS["Low"][1]  # Upper bound of Low
    high_threshold = LOG_ODDS_RISK_THRESHOLDS["High"][0]  # Lower bound of High

    if log_odds < low_threshold:
        return "Low"
    if log_odds >= high_threshold:
        return "High"
    return "Medium"


def get_risk_category(probability: float, threshold: float) -> str:
    """Categorize attrition risk based on probability and a dynamic threshold."""
    if probability >= threshold + 0.1:
        return "High"
    if probability <= threshold - 0.1:
        return "Low"
    return "Medium"


def display_confusion_matrix_and_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    title: str = "Confusion Matrix",
) -> dict:
    """Display confusion matrix with row-normalized percentages and classification metrics."""
    predictions = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, predictions)

    # Calculate row-normalized percentages
    # Handle division by zero if a row sum is 0
    cm_normalized = np.zeros_like(cm, dtype=float)
    row_sums = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        if row_sums[i] > 0:
            cm_normalized[i, :] = cm[i, :] / row_sums[i]

    st.subheader(title)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(
            figsize=(6, 6)
        )  # Increased figure size slightly for better readability

        # Use seaborn.heatmap for better control over annotations and appearance
        # Annotate with row-normalized percentages, formatted to 2 decimal places
        # Note: Overlap prevented by fixed format and sufficient figure size.

        # seaborn's annotation placement is generally robust for single values per cell.
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Stay", "Leave"],
            yticklabels=["Stay", "Leave"],
            linewidths=0.5,
            linecolor="black",
            annot_kws={"size": 12, "color": "black"},
        )  # Added for single, readable label

        ax.set_xlabel("Prediction", fontsize=14)  # Increased font size for clarity
        ax.set_ylabel("Truth", fontsize=14)  # Increased font size for clarity
        ax.set_title(
            "Confusion Matrix (Row-Normalized Percentages)", fontsize=16
        )  # Added a more descriptive title
        plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better fit
        plt.tight_layout()  # Ensure no overlapping text or elements
        st.pyplot(fig)
        plt.close(fig)  # Explicitly close the figure to prevent overlap
    with col2:
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)

        st.write(f"**Threshold: {threshold:.2f}**")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(
            f"Precision: {precision:.2f}"
        )
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }


def generate_shap_report_data(
    employee_data_with_predictions: pd.DataFrame,
    x_transformed_for_shap: pd.DataFrame,
    explainer: shap.TreeExplainer,
    all_features: list[str],
) -> list[dict]:
    """Generate SHAP waterfall plot data as base64 encoded images.

    Returns a list of dictionaries, each containing employee details and a
    base64 encoded PNG image of the SHAP waterfall plot.
    """
    shap_report_items = []

    if not isinstance(x_transformed_for_shap, pd.DataFrame):
        x_transformed_for_shap = pd.DataFrame(
            x_transformed_for_shap,
            columns=all_features,
        )

    all_shap_values = explainer(x_transformed_for_shap)

    for i, (_, row) in enumerate(employee_data_with_predictions.iterrows()):
        employee_id = row.get("id_employee", f"Employee {i+1}")
        risk_category = row["Risk_Attrition"]
        attrition_prob = row["Attrition_Risk_Percentage"]
        prediction_type = row["Prediction"]

        shap_values_row = all_shap_values[i]

        shap.plots.waterfall(
            shap_values_row,
            max_display=10,
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        shap_report_items.append(
            {
                "employee_id": employee_id,
                "risk_category": risk_category,
                "attrition_prob": attrition_prob,
                "prediction_type": prediction_type,
                "img_str": img_str,
            }
        )
    return shap_report_items
def _display_threshold_impact() -> None:
    """Display confusion matrices for different thresholds on training data.

    Displays confusion matrices for different thresholds on training data.
    """
    st.subheader("Understanding Threshold Impact (examples from training data)")
    st.write(
        "These examples illustrate how different thresholds affect the model's "
        "classification performance on the training data."
    )

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        metrics_t1 = display_confusion_matrix_and_metrics(
            st.session_state.y_train_loaded,
            st.session_state.train_prediction_proba,
            0.3,
            "Threshold 0.30",
        )
    with col_t2:
        metrics_t2 = display_confusion_matrix_and_metrics(
            st.session_state.y_train_loaded,
            st.session_state.train_prediction_proba,
            0.5,
            "Threshold 0.50 (Default)",
        )
    with col_t3:
        metrics_t3 = display_confusion_matrix_and_metrics(
            st.session_state.y_train_loaded,
            st.session_state.train_prediction_proba,
            0.7,
            "Threshold 0.70",
        )
    st.markdown("---")


def main() -> None:
    """Run the Streamlit application."""
    _setup_app()  # Ensure global variables are initialized
    # --- Streamlit App Layout ---
    st.set_page_config(layout="wide")
    st.title("Employee Attrition Risk")

    _display_threshold_impact()

    # --- Threshold Slider ---
    st.subheader("Adjust Prediction Threshold")
    main_threshold = st.slider(
        "Select Probability Threshold",
        0.0,
        1.0,
        0.5,
        0.01,
        help="Adjust this threshold to see how it impacts the model's "
        "classification on the training data.",
    )

    # --- Confusion Matrix (Live Update) ---
    metrics_live = display_confusion_matrix_and_metrics(
        st.session_state.y_train_loaded,
        st.session_state.train_prediction_proba,
        main_threshold,
        title=(
            f"Live Confusion Matrix on Training Data (Threshold: {main_threshold:.2f})"
        ),
    )

    st.markdown("---")

    _handle_file_uploads_and_predict(main_threshold)

    # --- Display Results (if triggered) ---
    if st.session_state.prediction_triggered:
        st.markdown("---")
        st.subheader("Prediction Results and Reports")

        report_data = st.session_state.report_data
        x_transformed_for_shap = st.session_state.processed_data_for_shap
        explainer = st.session_state.explainer
        all_features = st.session_state.all_features
        excel_tab2_data = st.session_state.excel_report_data

        # --- Generate Excel Report ---
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            # Tab 1: Summary (no employee name, no extra columns)
            tab1_df = report_data[
                [
                    "id_employee",
                    "Risk_Attrition",
                    "Attrition_Risk_Percentage",
                    "Prediction",
                ]
            ].copy()
            tab1_df.rename(columns={"id_employee": "Employee_ID"}, inplace=True)
            tab1_df.to_excel(writer, sheet_name="Summary", index=False)

            # Tab 2: Features (all features with coefficients; no employee name)
            tab2_df = excel_tab2_data.copy()
            # Ensure column names are exactly as required
            tab2_df.rename(
                columns={
                    "Employee_ID": "Employee_ID",
                    "Feature": "Feature",
                    "Coefficient": "Coefficient",
                },
                inplace=True,
            )
            tab2_df[["Employee_ID", "Feature", "Coefficient", "Prediction"]].to_excel(
                writer, sheet_name="Features", index=False
            )

            # Tab 3: Metrics (optional)
            summary_metrics_df = pd.DataFrame(
                {
                    "Metric": [
                        "Total Employees Processed",
                        "Predicted to Leave",
                        "Predicted to Stay",
                    ],
                    "Value": [
                        len(report_data),
                        report_data["Prediction"].value_counts().get("Leave", 0),
                        report_data["Prediction"].value_counts().get("Stay", 0),
                    ],
                }
            )
            summary_metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

        excel_buffer.seek(0)
        st.download_button(
            label="Download Excel Report",
            data=excel_buffer,
            file_name="employee_attrition_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # --- Display SHAP Visualization Report ---
        st.subheader("Employee Attrition SHAP Explanations")
        shap_report_items = st.session_state.shap_report_data
        if shap_report_items:
            for item in shap_report_items:
                st.markdown(f"### Employee ID: {item['employee_id']}")
                st.markdown(
                    f"Predicted Attrition Risk: **{item['risk_category']}** "
                    f"({item['attrition_prob']:.1%}) · Prediction: "
                    f"**{item['prediction_type']}**"
                )
                st.image(
                    f"data:image/png;base64,{item['img_str']}",
                    caption=f"SHAP Waterfall Plot for Employee {item['employee_id']}",
                    use_container_width=True,
                )
                st.markdown("---")
        else:
            st.info("No SHAP reports generated yet.")

        st.success("Reports generated successfully!")


if __name__ == "__main__":
    main()