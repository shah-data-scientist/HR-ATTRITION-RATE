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

    # Ensure x_transformed_for_shap is a DataFrame for consistent indexing
    if not isinstance(x_transformed_for_shap, pd.DataFrame):
        x_transformed_for_shap = pd.DataFrame(
            x_transformed_for_shap, columns=st.session_state.all_features
        )
    
    # Store processed data for SHAP
    st.session_state.processed_data_for_shap = x_transformed_for_shap

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


def display_confusion_matrix_and_metrics(y_true, y_proba, threshold, title):
    """
    Displays a confusion matrix with row-normalized percentages.
    Returns accuracy, recall, and the count of flagged employees.
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate confusion matrix components
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Calculate row-normalized percentages
    cm_normalized = np.zeros_like(cm, dtype=float)
    row_sums = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        if row_sums[i] > 0:
            cm_normalized[i, :] = cm[i, :] / row_sums[i]

    st.subheader(title)
    fig, ax = plt.subplots(figsize=(4, 3)) # Smaller figure size for visual proportionality
    sns.heatmap(cm_normalized, annot=True, fmt=".1%", cmap='Greens', cbar=False,
                xticklabels=['Predicted Stay', 'Predicted Leave'],
                yticklabels=['Actual Stay', 'Actual Leave'], ax=ax,
                annot_kws={"size": 10}) # Smaller annotation font size
    ax.set_ylabel('Actual Outcome')
    ax.set_xlabel('Predicted Outcome')
    st.pyplot(fig, use_container_width=True) # Ensure responsiveness
    plt.close(fig) # Close the plot to prevent it from displaying twice

    # Calculate Accuracy and Recall for external display
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Count of employees flagged as 'Leave' (True Positives + False Positives)
    flagged_count = tp + fp

    return accuracy, recall, flagged_count


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

def render_threshold_block():
    """
    Renders the confusion matrix with a threshold slider, Accuracy and Recall metrics,
    workload estimation, and an HR-friendly explanation.
    """
    st.header("Adjust Risk Threshold & Review Impact")

    col1, col2 = st.columns([0.7, 0.3]) # Reverted column widths

    with col2:
        threshold = st.slider(
            "Set 'High Risk' Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Adjust this slider to change the probability score at which an employee is flagged as 'High Risk' (predicted to leave)."
        )
        st.session_state.current_threshold = threshold # Store threshold in session state

    with col1:
        # Assuming y_train_loaded and train_prediction_proba are in st.session_state
        y_true = st.session_state.y_train_loaded
        y_proba = st.session_state.train_prediction_proba
        
        # Helper function call
        accuracy, recall, flagged_count = display_confusion_matrix_and_metrics(
            y_true, y_proba, threshold, "Prediction Accuracy Overview"
        )
        
        st.markdown(f"**Overall Correct Predictions:** {accuracy:.1%}")
        st.markdown(f"**Correctly Identified 'Leave' Cases:** {recall:.1%}")
        st.markdown(f"Estimated Review Workload: ~{flagged_count} employees flagged at this threshold.")

    st.markdown("""
        Understanding how our model predicts employee attrition is key. The 'Risk Score' (probability) tells us how likely an employee is to leave. We use a 'threshold' to decide when a score is high enough to flag an employee as 'High Risk' (predicted to leave).

        *   **Overall Correct Predictions (Accuracy):** This shows the percentage of all employees (both those who stay and those who leave) that our model predicted correctly. A high number here means the model is generally good at its job.
        *   **Correctly Identified 'Leave' Cases (Recall):** This is crucial for proactive HR. It tells us, out of all the employees who *actually* left, what percentage our model successfully flagged as 'High Risk'. A high Recall means we're good at catching potential leavers.

        Adjusting the 'High Risk' Threshold changes how many employees are flagged. A lower threshold means we flag more employees, increasing our 'Recall' (catching more potential leavers) but potentially also flagging more employees who would have stayed. A higher threshold flags fewer employees, reducing the 'workload' but risking missing some who might leave. It's a balance between catching all potential leavers and managing the number of employees HR needs to review.
    """)

    st.subheader("Threshold Examples:")
    st.markdown("""
        *   **Threshold 0.30 (More Proactive):** At a threshold of 0.30, the model is very sensitive. It flags more employees as 'High Risk', aiming to catch almost everyone who might leave. This means HR will review a larger group, ensuring fewer potential leavers are missed, but some flagged employees might have actually stayed.
        *   **Threshold 0.50 (Balanced Approach):** With a threshold of 0.50, the model takes a balanced approach. It flags employees with a 50% or higher risk score. This provides a good balance between identifying potential leavers and keeping the review workload manageable for HR.
        *   **Threshold 0.70 (More Conservative):** Using a threshold of 0.70, the model is more conservative. It only flags employees with a very high risk score. This significantly reduces the number of employees HR needs to review, focusing only on the most critical cases, but it might miss some employees who eventually leave.
    """)


def render_employee_overview_and_shap():
    """
    Renders a sortable employee table, a select box for deep-diving into SHAP,
    the SHAP waterfall plot, and a summary of top-3 drivers.
    Includes a download button for a comprehensive Excel report.
    """
    st.header("Employee Risk Overview")

    # Ensure report_data is available
    if 'report_data' not in st.session_state or st.session_state.report_data.empty:
        st.warning("No employee data available for overview. Please ensure data is loaded.")
        return

    df_display = st.session_state.report_data.copy()
    df_display = df_display.rename(columns={
        "id_employee": "Employee ID",
        "Attrition_Risk_Percentage": "Risk Score",
        "Prediction": "Prediction",
        "Risk_Attrition": "Risk Level" # Added Risk Level
    })
    
    # Format Risk Score as percentage
    df_display["Risk Score"] = df_display["Risk Score"].apply(lambda x: f"{x:.1%}")

    # Display sortable table
    st.dataframe(df_display[["Employee ID", "Risk Score", "Prediction", "Risk Level"]].sort_values(by="Risk Score", ascending=False), use_container_width=True)

    # --- Download Excel Report Button ---
    if st.session_state.prediction_triggered and st.session_state.report_data is not None:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            # Tab 1: Summary
            tab1_df = st.session_state.report_data[
                [
                    "id_employee",
                    "Risk_Attrition",
                    "Attrition_Risk_Percentage",
                    "Prediction",
                ]
            ].copy()
            tab1_df.rename(columns={"id_employee": "Employee_ID"}, inplace=True)
            tab1_df.to_excel(writer, sheet_name="Summary", index=False)

            # Tab 2: Features (all features with coefficients)
            # Need to merge report_data with excel_tab2_data to get all info
            if 'excel_report_data' in st.session_state and st.session_state.excel_report_data is not None:
                # Assuming excel_report_data already contains Employee_ID, Feature, Coefficient, Prediction
                tab2_df = st.session_state.excel_report_data.copy()
                tab2_df.to_excel(writer, sheet_name="Features", index=False)
            else:
                st.warning("SHAP feature data not available for Excel export.")
                pd.DataFrame({"Message": ["SHAP feature data not available"]}).to_excel(writer, sheet_name="Features", index=False)

            # Tab 3: Metrics (optional)
            summary_metrics_df = pd.DataFrame(
                {
                    "Metric": [
                        "Total Employees Processed",
                        "Predicted to Leave",
                        "Predicted to Stay",
                    ],
                    "Value": [
                        len(st.session_state.report_data),
                        st.session_state.report_data["Prediction"].value_counts().get("Leave", 0),
                        st.session_state.report_data["Prediction"].value_counts().get("Stay", 0),
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
            help="Download a detailed Excel report including predictions and feature contributions for all employees."
        )
    
    st.subheader("Deep Dive: Individual Employee Analysis")
    
    employee_ids = df_display["Employee ID"].unique() # Use renamed column
    selected_employee_id = st.selectbox(
        "Select Employee for Detail",
        options=employee_ids,
        help="Choose an employee ID to see a detailed breakdown of their attrition risk factors."
    )

    if selected_employee_id:
        st.subheader(f"Key Drivers for Employee {selected_employee_id}")

        try:
            # Get data for the selected employee
            employee_row_index = st.session_state.report_data[st.session_state.report_data["id_employee"] == selected_employee_id].index[0]
            
            # Ensure explainer and processed_data_for_shap are available
            if 'explainer' not in st.session_state or st.session_state.explainer is None:
                st.error("SHAP explainer not found in session state. Please ensure the model is loaded and predictions are made.")
                return
            if 'processed_data_for_shap' not in st.session_state or st.session_state.processed_data_for_shap.empty:
                st.error("Processed data for SHAP not found in session state. Cannot display SHAP plot.")
                return

            employee_shap_values = st.session_state.explainer.shap_values(st.session_state.processed_data_for_shap.iloc[employee_row_index])
            
            # For binary classification, shap_values returns a list of two arrays. We need the shap values for the 'Leave' class (index 1).
            shap_values_for_plot = employee_shap_values[1] if isinstance(employee_shap_values, list) else employee_shap_values

            # SHAP Waterfall Plot
            fig, ax = plt.subplots()
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_for_plot,
                    base_values=st.session_state.explainer.expected_value[1] if isinstance(st.session_state.explainer.expected_value, np.ndarray) else st.session_state.explainer.expected_value,
                    data=st.session_state.processed_data_for_shap.iloc[employee_row_index],
                    feature_names=st.session_state.all_features
                ),
                max_display=10, # Display top 10 features
                show=False, # Prevent matplotlib from showing the plot immediately
                # Removed ax=ax
            )
            st.pyplot(fig, bbox_inches='tight') # Display the plot in Streamlit
            plt.close(fig) # Close the plot to prevent it from displaying twice

            st.subheader("Top Reasons for Risk")
            # Get top 3 drivers in plain English
            
            # Create a series of SHAP values with feature names
            shap_series = pd.Series(shap_values_for_plot, index=st.session_state.all_features)
            
            # Sort by absolute SHAP value to find most impactful features
            top_drivers = shap_series.abs().sort_values(ascending=False).head(3)

            for feature_name in top_drivers.index:
                impact_value = shap_series[feature_name]
                direction = "increasing" if impact_value > 0 else "decreasing"
                # Placeholder for HR-friendly feature names - can be expanded with a mapping dictionary
                hr_feature_name = feature_name.replace("_", " ").title() 
                st.markdown(f"- **{hr_feature_name}:** This factor is **{direction}** the likelihood of attrition.")
        except Exception as e:
            st.error(f"Error retrieving detailed analysis for Employee {selected_employee_id}: {e}")
            st.exception(e) # Log the exception for debugging

# --- End of STREAMLIT CODE ---

def main() -> None:
    """Run the Streamlit application."""
    _setup_app()  # Ensure global variables are initialized
    # --- Streamlit App Layout ---
    st.set_page_config(layout="wide")
    st.title("Employee Attrition Risk")

    # Call the new rendering functions
    render_threshold_block()

    st.markdown("---")

    _handle_file_uploads_and_predict(st.session_state.get('current_threshold', 0.5)) # Use current_threshold from slider

    # --- Display Results (if triggered) ---
    if st.session_state.prediction_triggered:
        st.markdown("---")
        render_employee_overview_and_shap()

        st.success("Reports generated successfully!")


if __name__ == "__main__":
    main()