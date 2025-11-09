import streamlit as st
import pandas as pd
pd.set_option('future.no_silent_downcasting', True) # This was inserted here
import seaborn as sns # Import seaborn for enhanced plotting
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import sys # Import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import io
from datetime import datetime
import base64 # Import base64 for embedding images
import streamlit.components.v1 as components

# --- Configuration ---
# Define risk categories for Excel report and HTML visualization
RISK_THRESHOLDS = {
    "Low": (0.0, 0.3),
    "Medium": (0.3, 0.7),
    "High": (0.7, 1.0)
}

# --- Session State Initialization ---
if 'prediction_triggered' not in st.session_state:
    st.session_state.prediction_triggered = False
if 'report_data' not in st.session_state:
    st.session_state.report_data = None
if 'shap_html_content' not in st.session_state:
    st.session_state.shap_html_content = None
if 'excel_report_data' not in st.session_state:
    st.session_state.excel_report_data = None
if 'processed_data_for_shap' not in st.session_state:
    st.session_state.processed_data_for_shap = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'all_features' not in st.session_state:
    st.session_state.all_features = None

def clear_prediction_results():
    st.session_state.prediction_triggered = False
    st.session_state.report_data = None
    st.session_state.shap_html_content = None
    st.session_state.excel_report_data = None
    st.session_state.processed_data_for_shap = None
    st.session_state.explainer = None
    st.session_state.all_features = None

# --- Load Model and Data ---
@st.cache_resource
def load_model_and_data():
    """Loads the trained model and test/train data."""
    model = joblib.load('outputs/employee_attrition_pipeline.pkl')
    X_train_loaded = pd.read_parquet('outputs/X_train.parquet')
    y_train_loaded = pd.read_parquet('outputs/y_train.parquet').squeeze()
    X_test_loaded = pd.read_parquet('outputs/X_test.parquet')
    y_test_loaded = pd.read_parquet('outputs/y_test.parquet').squeeze()
    return model, X_train_loaded, y_train_loaded, X_test_loaded, y_test_loaded

# Function to set up the app's global resources
def _setup_app():
    global model, X_train_loaded, y_train_loaded, X_test_loaded, y_test_loaded, train_prediction_proba
    model, X_train_loaded, y_train_loaded, X_test_loaded, y_test_loaded = load_model_and_data()
    # Pre-calculate probabilities for training data (for confusion matrix)
    train_prediction_proba = model.predict_proba(X_train_loaded)[:, 1]

# Call setup function only if not in a test environment (or if Streamlit is running)
# This check is a common pattern to prevent global execution during imports in tests
if "streamlit" in sys.modules and "pytest" not in sys.modules:
    _setup_app()

# --- Helper Functions (from train.py) ---
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
    for col in ['nombre_heures_travailless', '...']:
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
    if '...' in merged.columns:
        merged.drop(columns=['...'], inplace=True, errors='ignore')
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

def get_expected_columns(pipeline):
    """Gets the list of columns the model was trained on."""
    preprocessor = pipeline.named_steps['preprocessor']
    # The feature_names_in_ attribute stores the names of features seen during fit
    return list(preprocessor.feature_names_in_)

def get_risk_category(probability, threshold):
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

def display_confusion_matrix_and_metrics(y_true, y_proba, threshold, title="Confusion Matrix"):
    """
    Displays confusion matrix with row-normalized percentages and classification metrics.
    - Shows ONLY percentages (row-normalized).
    - No overlapping text: renders a single value per cell (2 decimal %).
    - Axis labels: Y-axis: “Truth”, X-axis: “Prediction”.
    """
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
        fig, ax = plt.subplots(figsize=(6, 6)) # Increased figure size slightly for better readability

        # Use seaborn.heatmap for better control over annotations and appearance
        # Annotate with row-normalized percentages, formatted to 2 decimal places
        # Add a brief note in comments explaining how overlap was prevented for the confusion matrix
        # Overlap is prevented by using a fixed format string ('.2%') and ensuring sufficient figure size.
        # seaborn's annotation placement is generally robust for single values per cell.
        sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=['Stay', 'Leave'], yticklabels=['Stay', 'Leave'],
                    linewidths=.5, linecolor='black',
                    annot_kws={"size": 12, "color": "black"}) # Added for single, readable label
        
        ax.set_xlabel("Prediction", fontsize=14) # Increased font size for clarity
        ax.set_ylabel("Truth", fontsize=14) # Increased font size for clarity
        ax.set_title("Confusion Matrix (Row-Normalized Percentages)", fontsize=16) # Added a more descriptive title
        plt.yticks(rotation=0) # Ensure y-axis labels are horizontal
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better fit
        plt.tight_layout() # Ensure no overlapping text or elements
        st.pyplot(fig)
    with col2:
        st.write(f"**Threshold: {threshold:.2f}**")
        st.write(f"Accuracy: {accuracy_score(y_true, predictions):.2f}")
        st.write(f"Precision: {precision_score(y_true, predictions, zero_division=0):.2f}")
        st.write(f"Recall: {recall_score(y_true, predictions, zero_division=0):.2f}")
        st.write(f"F1-Score: {f1_score(y_true, predictions, zero_division=0):.2f}")

def generate_shap_html_report(employee_data_with_predictions, X_transformed_for_shap, explainer, all_features):
    """
    Generates an HTML report with SHAP explanations for each employee.
    - X-axis is "Attrition Probability (%)".
    - Ensures tick labels and axis title are readable and non-overlapping.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Employee Attrition SHAP Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #555; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
            .employee-card {{ border: 1px solid #eee; border-radius: 8px; padding: 15px; margin-bottom: 20px; box-shadow: 2px 2px 8px rgba(0,0,0,0.1); }}
            .risk-label {{ font-weight: bold; padding: 5px 10px; border-radius: 5px; display: inline-block; }}
            .risk-low {{ background-color: #d4edda; color: #155724; }}
            .risk-medium {{ background-color: #fff3cd; color: #856404; }}
            .risk-high {{ background-color: #f8d7da; color: #721c24; }}
            .shap-plot {{ margin-top: 10px; }}
        </style>
    </head>
    <body>
        <h1>Employee Attrition SHAP Explanation Report</h1>
        <p>Report generated on: {report_date}</p>
        <p>This report provides SHAP (SHapley Additive exPlanations) values for each employee, indicating how much each feature contributes to the model's prediction of attrition risk.</p>
    """.format(report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for i, (index, row) in enumerate(employee_data_with_predictions.iterrows()):
        employee_id = row.get('id_employee', f'Employee {i+1}')
        risk_category = row['Risk_Attrition']
        attrition_prob = row['Attrition_Risk_Percentage']
        prediction_type = row['Prediction'] # Get the prediction type (Leaver/Stayer)

        html_content += f"""
        <div class="employee-card">
            <h2>Employee ID: {employee_id}</h2>
            <p>Predicted Attrition Risk: <span class="risk-label risk-{risk_category.lower()}">{risk_category}</span> ({attrition_prob:.2%})</p>
            <p>Prediction Type: <strong>{prediction_type}</strong></p>
        """
        
        # Generate SHAP plot for this employee
        shap_values = explainer.shap_values(X_transformed_for_shap[i:i+1])
        
        # Create a SHAP Explanation object for the current employee
        explanation = shap.Explanation(values=shap_values[0], 
                                      base_values=explainer.expected_value, # Keep original for shap.waterfall_plot
                                      data=X_transformed_for_shap[i], 
                                      feature_names=all_features)
        
        # Plotting the waterfall plot
        # shap.waterfall_plot creates its own figure. We need to capture it and its axes.
        fig = plt.figure(figsize=(12, 7)) # Create a new figure for each plot
        shap.waterfall_plot(explanation, show=False)
        ax = plt.gca() # Get the current axes, which shap.waterfall_plot would have created

        # Hide any text labels that look like "f(x)" or "E[f(x)]" to avoid overlap with the axis/title
        for text in list(ax.texts):
            txt = text.get_text()
            if txt and ("f(x)" in txt or "E[f(x)]" in txt or "E(x)" in txt):
                text.set_visible(False)
        
        # --- Custom SHAP Waterfall Plot for Probability X-axis ---
        # Add a brief note in comments explaining how overlap was prevented for the SHAP X-axis
        # Overlap is prevented by converting log-odds to probability percentages for tick labels,
        # setting a clear x-axis label, and moving E(x)/f(x) to a subtitle.
        # plt.tight_layout() also helps adjust spacing.
        
        # Get base value and f(x) in log-odds space
        base_value_log_odds = explainer.expected_value
        # Sum of SHAP values for this instance
        sum_shap_values_log_odds = np.sum(shap_values[0])
        # Model output (f(x)) in log-odds space
        model_output_log_odds = base_value_log_odds + sum_shap_values_log_odds

        # Convert to probability space
        base_value_prob = 1 / (1 + np.exp(-base_value_log_odds))
        model_output_prob = 1 / (1 + np.exp(-model_output_log_odds))
        
        # Get current x-axis limits and ticks (these are in log-odds)
        xmin, xmax = ax.get_xlim()
        xticks = ax.get_xticks()
        
        # Convert log-odds ticks to probabilities
        # Filter out ticks that result in extreme probabilities (0 or 1) to avoid inf/nan in log-odds conversion
        xtick_probs = []
        xtick_labels_formatted = []
        for x in xticks:
            prob = 1 / (1 + np.exp(-x))
            if 0 < prob < 1: # Ensure probability is not exactly 0 or 1
                xtick_probs.append(prob * 100)
                xtick_labels_formatted.append(f"{prob * 100:.0f}%")
            else: # For extreme values, just use the raw probability or a placeholder
                xtick_probs.append(prob * 100)
                xtick_labels_formatted.append(f"{prob * 100:.0f}%") # Still format as percentage
        
        ax.set_xticks(xticks) # Keep original tick positions
        ax.set_xticklabels(xtick_labels_formatted, fontsize=10) # Apply formatted labels
        ax.set_xlabel("Attrition Probability (%)", fontsize=12)
        
        # Add subtitle for original E(x) and f(x) to avoid overlapping the axis
        ax.set_title(
            f"SHAP Waterfall — Baseline: {base_value_prob:.1%} · Prediction: {model_output_prob:.1%}",
            fontsize=11, pad=12
        )
        
        plt.tight_layout() # Ensure no overlapping text
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight') # Use bbox_inches='tight' to prevent labels from being cut off
        plt.close(fig) # Close the figure to free memory
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        html_content += f"""
            <div class="shap-plot">
                <img src="data:image/png;base64,{img_str}" alt="SHAP Waterfall Plot for Employee {employee_id}">
            </div>
        </div>
        """
    html_content += "</body></html>"
    return html_content

def main():
    _setup_app() # Call the setup function to load model and data

    # --- Streamlit App Layout ---
    st.set_page_config(layout="wide")
    st.title("Employee Attrition Risk")

    # --- Understanding Threshold Impact ---
    st.subheader("Understanding Threshold Impact (examples from training data)")
    st.write("These examples illustrate how different thresholds affect the model's classification performance on the training data.")

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        display_confusion_matrix_and_metrics(y_train_loaded, train_prediction_proba, 0.3, "Threshold 0.30")
    with col_t2:
        display_confusion_matrix_and_metrics(y_train_loaded, train_prediction_proba, 0.5, "Threshold 0.50 (Default)")
    with col_t3:
        display_confusion_matrix_and_metrics(y_train_loaded, train_prediction_proba, 0.7, "Threshold 0.70")

    st.markdown("---")

    # --- Threshold Slider ---
    st.subheader("Adjust Prediction Threshold")
    main_threshold = st.slider(
        "Select Probability Threshold",
        0.0, 1.0, 0.5, 0.01,
        help="Adjust this threshold to see how it impacts the model's classification on the training data."
    )

    # --- Confusion Matrix (Live Update) ---
    display_confusion_matrix_and_metrics(
        y_train_loaded, train_prediction_proba, main_threshold,
        title=f"Live Confusion Matrix on Training Data (Threshold: {main_threshold:.2f})"
    )

    st.markdown("---")

    # --- File Uploads ---
    st.subheader("Upload Employee Data for Prediction")
    st.write("Upload the three required CSV files to get attrition risk predictions.")

    uploaded_files = st.file_uploader(
        "Choose the three CSV files (`extrait_eval.csv`, `extrait_sirh.csv`, `extrait_sondage.csv`)",
        type="csv",
        accept_multiple_files=True,
        on_change=clear_prediction_results # Clear results when new files are selected
    )

    raw_uploaded_data = None
    if len(uploaded_files) == 3:
        eval_file = None
        sirh_file = None
        sondage_file = None

        for f in uploaded_files:
            if 'eval' in f.name:
                eval_file = f
            elif 'sirh' in f.name:
                sirh_file = f
            elif 'sondage' in f.name:
                sondage_file = f

        if eval_file and sirh_file and sondage_file:
            eval_df = pd.read_csv(eval_file)
            sirh_df = pd.read_csv(sirh_file)
            sondage_df = pd.read_csv(sondage_file)
            raw_uploaded_data = load_and_merge_data(eval_df, sirh_df, sondage_df)
        else:
            st.warning("Please make sure to upload the three required files: `extrait_eval.csv`, `extrait_sirh.csv`, and `extrait_sondage.csv`.")
    elif uploaded_files:
        st.warning("Please upload all three required CSV files.")

    # --- Predict Button ---
    if raw_uploaded_data is not None:
        if st.button('Predict Attrition Risk'):
            # Get the list of columns the model expects
            expected_cols = get_expected_columns(model)
            
            # Create a new DataFrame with the expected columns
            data_for_prediction = pd.DataFrame(columns=expected_cols)
            
            # Copy data from the uploaded file to the new DataFrame
            for col in expected_cols:
                if col in raw_uploaded_data.columns:
                    data_for_prediction[col] = raw_uploaded_data[col]
            
            # Clean and engineer features
            processed_data = clean_and_engineer_features(data_for_prediction)
            
            # Make predictions
            prediction_proba = model.predict_proba(processed_data)[:, 1]
            predictions = (prediction_proba >= main_threshold).astype(int)

            # Prepare data for reports
            report_data = raw_uploaded_data.copy()
            report_data['Attrition_Risk_Percentage'] = prediction_proba
            report_data['Risk_Attrition'] = report_data['Attrition_Risk_Percentage'].apply(lambda x: get_risk_category(x, main_threshold))
            report_data['Prediction'] = ['Leave' if pred == 1 else 'Stay' for pred in predictions]

            # --- SHAP Explanations for Reports ---
            preprocessor = model.named_steps['preprocessor']
            logreg_model = model.named_steps['model']
            X_transformed_for_shap = preprocessor.transform(processed_data)
            
            try:
                ohe = preprocessor.named_transformers_['cat']
                cat_names = ohe.get_feature_names_out(processed_data.select_dtypes(include=['object', 'category', 'string', 'bool']).columns).tolist()
                num_cols = processed_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                all_features = num_cols + cat_names
            except Exception as e:
                st.warning(f"Could not get feature names for SHAP plots. Error: {e}")
                all_features = [f"Feature {i}" for i in range(X_transformed_for_shap.shape[1])]

            explainer = shap.LinearExplainer(logreg_model, X_transformed_for_shap)
            shap_values = explainer.shap_values(X_transformed_for_shap)

            # Add Top 10 Features and Coefficients to report_data
            top_features_list = []
            corresponding_coeffs_list = []
            features_and_coeffs_tab2 = [] # For Excel Tab 2

            for i in range(len(report_data)):
                employee_id = report_data.loc[i, 'id_employee']
                employee_shap_values = shap_values[i]
                shap_df_employee = pd.DataFrame({'feature': all_features, 'shap_value': employee_shap_values})
                shap_df_employee['abs_shap'] = shap_df_employee['shap_value'].abs()
                top_10 = shap_df_employee.sort_values('abs_shap', ascending=False).head(10)
                
                top_features_list.append("; ".join(top_10['feature'].tolist()))
                corresponding_coeffs_list.append("; ".join(top_10['shap_value'].round(4).astype(str).tolist()))

                # Prepare data for Excel Tab 2 (all features)
                prediction_type_for_employee = report_data.loc[i, 'Prediction']
                for _, row_shap in shap_df_employee.iterrows():
                    features_and_coeffs_tab2.append({
                        'Employee_ID': employee_id,
                        'Feature': row_shap['feature'],
                        'Coefficient': row_shap['shap_value'],
                        'Prediction': prediction_type_for_employee # Add prediction type here
                    })
            
            report_data['Top_10_Features'] = top_features_list
            report_data['Corresponding_Coefficients'] = corresponding_coeffs_list

            # Store results in session state
            st.session_state.prediction_triggered = True
            st.session_state.report_data = report_data
            st.session_state.processed_data_for_shap = X_transformed_for_shap
            st.session_state.explainer = explainer
            st.session_state.all_features = all_features
            st.session_state.excel_report_data = pd.DataFrame(features_and_coeffs_tab2)
        
    # --- Display Results (if triggered) ---
    if st.session_state.prediction_triggered:
        st.markdown("---")
        st.subheader("Prediction Results and Reports")

        report_data = st.session_state.report_data
        X_transformed_for_shap = st.session_state.processed_data_for_shap
        explainer = st.session_state.explainer
        all_features = st.session_state.all_features
        excel_tab2_data = st.session_state.excel_report_data

        # --- Generate Excel Report ---
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Tab 1: Summary (no employee name, no extra columns)
            tab1_df = report_data[['id_employee', 'Risk_Attrition', 'Attrition_Risk_Percentage']].copy()
            tab1_df.rename(columns={'id_employee':'Employee_ID'}, inplace=True)
            tab1_df.to_excel(writer, sheet_name='Summary', index=False)

            # Tab 2: Features (all features with coefficients; no employee name)
            tab2_df = excel_tab2_data.copy()
            # Ensure column names are exactly as required
            tab2_df.rename(columns={'Employee_ID': 'Employee_ID', 'Feature': 'Feature', 'Coefficient': 'Coefficient'}, inplace=True)
            tab2_df[['Employee_ID','Feature','Coefficient']].to_excel(writer, sheet_name='Features', index=False)

            # Tab 3: Metrics (optional)
            summary_metrics_df = pd.DataFrame({
                'Metric': ['Total Employees Processed', 'Predicted to Leave', 'Predicted to Stay'],
                'Value': [
                    len(report_data),
                    report_data['Prediction'].value_counts().get('Leave', 0),
                    report_data['Prediction'].value_counts().get('Stay', 0)
                ]
            })
            summary_metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        
        excel_buffer.seek(0)
        st.download_button(
            label="Download Excel Report",
            data=excel_buffer,
            file_name="employee_attrition_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # --- Generate HTML Visualization Report ---
        html_report_content = generate_shap_html_report(
            report_data, X_transformed_for_shap, explainer, all_features
        )

        st.subheader("Employee Attrition SHAP Report")
        # Render inline in the same browser/tab
        components.html(html_report_content, height=900, scrolling=True)

        # Optional: also provide a link to open in a new tab (no download)
        b64_html = base64.b64encode(html_report_content.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64_html}" target="_blank">Open SHAP Report in a new tab</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.success("Reports generated successfully!")

if __name__ == "__main__":
    main()