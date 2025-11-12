import streamlit as st
import pandas as pd
import numpy as np
import shap # Assuming shap is installed and available
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper function for Confusion Matrix and Metrics ---
def display_confusion_matrix_and_metrics(y_true, y_proba, threshold, title):
    """
    Displays a confusion matrix and calculates Accuracy and Recall.
    Returns accuracy, recall, and the count of flagged employees.
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate confusion matrix components
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Using labels=[0, 1] to ensure consistent order for binary classification
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Calculate Accuracy and Recall
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Display Confusion Matrix
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Predicted Stay', 'Predicted Leave'],
                yticklabels=['Actual Stay', 'Actual Leave'], ax=ax)
    ax.set_ylabel('Actual Outcome')
    ax.set_xlabel('Predicted Outcome')
    st.pyplot(fig)
    plt.close(fig) # Close the plot to prevent it from displaying twice

    # Count of employees flagged as 'Leave' (True Positives + False Positives)
    flagged_count = tp + fp

    return accuracy, recall, flagged_count


# --- STREAMLIT CODE ---

def render_threshold_block():
    """
    Renders the confusion matrix with a threshold slider, Accuracy and Recall metrics,
    workload estimation, and an HR-friendly explanation.
    """
    st.header("Adjust Risk Threshold & Review Impact")

    col1, col2 = st.columns([0.7, 0.3]) # Confusion matrix LEFT, slider RIGHT

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
        "Prediction": "Prediction"
    })
    
    # Format Risk Score as percentage
    df_display["Risk Score"] = df_display["Risk Score"].apply(lambda x: f"{x:.1%}")

    # Display sortable table
    st.dataframe(df_display[["Employee ID", "Risk Score", "Prediction"]].sort_values(by="Risk Score", ascending=False), use_container_width=True)

    st.subheader("Deep Dive: Individual Employee Analysis")
    
    employee_ids = st.session_state.report_data["Employee ID"].unique() # Use renamed column
    selected_employee_id = st.selectbox(
        "Select Employee for Detail",
        options=employee_ids,
        help="Choose an employee ID to see a detailed breakdown of their attrition risk factors."
    )

    if selected_employee_id:
        st.subheader(f"Key Drivers for Employee {selected_employee_id}")

        # Get data for the selected employee
        employee_row_index = st.session_state.report_data[st.session_state.report_data["id_employee"] == selected_employee_id].index[0]
        
        # Ensure explainer and processed_data_for_shap are available
        if 'explainer' not in st.session_state or 'processed_data_for_shap' not in st.session_state:
            st.error("SHAP explainer or data not found in session state. Cannot display SHAP plot.")
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
            ax=ax
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

# --- End of STREAMLIT CODE ---
