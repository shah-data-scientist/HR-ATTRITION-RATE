# -*- coding: utf-8 -*-

"""
This module defines the microcopy for the Streamlit application.
Using a centralized dictionary for UI texts (UI_TEXTS) makes it easy to
manage, update, and translate the application's content.
"""

UI_TEXTS = {
    # --- App General ---
    "app_title": "Employee Attrition Insights",

    # --- Header / Threshold Section ---
    "threshold_header": "Set Attrition Risk Cut-off",
    "threshold_slider_label": "Attrition Risk Cut-off",
    "threshold_slider_help": "Employees at or above this score are flagged as high risk.",
    "prediction_accuracy_overview_title": "Model Accuracy Overview",
    "overall_correct_predictions": "Overall Accuracy (% correct):",
    "correctly_identified_leave_cases": "Leavers Captured (% recall):",
    "estimated_review_workload": "Employees to Review: {}\n",

    # --- Introduction and Explanations ---
    "understanding_predictions_intro": "\n        Understanding how the model predicts attrition helps you interpret the scores.\n        The **Risk Score** shows how likely an employee is to leave. The **Attrition Risk Cut-off** decides when a score is high enough to flag an employee as *High Risk*.\n\n        - **Overall Accuracy:** % of all employees correctly predicted to stay or leave.\n        - **Leavers Captured (Recall):** % of employees who actually left and were flagged as *High Risk*.\n\n        Lowering the cut-off flags more employees (higher recall, more reviews).\n        Raising it flags fewer employees (lower recall, less workload).\n        Choose a balance that fits your HR capacity and retention priorities.\n    ",
    "threshold_examples_subheader": "Threshold Examples",
    "threshold_examples_list": [
        "**Threshold 0.30 – Proactive**\n> Flags many employees to ensure few potential leavers are missed.\n> HR reviews a larger group, catching almost everyone at risk.",
        "**Threshold 0.50 – Balanced**\n> Flags employees with ≥50% risk.\n> A balanced trade-off between recall and HR workload.",
        "**Threshold 0.70 – Conservative**\n> Flags only the most critical, high-risk employees.\n> Easier to manage but may miss moderate-risk leavers."
    ],

    # --- File Uploader ---
    "upload_subheader": "Upload Employee Data",
    "upload_file_uploader_label": "Upload 'extrait_eval.csv', 'extrait_sirh.csv', and 'extrait_sondage.csv'",
    "upload_info_no_files": "Please upload the 3 required CSV files.",
    "upload_warning_all_files": "Please upload all 3 required files: 'extrait_eval.csv', 'extrait_sirh.csv', and 'extrait_sondage.csv'.",
    "upload_warning_correct_names": "Please make sure the uploaded files are named correctly: 'extrait_eval.csv', 'extrait_sirh.csv', and 'extrait_sondage.csv'.",

    # --- API and Predictions ---
    "predict_button_label": "Predict Attrition",
    "prediction_success": "Predictions generated successfully!",
    "api_error": "API Error:",
    "unexpected_error": "An unexpected error occurred:",

    # --- Results and Reporting ---
    "overview_header": "High-Risk Employees",
    "overview_warning_no_data": "No prediction data to display. Please upload files and run prediction.",
    "reports_generated_success": "Reports generated successfully.",
    "employee_table_col_probability": "Attrition Risk (%)",
    "employee_table_col_prediction": "Model Decision",
    "download_excel_report_label": "Download Full Report (Excel)",
    "download_excel_report_help": "Download a detailed Excel report with employee risk scores and other details.",
    "excel_shap_message": "SHAP value explanations are available in the interactive dashboard.",

    # --- SHAP Explanations ---
    "employee_select_box_label": "Select an Employee for Details",
    "shap_plot_title": "Top Factors Driving Risk for Employee {Employee_ID}",
}
