import pandas as pd
import io
import base64
import numpy as np
import shap  # Required for shap.Explanation
from pydantic import BaseModel
from typing import Optional, List


# Replicate PredictionOutput schema for local testing
class PredictionOutput(BaseModel):
    id_employee: int
    prediction: str
    probability: float
    risk_category: str
    message: str
    trace_id: Optional[str] = None
    shap_values: Optional[List[float]] = None
    base_value: Optional[float] = None
    feature_names: Optional[List[str]] = None


def generate_excel_report_debug(predictions_output: List[PredictionOutput]):
    """
    Debug function to isolate Excel generation logic.
    Mimics the logic from predict_excel and predict_attrition_report.
    """
    try:
        # Build Summary sheet
        summary_rows = []
        for p in predictions_output:
            summary_rows.append(
                {
                    "Employee_ID": p.id_employee,
                    "Risk_Attrition": p.risk_category,
                    "Attrition_Risk_Percentage": round(float(p.probability) * 100, 1),
                    "Prediction": p.prediction,
                }
            )
        summary_df = pd.DataFrame(summary_rows)

        # Build Features sheet (stacked SHAP values)
        features_frames = []
        for p in predictions_output:
            if p.shap_values is not None and p.feature_names is not None:
                feature_names = (
                    p.feature_names
                    if len(p.feature_names) == len(p.shap_values)
                    else [f"Feature {i}" for i in range(len(p.shap_values))]
                )

                # Explicitly cast to ensure no unexpected types
                df = pd.DataFrame(
                    {
                        "Feature": pd.Series(feature_names).astype(str).tolist(),
                        "Coefficient": pd.Series(p.shap_values).astype(float).tolist(),
                    }
                )
                df["Employee_ID"] = p.id_employee
                df["Prediction"] = p.prediction
                features_frames.append(df)
        features_df = (
            pd.concat(features_frames, ignore_index=True)
            if features_frames
            else pd.DataFrame()
        )

        # Build Metrics sheet
        total = len(predictions_output)
        predicted_leave = sum(1 for p in predictions_output if p.prediction == "Leave")
        predicted_stay = sum(1 for p in predictions_output if p.prediction == "Stay")
        metrics_df = pd.DataFrame(
            {
                "Metric": [
                    "Total Employees Processed",
                    "Predicted to Leave",
                    "Predicted to Stay",
                ],
                "Value": [total, predicted_leave, predicted_stay],
            }
        )

        print(f"\n--- Debugging DataFrames before Excel Writing ---")
        print(f"Summary DataFrame dtypes: \n{summary_df.dtypes}")
        if not features_df.empty:
            print(f"Features DataFrame dtypes: \n{features_df.dtypes}")
        print(f"Metrics DataFrame dtypes: \n{metrics_df.dtypes}")
        print(f"--- End Debugging DataFrames ---")

        # Write Excel to bytes
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            summary_df.astype(str).to_excel(writer, sheet_name="Summary", index=False)
            if not features_df.empty:
                features_df.astype(str)[
                    ["Employee_ID", "Feature", "Coefficient", "Prediction"]
                ].to_excel(writer, sheet_name="Features", index=False)
            metrics_df.astype(str).to_excel(writer, sheet_name="Metrics", index=False)
        excel_buffer.seek(0)
        excel_b64 = base64.b64encode(excel_buffer.read()).decode("utf-8")
        print(f"Excel report generated successfully (base64 length: {len(excel_b64)})")
        return excel_b64

    except Exception as e:
        print(f"Error during Excel report generation: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


# --- Sample Data ---
# This data structure needs to be fully representative of what generate_predictions returns
# Taken from tests/test_deep_coverage.py sample payload
sample_predictions_output = [
    PredictionOutput(
        id_employee=90001,
        prediction="Leave",
        probability=0.6116593720765086,
        risk_category="High",
        message="Employee 90001 is predicted to Leave with 61.17% attrition risk (Risk: High).",
        trace_id="test_trace_id_123",
        shap_values=[0.1, 0.05, -0.03, 0.02, -0.01],  # Sample SHAP values
        base_value=-0.5,  # Sample base value
        feature_names=[
            "Feature_A",
            "Feature_B",
            "Feature_C",
            "Feature_D",
            "Feature_E",
        ],  # Sample feature names
    )
]

if __name__ == "__main__":
    print("Running debug script for Excel generation.")
    result = generate_excel_report_debug(sample_predictions_output)
    if result:
        print("Debug script finished successfully.")
    else:
        print("Debug script failed.")
