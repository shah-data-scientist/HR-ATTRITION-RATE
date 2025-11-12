# COLOR & LABELS

*   **Confusion Matrix Colors:**
    *   Green: For True Negatives (Correctly Predicted Stay) and True Positives (Correctly Predicted Leave).
    *   Amber (or light orange/yellow): For False Positives (Incorrectly Predicted Leave).
    *   Red: For False Negatives (Incorrectly Predicted Stay).
    *   *Note: `seaborn.heatmap`'s `cmap` can be adjusted. For instance, a diverging colormap like `RdYlGn` or a custom colormap could be used to represent these states. For simplicity, `Greens` is used in the code, but a custom one would be better for amber/red.*
*   **Confusion Matrix Axis Labels:**
    *   X-axis: "Predicted Outcome"
    *   Y-axis: "Actual Outcome"
    *   Tick Labels: "Stay", "Leave" (for both predicted and actual)
*   **SHAP Waterfall Plot:**
    *   Colors: SHAP plots typically use red for positive impact (increasing prediction) and blue for negative impact (decreasing prediction). This is standard and generally understood.
    *   Axis Labels: Default SHAP labels are usually clear.
    *   Feature Names: Use the `all_features` list, which should ideally contain human-readable names. If not, a mapping to HR-friendly names should be applied before passing to SHAP.
*   **General Text/Microcopy:**
    *   Use bolding (`**text**`) for emphasis on key terms like "High Risk", "Overall Correct Predictions", "Correctly Identified 'Leave' Cases", "Estimated Review Workload", "Risk Score", "Prediction", "increasing", "decreasing".
    *   Keep sentences short and direct.
    *   Avoid technical terms like "false positive", "false negative", "precision", "F1-score".
