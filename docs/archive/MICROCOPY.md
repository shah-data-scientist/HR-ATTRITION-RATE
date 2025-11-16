
# MICROCOPY

**Main App Title:** Employee Attrition Insights
**Threshold Section Title:** Set Attrition Risk Cut-off
**Threshold Slider Label:** Attrition Risk Cut-off
**Tooltip:** “Employees at or above this score are flagged as high risk.”
**Confusion Matrix Title:** Model Accuracy Overview
**Accuracy Metric Label:** Overall Accuracy (% correct)
**Recall Metric Label:** Leavers Captured (% recall)
**Workload Estimate Label:** Employees to Review
**Employee Table Title:** High-Risk Employees
**Employee Table Column: Employee ID:** "Employee ID"
**Employee Table Column: Probability:** | Risk Score  | Attrition Risk (%) |
**Employee Table Column: Prediction:** | Prediction  | Model Decision     |
**Employee Select Box Label:** Select an Employee for Details
**SHAP Plot Title:** Top Factors Driving Risk for Employee {Employee_ID}
**Top Drivers Summary Title:** Main Reasons for Attrition Risk

**Explanation of Accuracy vs. Recall and Threshold Trade-off:**

> Understanding how the model predicts attrition helps you interpret the scores.
> The **Risk Score** shows how likely an employee is to leave. The **Attrition Risk Cut-off** decides when a score is high enough to flag an employee as *High Risk*.
>
> * **Overall Accuracy:** % of all employees correctly predicted to stay or leave.
> * **Leavers Captured (Recall):** % of employees who actually left and were flagged as *High Risk*.
>
> Lowering the cut-off flags more employees (higher recall, more reviews).
> Raising it flags fewer employees (lower recall, less workload).
> Choose a balance that fits your HR capacity and retention priorities.

**Three Threshold Examples:**

**Threshold 0.30 – Proactive**

> Flags many employees to ensure few potential leavers are missed.
> HR reviews a larger group, catching almost everyone at risk.

**Threshold 0.50 – Balanced**

> Flags employees with ≥50% risk.
> A balanced trade-off between recall and HR workload.

**Threshold 0.70 – Conservative**

> Flags only the most critical, high-risk employees.
> Easier to manage but may miss moderate-risk leavers.
