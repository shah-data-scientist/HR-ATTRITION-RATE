
# MICROCOPY

**Main App Title:** "Employee Attrition Risk Dashboard"
**Threshold Section Title:** "Adjust Risk Threshold & Review Impact"
**Threshold Slider Label:** "Set 'High Risk' Threshold"
**Confusion Matrix Title:** "Prediction Accuracy Overview"
**Accuracy Metric Label:** "Overall Correct Predictions"
**Recall Metric Label:** "Correctly Identified 'Leave' Cases"
**Workload Estimate Label:** "Estimated Review Workload"
**Employee Table Title:** "Employee Risk Overview"
**Employee Table Column: Employee ID:** "Employee ID"
**Employee Table Column: Probability:** "Risk Score"
**Employee Table Column: Prediction:** "Prediction"
**Employee Select Box Label:** "Select Employee for Detail"
**SHAP Plot Title:** "Key Drivers for Employee {Employee ID}"
**Top Drivers Summary Title:** "Top Reasons for Risk"

**Explanation of Accuracy vs. Recall and Threshold Trade-off:**

"Understanding how our model predicts employee attrition is key. The 'Risk Score' (probability) tells us how likely an employee is to leave. We use a 'threshold' to decide when a score is high enough to flag an employee as 'High Risk' (predicted to leave).

*   **Overall Correct Predictions (Accuracy):** This shows the percentage of all employees (both those who stay and those who leave) that our model predicted correctly. A high number here means the model is generally good at its job.
*   **Correctly Identified 'Leave' Cases (Recall):** This is crucial for proactive HR. It tells us, out of all the employees who *actually* left, what percentage our model successfully flagged as 'High Risk'. A high Recall means we're good at catching potential leavers.

Adjusting the 'High Risk' Threshold changes how many employees are flagged. A lower threshold means we flag more employees, increasing our 'Recall' (catching more potential leavers) but potentially also flagging more employees who would have stayed. A higher threshold flags fewer employees, reducing the 'workload' but risking missing some who might leave. It's a balance between catching all potential leavers and managing the number of employees HR needs to review."

**Three Threshold Examples:**

*   **Threshold 0.30 (More Proactive):** "At a threshold of 0.30, the model is very sensitive. It flags more employees as 'High Risk', aiming to catch almost everyone who might leave. This means HR will review a larger group, ensuring fewer potential leavers are missed, but some flagged employees might have actually stayed."
*   **Threshold 0.50 (Balanced Approach):** "With a threshold of 0.50, the model takes a balanced approach. It flags employees with a 50% or higher risk score. This provides a good balance between identifying potential leavers and keeping the review workload manageable for HR."
*   **Threshold 0.70 (More Conservative):** "Using a threshold of 0.70, the model is more conservative. It only flags employees with a very high risk score. This significantly reduces the number of employees HR needs to review, focusing only on the most critical cases, but it might miss some employees who eventually leave."
