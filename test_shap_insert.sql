INSERT INTO shap_analysis (trace_id, shap_values, base_value, feature_names)
VALUES (22, '[1.0,2.0,3.0]'::json, -1.0379, '["f1","f2","f3"]'::json)
RETURNING shap_id, trace_id;
