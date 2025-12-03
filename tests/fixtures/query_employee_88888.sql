-- Query all tables for Employee 88888 (trace_id 18)

-- 1. EMPLOYEES TABLE
SELECT 'TABLE 1: employees' as table_info;
SELECT 
    id_employee, 
    user_id, 
    age, 
    genre, 
    departement, 
    poste, 
    annees_dans_l_entreprise,
    total_satisfaction,
    improvement_evaluation,
    work_mobility,
    date_ingestion
FROM employees 
WHERE id_employee = 88888;

-- 2. MODEL_INPUTS TABLE  
SELECT 'TABLE 2: model_inputs' as table_info;
SELECT 
    input_id,
    id_employee,
    prediction_timestamp,
    substring(features::text, 1, 200) as features_sample
FROM model_inputs 
WHERE id_employee = 88888;

-- 3. MODEL_OUTPUTS TABLE
SELECT 'TABLE 3: model_outputs' as table_info;
SELECT 
    output_id,
    prediction_proba,
    risk_category,
    prediction_label,
    log_odds,
    threshold,
    prediction_timestamp
FROM model_outputs 
WHERE output_id = 18;

-- 4. PREDICTIONS_TRACEABILITY TABLE
SELECT 'TABLE 4: predictions_traceability' as table_info;
SELECT 
    trace_id,
    input_id,
    output_id,
    model_version,
    prediction_source,
    request_metadata,
    created_at
FROM predictions_traceability 
WHERE trace_id = 18;

-- 5. SHAP_ANALYSIS TABLE (NEW!)
SELECT 'TABLE 5: shap_analysis' as table_info;
SELECT 
    shap_id,
    trace_id,
    base_value,
    json_array_length(shap_values::json) as num_shap_values,
    json_array_length(feature_names::json) as num_features,
    substring(shap_values::text, 1, 150) as shap_values_sample,
    substring(feature_names::text, 1, 150) as feature_names_sample,
    created_at
FROM shap_analysis 
WHERE trace_id = 18;
