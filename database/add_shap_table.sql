-- Add SHAP Analysis table to store SHAP values for predictions
CREATE TABLE IF NOT EXISTS shap_analysis (
    shap_id SERIAL PRIMARY KEY,
    trace_id INTEGER NOT NULL UNIQUE REFERENCES predictions_traceability(trace_id) ON DELETE CASCADE,
    shap_values JSON NOT NULL,
    base_value DOUBLE PRECISION NOT NULL,
    feature_names JSON NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on trace_id for faster lookups
CREATE INDEX IF NOT EXISTS ix_shap_analysis_trace_id ON shap_analysis(trace_id);
CREATE INDEX IF NOT EXISTS ix_shap_analysis_shap_id ON shap_analysis(shap_id);

-- Add comment to table
COMMENT ON TABLE shap_analysis IS 'Stores SHAP (SHapley Additive exPlanations) values for model predictions to enable explainability and interpretability';
COMMENT ON COLUMN shap_analysis.shap_values IS 'Array of SHAP values for each feature';
COMMENT ON COLUMN shap_analysis.base_value IS 'Base value (expected value) from SHAP explainer';
COMMENT ON COLUMN shap_analysis.feature_names IS 'Array of feature names corresponding to SHAP values';
