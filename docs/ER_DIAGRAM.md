# Entity-Relationship Diagram

## Database Schema Overview

This document describes the relational database schema for the HR Attrition Rate prediction system. The database consists of 6 main entities that track employees, predictions, SHAP explanations, and background jobs.

## ER Diagram (Mermaid)

```mermaid
erDiagram
    EMPLOYEES ||--o{ MODEL_INPUTS : "generates"
    MODEL_INPUTS ||--|| PREDICTIONS_TRACEABILITY : "tracked_by"
    MODEL_OUTPUTS ||--|| PREDICTIONS_TRACEABILITY : "tracked_by"
    PREDICTIONS_TRACEABILITY ||--o| SHAP_ANALYSIS : "explains"
    JOBS }|--|| USER : "created_by"

    EMPLOYEES {
        SERIAL id_employee PK "Employee unique identifier"
        INTEGER age "Employee age"
        INTEGER genre "Gender (encoded)"
        FLOAT revenu_mensuel "Monthly income"
        VARCHAR statut_marital "Marital status"
        VARCHAR departement "Department"
        VARCHAR poste "Job title"
        INTEGER nombre_experiences_precedentes "Number of previous jobs"
        INTEGER annee_experience_totale "Total years of experience"
        FLOAT annees_dans_l_entreprise "Years at company"
        FLOAT annees_dans_le_poste_actuel "Years in current role"
        INTEGER nombre_participation_pee "PEE participation count"
        INTEGER nb_formations_suivies "Number of trainings completed"
        INTEGER nombre_employee_sous_responsabilite "Number of direct reports"
        INTEGER distance_domicile_travail "Commute distance (km)"
        INTEGER niveau_education "Education level"
        VARCHAR domaine_etude "Field of study"
        INTEGER ayant_enfants "Has children flag"
        VARCHAR frequence_deplacement "Travel frequency"
        INTEGER annees_depuis_la_derniere_promotion "Years since last promotion"
        INTEGER annes_sous_responsable_actuel "Years under current manager"
        FLOAT satisfaction_employee_environnement "Environment satisfaction"
        FLOAT note_evaluation_precedente "Previous evaluation score"
        FLOAT niveau_hierarchique_poste "Job hierarchical level"
        FLOAT satisfaction_employee_nature_travail "Job satisfaction"
        FLOAT satisfaction_employee_equipe "Team satisfaction"
        FLOAT satisfaction_employee_equilibre_pro_perso "Work-life balance satisfaction"
        FLOAT note_evaluation_actuelle "Current evaluation score"
        INTEGER heures_supplementaires "Overtime hours"
        FLOAT augmentation_salaire_precedente "Previous salary increase %"
        FLOAT augementation_salaire_precedente "Previous salary increase % (alt)"
        FLOAT nombre_heures_travailless "Weekly work hours"
        FLOAT improvement_evaluation "Evaluation improvement"
        FLOAT total_satisfaction "Total satisfaction score"
        FLOAT work_mobility "Work mobility score"
        VARCHAR user_id FK "User who created record"
        TIMESTAMP date_ingestion "Record creation timestamp"
    }

    MODEL_INPUTS {
        SERIAL input_id PK "Input record unique identifier"
        INTEGER id_employee FK "Employee reference"
        TIMESTAMP prediction_timestamp "When prediction was requested"
        JSON features "Preprocessed feature values"
    }

    MODEL_OUTPUTS {
        SERIAL output_id PK "Output record unique identifier"
        FLOAT prediction_proba "Attrition probability (0-1)"
        VARCHAR risk_category "Risk level: Low/Medium/High"
        VARCHAR prediction_label "Attrition/No Attrition"
        FLOAT log_odds "Log-odds of attrition"
        FLOAT threshold "Classification threshold used"
        TIMESTAMP prediction_timestamp "When prediction was made"
    }

    PREDICTIONS_TRACEABILITY {
        SERIAL trace_id PK "Trace unique identifier"
        INTEGER input_id FK "Reference to model input"
        INTEGER output_id FK "Reference to model output"
        VARCHAR model_version "Model version identifier"
        VARCHAR prediction_source "Source: API/UI/Batch"
        JSON request_metadata "Additional request context"
        TIMESTAMP created_at "Trace creation timestamp"
    }

    SHAP_ANALYSIS {
        SERIAL shap_id PK "SHAP record unique identifier"
        INTEGER trace_id FK "Reference to prediction trace"
        JSON shap_values "SHAP values for each feature"
        FLOAT base_value "Expected value (baseline)"
        JSON feature_names "Feature names in order"
        TIMESTAMP created_at "Analysis creation timestamp"
    }

    JOBS {
        VARCHAR job_id PK "UUID for background job"
        VARCHAR job_type "Job type: report/batch/etc"
        VARCHAR status "Status: queued/processing/completed/failed"
        JSON payload_json "Job input parameters"
        JSON result_json "Job output results"
        VARCHAR error "Error message if failed"
        VARCHAR user_id FK "User who created job"
        TIMESTAMP created_at "Job creation timestamp"
        TIMESTAMP updated_at "Job last update timestamp"
    }
```

## Entity Descriptions

### 1. EMPLOYEES
**Purpose**: Stores raw employee data from HR systems (SIRH, evaluations, surveys).

**Key Characteristics**:
- Primary key: `id_employee` (auto-incremented)
- Contains 35+ employee attributes covering demographics, job details, satisfaction scores
- Supports multi-tenancy via `user_id` field
- Tracks data ingestion timestamp for audit purposes

**Relationships**:
- One employee → Many model inputs (1:N)

### 2. MODEL_INPUTS
**Purpose**: Captures preprocessed feature vectors sent to the ML model for prediction.

**Key Characteristics**:
- Primary key: `input_id` (auto-incremented)
- Foreign key: `id_employee` references EMPLOYEES
- Stores features as JSON for flexibility (supports schema evolution)
- Timestamps when prediction was requested

**Relationships**:
- Many model inputs → One employee (N:1)
- One model input → One prediction trace (1:1)

### 3. MODEL_OUTPUTS
**Purpose**: Stores ML model predictions and risk assessments.

**Key Characteristics**:
- Primary key: `output_id` (auto-incremented)
- Stores prediction probability (0-1 scale)
- Categorizes risk as Low/Medium/High
- Includes log-odds for advanced analysis
- Tracks classification threshold used

**Relationships**:
- One model output → One prediction trace (1:1)

### 4. PREDICTIONS_TRACEABILITY
**Purpose**: Links model inputs and outputs for full audit trail and explainability.

**Key Characteristics**:
- Primary key: `trace_id` (auto-incremented)
- Foreign keys: `input_id`, `output_id`
- Tracks model version for reproducibility
- Records prediction source (API/UI/Batch)
- Stores request metadata (user info, session data)

**Relationships**:
- One trace → One model input (1:1)
- One trace → One model output (1:1)
- One trace → One SHAP analysis (1:1)

### 5. SHAP_ANALYSIS
**Purpose**: Stores SHAP (SHapley Additive exPlanations) values for model interpretability.

**Key Characteristics**:
- Primary key: `shap_id` (auto-incremented)
- Foreign key: `trace_id` (unique constraint ensures 1:1 with traces)
- Stores SHAP values as JSON array (one value per feature)
- Includes base value (expected model output)
- Feature names array for mapping SHAP values to features

**Relationships**:
- One SHAP analysis → One prediction trace (1:1)

### 6. JOBS
**Purpose**: Manages asynchronous background tasks (report generation, batch predictions).

**Key Characteristics**:
- Primary key: `job_id` (UUID)
- Tracks job lifecycle: queued → processing → completed/failed
- Stores job parameters as JSON (payload_json)
- Stores job results as JSON (result_json)
- Supports multi-tenancy via `user_id`
- Tracks creation and update timestamps

**Relationships**:
- Many jobs → One user (N:1, implicit)

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant DB
    participant MLModel
    participant SHAP

    User->>API: Upload employee data
    API->>DB: Insert into EMPLOYEES
    DB-->>API: id_employee
    
    User->>API: Request prediction
    API->>DB: Fetch employee data
    API->>API: Preprocess features
    API->>DB: Insert MODEL_INPUTS
    DB-->>API: input_id
    
    API->>MLModel: Predict attrition
    MLModel-->>API: prediction_proba, log_odds
    API->>DB: Insert MODEL_OUTPUTS
    DB-->>API: output_id
    
    API->>DB: Insert PREDICTIONS_TRACEABILITY
    DB-->>API: trace_id
    
    API->>SHAP: Calculate SHAP values
    SHAP-->>API: shap_values, base_value
    API->>DB: Insert SHAP_ANALYSIS
    
    API-->>User: Return prediction + SHAP
```

## Indexes and Performance

**Recommended Indexes** (for production):

```sql
-- Primary keys (auto-indexed)
CREATE INDEX idx_employees_pk ON employees(id_employee);
CREATE INDEX idx_model_inputs_pk ON model_inputs(input_id);
CREATE INDEX idx_model_outputs_pk ON model_outputs(output_id);
CREATE INDEX idx_predictions_traceability_pk ON predictions_traceability(trace_id);
CREATE INDEX idx_shap_analysis_pk ON shap_analysis(shap_id);
CREATE INDEX idx_jobs_pk ON jobs(job_id);

-- Foreign key indexes
CREATE INDEX idx_model_inputs_employee ON model_inputs(id_employee);
CREATE INDEX idx_predictions_trace_input ON predictions_traceability(input_id);
CREATE INDEX idx_predictions_trace_output ON predictions_traceability(output_id);
CREATE INDEX idx_shap_analysis_trace ON shap_analysis(trace_id);

-- Query optimization indexes
CREATE INDEX idx_employees_user_id ON employees(user_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_user_id ON jobs(user_id);
CREATE INDEX idx_predictions_created_at ON predictions_traceability(created_at);
```

## Design Principles

### 1. **Separation of Concerns**
- **Input/Output Separation**: Separate tables for model inputs and outputs allow independent schema evolution
- **Traceability Layer**: Dedicated table links inputs/outputs without tight coupling

### 2. **Auditability**
- All tables have timestamps (`created_at`, `updated_at`, `date_ingestion`)
- `PREDICTIONS_TRACEABILITY` tracks model version and source
- `request_metadata` JSON field captures additional context

### 3. **Flexibility**
- JSON columns (`features`, `shap_values`, `payload_json`) support schema changes without migrations
- Generic `JOBS` table handles multiple job types

### 4. **Multi-Tenancy**
- `user_id` field in `EMPLOYEES` and `JOBS` enables data isolation
- Foreign key relationships maintain data integrity per user

### 5. **Explainability**
- Dedicated `SHAP_ANALYSIS` table ensures interpretability
- 1:1 relationship with traces guarantees every prediction can be explained

## Migration History

1. **Initial Schema** - Core tables (employees, model_inputs, model_outputs, predictions_traceability)
2. **SHAP Integration** - Added `shap_analysis` table for explainability
3. **Background Jobs** - Added `jobs` table for async processing
4. **Multi-Tenancy** - Added `user_id` to employees and jobs
5. **Threshold Tracking** - Added `threshold` column to model_outputs

## Notes

- **Data Types**: PostgreSQL-specific types used (SERIAL, TIMESTAMP WITH TIME ZONE, JSON)
- **Normalization**: Schema is normalized to 3NF (Third Normal Form)
- **JSON Usage**: JSON fields used strategically for flexibility vs. queryability trade-off
- **Future Extensions**: Schema supports adding new feature columns, prediction types, or job types without major refactoring
