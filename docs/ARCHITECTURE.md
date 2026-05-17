# Architecture

## System Overview

```
┌──────────────────────────────────────────────────────┐
│                   User Browser                        │
└────────────────────┬─────────────────────────────────┘
                     │ HTTPS
                     ▼
┌──────────────────────────────────────────────────────┐
│   Streamlit UI  (ui/app_authenticated.py)             │
│                                                       │
│   • Login page (bcrypt auth via API)                  │
│   • CSV upload (3 files)                              │
│   • Prediction results dashboard                      │
│   • SHAP waterfall / force charts                     │
│   • Excel report download                             │
│   • Async job polling                                 │
└────────────────────┬─────────────────────────────────┘
                     │ HTTP  X-API-Key header
                     ▼
┌──────────────────────────────────────────────────────┐
│   FastAPI  (api/app/main.py)                          │
│                                                       │
│   Endpoints                                           │
│   ├── POST /predict            → JSON predictions     │
│   ├── POST /predict_report     → JSON + Excel         │
│   ├── POST /predict_excel      → Excel upload         │
│   ├── POST /predict_shap_images → ZIP of PNGs        │
│   ├── POST /predict_shap_html  → HTML force plot      │
│   ├── POST /jobs/report        → async job            │
│   ├── GET  /jobs/{id}          → job status           │
│   ├── GET  /jobs/{id}/result   → job output           │
│   ├── POST /auth/login         → session token        │
│   └── GET  /auth/user/{name}   → user info            │
│                                                       │
│   Security                                            │
│   ├── API key authentication (api/auth.py)            │
│   ├── Security headers: XSS, CSRF, CSP, HSTS         │
│   ├── CORS configuration                              │
│   └── GZip compression                               │
└────────┬──────────────────────┬──────────────────────┘
         │                      │
         ▼                      ▼
┌────────────────┐    ┌──────────────────────────────┐
│  PostgreSQL    │    │   ML Pipeline                 │
│  (6 tables)    │    │   models/                     │
│                │    │                               │
│  employees     │    │  employee_attrition_pipeline  │
│  model_inputs  │    │  .pkl (sklearn Pipeline)      │
│  model_outputs │    │                               │
│  predictions_  │    │  Steps:                       │
│  traceability  │    │  1. preprocessor              │
│  shap_analysis │    │     (ColumnTransformer)       │
│  jobs          │    │  2. model (Linear classifier) │
│  users         │    │                               │
└────────────────┘    │  SHAP: LinearExplainer        │
                      │  trained on X_train.parquet   │
                      └──────────────────────────────┘
         ▲
         │ polls jobs table
┌────────────────┐
│ Background     │
│ Worker         │
│ scripts/       │
│ worker.py      │
└────────────────┘
```

---

## Data Flow — Prediction Request

```
1. User uploads 3 CSVs in the UI
   extrait_eval.csv   → evaluation scores, overtime, salary data
   extrait_sirh.csv   → demographics, tenure, salary
   extrait_sondage.csv→ survey: training, travel, education

2. UI sends RawBatchPredictionInput JSON to POST /predict

3. API merges the 3 sources on id_employee / eval_number / code_sondage

4. core/data_processing.py — clean_and_engineer_features()
   - Normalise genre codes (M/H/Homme → 0, F/Femme → 1)
   - Parse overtime string ("Oui"/"Non" → 1/0)
   - Extract salary increase percentage from string ("11 %" → 11.0)
   - Engineer 3 composite features:
       improvement_evaluation = note_actuelle - note_precedente
       total_satisfaction     = mean(4 satisfaction scores)
       work_mobility          = annees_dans_poste / annees_dans_entreprise

5. core/preprocess.py — enforce_schema()
   - Reorder columns to match training feature order
   - Fill missing columns with 0
   - Coerce dtypes to match model expectations
   - Validate numeric ranges

6. ML Pipeline inference
   - preprocessor.transform(X) → scaled/encoded features
   - model.predict_proba(X) → attrition probability [0, 1]
   - Risk category: Low (<0.3), Medium (0.3–0.7), High (≥0.7)
   - Optimal decision threshold: 0.2876

7. SHAP explanation (per employee)
   - LinearExplainer(model, X_train) → shap_values[]
   - base_value (expected prediction)
   - feature_names[]

8. PostgreSQL storage (when DISABLE_DB=false)
   - employees table ← cleaned input features
   - model_inputs  ← features JSON
   - model_outputs ← probability, risk_category, label, log_odds
   - predictions_traceability ← links input + output + metadata
   - shap_analysis ← shap_values JSON, base_value, feature_names

9. Response: list[PredictionOutput]
   - id_employee, prediction (Stay/Leave), probability, risk_category
   - shap_values, base_value, feature_names (optional)
   - trace_id (for audit)
```

---

## Database Schema

### `employees`
Stores cleaned, engineered employee features from each prediction request.

| Column | Type | Notes |
|--------|------|-------|
| id_employee | Integer PK | Unique employee identifier |
| age, genre, revenu_mensuel | Mixed | Demographics |
| departement, poste, statut_marital | String | Organisation |
| annees_dans_l_entreprise, annee_experience_totale | Float | Tenure |
| satisfaction_employee_* (4 cols) | Float | Survey scores 1–4 |
| note_evaluation_precedente, note_evaluation_actuelle | Float | Annual reviews |
| improvement_evaluation | Float | Engineered: current – previous |
| total_satisfaction | Float | Engineered: mean of 4 scores |
| work_mobility | Float | Engineered: role tenure / company tenure |
| user_id | String | Who submitted the prediction |
| date_ingestion | DateTime | Ingestion timestamp |

### `model_inputs`
One row per prediction per employee. Stores raw feature JSON sent to the model.

| Column | Type | Notes |
|--------|------|-------|
| input_id | Integer PK | |
| id_employee | FK → employees | |
| features | JSON | Full feature vector |
| prediction_timestamp | DateTime | |

### `model_outputs`
One row per prediction. Stores model output.

| Column | Type | Notes |
|--------|------|-------|
| output_id | Integer PK | |
| prediction_proba | Float | Attrition probability 0–1 |
| risk_category | String | Low / Medium / High |
| prediction_label | String | Stay / Leave |
| log_odds | Float | Raw model output |
| threshold | Float | Decision threshold used |

### `predictions_traceability`
Links each input to its output, with request metadata.

| Column | Type | Notes |
|--------|------|-------|
| trace_id | Integer PK | |
| input_id | FK → model_inputs | |
| output_id | FK → model_outputs | |
| model_version | String | |
| prediction_source | String | api / ui / batch |
| request_metadata | JSON | Headers, client info |

### `shap_analysis`
One row per trace (one per prediction). Stores full SHAP output.

| Column | Type | Notes |
|--------|------|-------|
| shap_id | Integer PK | |
| trace_id | FK → predictions_traceability (unique) | |
| shap_values | JSON | Array of float per feature |
| base_value | Float | SHAP expected value |
| feature_names | JSON | Array of feature name strings |

### `jobs`
Async job queue for background report generation.

| Column | Type | Notes |
|--------|------|-------|
| job_id | String PK (UUID) | |
| job_type | String | e.g. "report" |
| status | String | queued / processing / completed / failed |
| payload_json | JSON | Input data |
| result_json | JSON | Output when completed |
| error | String | Error message if failed |
| user_id | String | Who created the job |

### `users`
Application users for Streamlit login.

| Column | Type | Notes |
|--------|------|-------|
| user_id | Integer PK | |
| username | String (unique) | Login name |
| password_hash | String | bcrypt hash |
| role | String | admin / user |
| is_active | Integer | 1=active, 0=inactive |
| last_login | DateTime | |

---

## ML Model

- **Type:** scikit-learn `Pipeline` with two named steps:
  - `preprocessor`: `ColumnTransformer` (scaling + encoding)
  - `model`: Linear classifier (logistic regression family)
- **Explainability:** `shap.LinearExplainer` initialised from `X_train.parquet`
- **Decision threshold:** 0.2876 (optimised during training, lower = more sensitive)
- **Risk thresholds:** Low < 0.3, Medium 0.3–0.7, High ≥ 0.7

---

## Authentication

### API (service-to-service)
- `X-API-Key` header required on all prediction and job endpoints
- Key verified against `API_KEY` environment variable using constant-time comparison
- No expiry — rotate by changing `API_KEY` in `.env`

### UI (user-facing)
- Username + password login form in Streamlit
- Password hashed with bcrypt (stored in `users` table)
- Session stored in `st.session_state` (browser tab scope)
- Roles: `admin` and `user` (role checked on protected actions)

---

## Background Worker

`scripts/worker.py` polls the `jobs` table every `WORKER_POLL_SEC` seconds. When it finds a job with status `queued`, it:

1. Sets status → `processing`
2. Loads the payload, runs `generate_predictions()` + Excel generation
3. Sets status → `completed`, stores result in `result_json`
4. On error: sets status → `failed`, stores error message

The worker runs in the production Docker profile as a separate container sharing the API image.
