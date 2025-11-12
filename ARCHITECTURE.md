# Architecture and Data Flow

This document provides an overview of the project's architecture, database schema, and data flow.

## System Architecture

The system is composed of three main components:

1.  **Streamlit UI (`app.py`):** A web-based user interface for interactive analysis. It allows users to upload employee data, adjust prediction thresholds, and visualize model predictions and feature importance (SHAP).
2.  **FastAPI Backend (`api/app/main.py`):** A RESTful API that serves the machine learning model. It receives employee data, runs predictions, and is designed to log these interactions in the database.
3.  **PostgreSQL Database:** A relational database that stores employee data, model predictions, and traceability information.

## Database Schema

The database schema is defined using SQLAlchemy ORM in `database/models.py`. It consists of four main tables designed to store employee information and track model predictions.

### Schema Diagram (Text-based)

```
+----------------+      +------------------+      +--------------------------+
|   employees    |      |   model_inputs   |      | predictions_traceability |
|----------------|      |------------------|      |--------------------------|
| id_employee (PK)|--+--<| id_employee (FK) |      | trace_id (PK)            |
| age            |      | input_id (PK)    |----+-<| input_id (FK)            |
| genre          |      | features (JSON)  |      | output_id (FK)           |
| revenu_mensuel |      | ...              |      | model_version            |
| ...            |      +------------------+      | ...                      |
+----------------+                                +--------------------------+
                                                          |
                                                          |      +----------------+
                                                          +----->|  model_outputs |
                                                                 |----------------|
                                                                 | output_id (PK) |
                                                                 | prediction_proba|
                                                                 | risk_category  |
                                                                 | ...            |
                                                                 +----------------+
```

### Table Descriptions

*   **`employees`**
    *   **Purpose:** Stores the raw and engineered features for each employee. This table serves as the main source of truth for employee data.
    *   **Primary Key:** `id_employee`

*   **`model_inputs`**
    *   **Purpose:** Records the exact set of features sent to the model for a prediction. This is crucial for reproducibility.
    *   **Primary Key:** `input_id`
    *   **Foreign Key:** `id_employee` -> `employees.id_employee`

*   **`model_outputs`**
    *   **Purpose:** Stores the results of a model prediction, including the probability, risk category, and final label.
    *   **Primary Key:** `output_id`

*   **`predictions_traceability`**
    *   **Purpose:** Acts as a central log, linking inputs to outputs. It stores metadata about the prediction, such as the model version used, the source of the request (e.g., 'API', 'Streamlit'), and other request-specific details.
    *   **Primary Key:** `trace_id`
    *   **Foreign Keys:**
        *   `input_id` -> `model_inputs.input_id`
        *   `output_id` -> `model_outputs.output_id`

## Data Flow

1.  **Initial Data Load:**
    *   The `database/init_db.py` script is run.
    *   It reads raw data from CSV files located in the `data/` directory.
    *   The data is cleaned, merged, and has features engineered.
    *   The processed employee data is loaded into the `employees` table.

2.  **Prediction via API:**
    *   A client sends a POST request with employee data to a prediction endpoint (e.g., `/predict`).
    *   The FastAPI application receives the data.
    *   The application creates a new record in the `model_inputs` table with the features from the request.
    *   The model is loaded and makes a prediction.
    *   The prediction result is stored in a new record in the `model_outputs` table.
    *   A `predictions_traceability` record is created, linking the `input_id` and `output_id`, and storing metadata like the model version.
    *   The prediction result is returned to the client.

3.  **Prediction via Streamlit UI:**
    *   A user uploads CSV files through the web interface.
    *   The Streamlit application processes the data and calls the model to get predictions.
    *   *(Currently, the Streamlit app does not log to the database, but it could be extended to do so by making requests to the FastAPI backend).*
    *   The results (predictions, SHAP plots) are displayed to the user.
