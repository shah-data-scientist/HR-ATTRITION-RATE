# Employee Attrition Prediction API

This API provides predictions for employee attrition risk using a pre-trained machine learning model. It is built with FastAPI, offering automatic interactive documentation (Swagger UI) and robust data validation.

## Features

*   **Prediction Endpoint:** `POST /predict` to get attrition risk predictions for single or multiple employees.
*   **Model Information:** `GET /` for basic API information.
*   **Integrated Documentation:** Automatic Swagger UI (`/docs`) and ReDoc (`/redoc`) documentation.
*   **Data Validation:** Input data is validated using Pydantic models.

## Setup

### Prerequisites

*   Python 3.9+
*   Poetry (for dependency management)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd HR Attrition Rate
    ```

2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```

3.  **Ensure the ML model is available:**
    The API expects a pre-trained model file at `outputs/employee_attrition_pipeline.pkl`. If you haven't trained the model yet, please run the `train.py` script or ensure the file is present.

## Running the API

To start the FastAPI application, navigate to the project root directory (`HR Attrition Rate/`) and run:

```bash
poetry run uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload
```

*   `api.app.main:app`: Specifies that the FastAPI application instance `app` is located in the `main.py` file within the `api/app` directory.
*   `--host 0.0.0.0`: Makes the server accessible from all network interfaces.
*   `--port 8000`: Runs the server on port 8000.
*   `--reload`: Enables auto-reloading on code changes (useful for development).

## Accessing the Documentation

Once the API is running, you can access the automatically generated interactive documentation:

*   **Swagger UI:** Open your web browser and go to `http://127.0.0.1:8000/docs`
*   **ReDoc:** Open your web browser and go to `http://127.0.0.1:8000/redoc`

The Swagger UI allows you to:
*   View all available endpoints.
*   Understand the expected request and response schemas.
*   Try out the endpoints directly from the browser.

## Example Usage (using `/predict` endpoint)

You can use the Swagger UI to test the `/predict` endpoint. Here's an example of a request body for a single employee:

```json
{
  "employees": [
    {
      "id_employee": 12345,
      "age": 35,
      "genre": 1,
      "revenu_mensuel": 6000.0,
      "statut_marital": "Marié",
      "departement": "R&D",
      "poste": "Développeur",
      "nombre_experiences_precedentes": 2,
      "annee_experience_totale": 10,
      "annees_dans_l_entreprise": 5,
      "annees_dans_le_poste_actuel": 3,
      "nombre_participation_pee": 1,
      "nb_formations_suivies": 2,
      "nombre_employee_sous_responsabilite": 0,
      "distance_domicile_travail": 15,
      "niveau_education": 3,
      "domaine_etude": "Informatique",
      "ayant_enfants": 1,
      "frequence_deplacement": "Rarement",
      "annees_depuis_la_derniere_promotion": 2,
      "annes_sous_responsable_actuel": 2,
      "satisfaction_employee_environnement": 3,
      "note_evaluation_precedente": 3.5,
      "niveau_hierarchique_poste": 2,
      "satisfaction_employee_nature_travail": 4,
      "satisfaction_employee_equipe": 3,
      "satisfaction_employee_equilibre_pro_perso": 3,
      "note_evaluation_actuelle": 4.0,
      "heures_supplementaires": 0,
      "augementation_salaire_precedente": 0.07
    }
  ]
}
```

The API will return a response similar to:

```json
{
  "predictions": [
    {
      "id_employee": 12345,
      "prediction": "Leave",
      "probability": 0.78,
      "risk_category": "High",
      "message": "Employee 12345 is predicted to Leave with 78.00% attrition risk (Risk: High)."
    }
  ]
}
```
