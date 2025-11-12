"""Pydantic schemas for API input validation.

This module defines the Pydantic models for validating raw input data
before it goes through cleaning and feature engineering.
"""

from pydantic import BaseModel, ConfigDict, Field


class EmployeeInputSchema(BaseModel):
    """Raw employee input schema (before cleaning/transformation).

    This matches the canonical input format from the requirements.
    """

    # Satisfaction & Evaluation (1-4 range)
    satisfaction_employee_environnement: int = Field(
        ..., ge=1, le=4, description="Employee satisfaction with environment (1-4)"
    )
    note_evaluation_precedente: int = Field(
        ..., ge=1, le=4, description="Previous evaluation score (1-4)"
    )
    satisfaction_employee_nature_travail: int = Field(
        ..., ge=1, le=4, description="Employee satisfaction with nature of work (1-4)"
    )
    satisfaction_employee_equipe: int = Field(
        ..., ge=1, le=4, description="Employee satisfaction with team (1-4)"
    )
    satisfaction_employee_equilibre_pro_perso: int = Field(
        ..., ge=1, le=4, description="Employee satisfaction with work-life balance (1-4)"
    )
    note_evaluation_actuelle: int = Field(
        ..., ge=1, le=4, description="Current evaluation score (1-4)"
    )

    # Hierarchical level (1-5)
    niveau_hierarchique_poste: int = Field(
        ..., ge=1, le=5, description="Hierarchical level of the position (1-5)"
    )

    # Overtime (categorical "Oui"/"Non")
    heure_supplementaires: str = Field(
        ..., description="Overtime ('Oui' or 'Non')"
    )

    # Salary increase (0.0-35.0 percentage)
    augementation_salaire_precedente: float = Field(
        ..., ge=0.0, le=35.0, description="Previous salary increase percentage (0.0-35.0)"
    )

    # Employee identifiers
    id_employee: int = Field(..., ge=0, description="Unique employee identifier")
    eval_number: str | None = Field(None, description="Evaluation number (e.g., 'E_123')")

    # Demographics
    age: int = Field(..., ge=18, le=70, description="Employee age (18-70)")
    genre: str = Field(..., description="Gender ('Homme', 'Femme', etc.)")

    # Financial
    revenu_mensuel: float = Field(..., ge=0, description="Monthly income")

    # Categorical attributes
    statut_marital: str = Field(..., description="Marital status")
    departement: str = Field(..., description="Department")
    poste: str = Field(..., description="Job position")
    domaine_etude: str = Field(..., description="Field of study")
    frequence_deplacement: str = Field(..., description="Travel frequency")

    # Work experience & tenure
    nombre_experiences_precedentes: int = Field(
        ..., ge=0, description="Number of previous work experiences"
    )
    nombre_heures_travailless: int = Field(
        ..., ge=0, description="Number of work hours"
    )
    annee_experience_totale: int = Field(
        ..., ge=0, description="Total years of work experience"
    )
    annees_dans_l_entreprise: int = Field(
        ..., ge=0, description="Years in the company"
    )
    annees_dans_le_poste_actuel: int = Field(
        ..., ge=0, description="Years in current position"
    )
    annees_depuis_la_derniere_promotion: int = Field(
        ..., ge=0, description="Years since last promotion"
    )
    annes_sous_responsable_actuel: int = Field(
        ..., ge=0, description="Years under current manager"
    )

    # Company benefits & development
    nombre_participation_pee: int = Field(
        ..., ge=0, description="Number of PEE participations"
    )
    nb_formations_suivies: int = Field(
        ..., ge=0, le=3, description="Number of trainings attended (0-3)"
    )

    # Management
    nombre_employee_sous_responsabilite: int = Field(
        ..., ge=0, le=20, description="Number of employees under responsibility (0-20)"
    )

    # Survey code (constant)
    code_sondage: str | None = Field(None, description="Survey code (usually '80')")

    # Other
    distance_domicile_travail: int = Field(
        ..., ge=0, le=50, description="Distance from home to work (0-50 km)"
    )
    niveau_education: int = Field(
        ..., ge=1, le=5, description="Education level (1-5)"
    )
    ayant_enfants: str | int = Field(
        ..., description="Has children ('Oui'/'Non' or 0/1)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "satisfaction_employee_environnement": 3,
                "note_evaluation_precedente": 3,
                "niveau_hierarchique_poste": 2,
                "satisfaction_employee_nature_travail": 4,
                "satisfaction_employee_equipe": 3,
                "satisfaction_employee_equilibre_pro_perso": 3,
                "note_evaluation_actuelle": 4,
                "heure_supplementaires": "Non",
                "augementation_salaire_precedente": 7.0,
                "id_employee": 12345,
                "eval_number": "E_12345",
                "age": 35,
                "genre": "Homme",
                "revenu_mensuel": 6000.0,
                "statut_marital": "Marié",
                "departement": "IT",
                "poste": "Développeur",
                "nombre_experiences_precedentes": 2,
                "nombre_heures_travailless": 190,
                "annee_experience_totale": 10,
                "annees_dans_l_entreprise": 5,
                "annees_dans_le_poste_actuel": 3,
                "nombre_participation_pee": 1,
                "nb_formations_suivies": 2,
                "nombre_employee_sous_responsabilite": 0,
                "code_sondage": "80",
                "distance_domicile_travail": 15,
                "niveau_education": 3,
                "domaine_etude": "Informatique",
                "ayant_enfants": "Oui",
                "frequence_deplacement": "Rarement",
                "annees_depuis_la_derniere_promotion": 2,
                "annes_sous_responsable_actuel": 2,
            }
        }
    )


class PredictionOutput(BaseModel):
    """Prediction output schema."""

    id_employee: int = Field(..., description="Employee ID")
    prediction: str = Field(..., description="Predicted outcome ('Stay' or 'Leave')")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Attrition probability"
    )
    risk_category: str = Field(..., description="Risk category ('Low', 'Medium', 'High')")
    message: str = Field(..., description="Descriptive message about the prediction")
    trace_id: int | None = Field(None, description="Traceability ID")


class BatchPredictionInput(BaseModel):
    """Batch prediction input schema."""

    employees: list[EmployeeInputSchema] = Field(
        ..., description="List of employee records"
    )


class BatchPredictionOutput(BaseModel):
    """Batch prediction output schema."""

    predictions: list[PredictionOutput] = Field(..., description="List of predictions")
