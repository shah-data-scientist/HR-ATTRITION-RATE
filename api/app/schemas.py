from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List


class EmployeeFeatures(BaseModel):
    id_employee: Optional[int] = Field(
        None, description="Unique identifier for the employee"
    )
    age: int = Field(..., ge=18, le=65, description="Age of the employee")
    genre: Optional[int] = Field(
        None, description="Gender (0 for Female, 1 for Male)"
    )  # Can be 0 or 1
    revenu_mensuel: float = Field(..., ge=0, description="Monthly income")
    statut_marital: str = Field(
        ..., description="Marital status (e.g., 'Célibataire', 'Marié', 'Divorcé')"
    )
    departement: str = Field(
        ..., description="Department (e.g., 'R&D', 'Ventes', 'RH')"
    )
    poste: str = Field(
        ..., description="Job role (e.g., 'Manager', 'Développeur', 'Analyste')"
    )
    nombre_experiences_precedentes: int = Field(
        ..., ge=0, description="Number of previous work experiences"
    )
    annee_experience_totale: int = Field(
        ..., ge=0, description="Total years of work experience"
    )
    annees_dans_l_entreprise: int = Field(
        ..., ge=0, description="Years spent in the current company"
    )
    annees_dans_le_poste_actuel: int = Field(
        ..., ge=0, description="Years in current job role"
    )
    nombre_participation_pee: int = Field(
        ..., ge=0, description="Number of participations in PEE"
    )
    nb_formations_suivies: int = Field(
        ..., ge=0, description="Number of trainings attended"
    )
    nombre_employee_sous_responsabilite: int = Field(
        ..., ge=0, description="Number of employees under responsibility"
    )
    distance_domicile_travail: int = Field(
        ..., ge=1, description="Distance from home to work (km)"
    )
    niveau_education: int = Field(
        ..., ge=1, le=5, description="Education level (1 to 5)"
    )
    domaine_etude: str = Field(
        ..., description="Field of study (e.g., 'Sciences', 'Marketing', 'Technique')"
    )
    ayant_enfants: Optional[int] = Field(
        None, description="Has children (0 for No, 1 for Yes)"
    )  # Can be 0 or 1
    frequence_deplacement: str = Field(
        ...,
        description="Frequency of travel (e.g., 'Rarement', 'Fréquemment', 'Jamais')",
    )
    annees_depuis_la_derniere_promotion: int = Field(
        ..., ge=0, description="Years since last promotion"
    )
    annes_sous_responsable_actuel: int = Field(
        ..., ge=0, description="Years under current manager"
    )

    # Features that might be directly provided or engineered
    satisfaction_employee_environnement: int = Field(
        ..., ge=1, le=4, description="Employee satisfaction with environment (1-4)"
    )
    note_evaluation_precedente: float = Field(
        ..., ge=0.0, le=5.0, description="Previous evaluation score"
    )
    niveau_hierarchique_poste: int = Field(
        ..., ge=1, le=5, description="Hierarchical level of the position (1-5)"
    )
    satisfaction_employee_nature_travail: int = Field(
        ..., ge=1, le=4, description="Employee satisfaction with nature of work (1-4)"
    )
    satisfaction_employee_equipe: int = Field(
        ..., ge=1, le=4, description="Employee satisfaction with team (1-4)"
    )
    satisfaction_employee_equilibre_pro_perso: int = Field(
        ...,
        ge=1,
        le=4,
        description="Employee satisfaction with work-life balance (1-4)",
    )
    note_evaluation_actuelle: float = Field(
        ..., ge=0.0, le=5.0, description="Current evaluation score"
    )
    heures_supplementaires: Optional[int] = Field(
        None, description="Overtime (0 for No, 1 for Yes)"
    )  # Can be 0 or 1
    augementation_salaire_precedente: float = Field(
        ...,
        ge=0.0,
        description="Previous salary increase percentage (e.g., 0.05 for 5%)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "augementation_salaire_precedente": 0.07,
            }
        }
    )


class PredictionOutput(BaseModel):
    id_employee: int = Field(..., description="Unique identifier for the employee")
    prediction: str = Field(
        ..., description="Predicted outcome (e.g., 'Stay' or 'Leave')"
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted probability of attrition"
    )
    risk_category: str = Field(
        ..., description="Categorized risk level (e.g., 'Low', 'Medium', 'High')"
    )
    message: str = Field(..., description="A descriptive message about the prediction")
    trace_id: Optional[int] = Field(
        None, description="ID for prediction traceability record"
    )


class BatchPredictionInput(BaseModel):
    employees: List[EmployeeFeatures] = Field(
        ..., description="List of employee feature sets for batch prediction"
    )


class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput] = Field(
        ..., description="List of predictions for each employee in the batch"
    )
