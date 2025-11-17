from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from .database import Base


class Job(Base):
    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_type = Column(String, nullable=False)  # e.g., 'report'
    status = Column(
        String, nullable=False, default="queued"
    )  # queued|processing|completed|failed
    payload_json = Column(JSON, nullable=False)
    result_json = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    user_id = Column(String(50), nullable=False, default="demo1")  # User who created the job
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), server_default=func.now()
    )


class Employee(Base):
    __tablename__ = "employees"

    id_employee = Column(Integer, primary_key=True, index=True)
    age = Column(Integer, nullable=True)
    genre = Column(String, nullable=True)  # Raw string ('M'/'F', 'H'/'F')
    revenu_mensuel = Column(Float, nullable=True)
    statut_marital = Column(String, nullable=True)
    departement = Column(String, nullable=True)
    poste = Column(String, nullable=True)
    nombre_experiences_precedentes = Column(Integer, nullable=True)
    annee_experience_totale = Column(Integer, nullable=True)
    annees_dans_l_entreprise = Column(
        Float, nullable=True
    )  # Changed to Float for consistency with utils.py
    annees_dans_le_poste_actuel = Column(
        Float, nullable=True
    )  # Changed to Float for consistency with utils.py
    nombre_participation_pee = Column(Integer, nullable=True)
    nb_formations_suivies = Column(Integer, nullable=True)
    nombre_employee_sous_responsabilite = Column(Integer, nullable=True)
    distance_domicile_travail = Column(Integer, nullable=True)
    niveau_education = Column(Integer, nullable=True)
    domaine_etude = Column(String, nullable=True)
    ayant_enfants = Column(String, nullable=True)  # Raw string ('Y'/'N', 'Oui'/'Non')
    frequence_deplacement = Column(
        String, nullable=True
    )  # Raw string ('Occasionnel', 'Frequent', etc.)
    annees_depuis_la_derniere_promotion = Column(Integer, nullable=True)
    annes_sous_responsable_actuel = Column(Integer, nullable=True)
    satisfaction_employee_environnement = Column(Float, nullable=True)
    note_evaluation_precedente = Column(Float, nullable=True)
    niveau_hierarchique_poste = Column(Float, nullable=True)
    satisfaction_employee_nature_travail = Column(Float, nullable=True)
    satisfaction_employee_equipe = Column(Float, nullable=True)
    satisfaction_employee_equilibre_pro_perso = Column(Float, nullable=True)
    note_evaluation_actuelle = Column(Float, nullable=True)
    heures_supplementaires = Column(Integer, nullable=True)
    augmentation_salaire_precedente = Column(
        String, nullable=True
    )  # Raw string like '11 %', '13 %'
    augementation_salaire_precedente = Column(
        String, nullable=True
    )  # Keeping the typo version for now to match input - raw percentage strings
    nombre_heures_travailless = Column(Float, nullable=True)  # From sirh_df
    improvement_evaluation = Column(Float, nullable=True)  # Engineered feature
    total_satisfaction = Column(Float, nullable=True)  # Engineered feature
    work_mobility = Column(Float, nullable=True)  # Engineered feature
    user_id = Column(String(50), nullable=False, default="demo1")  # User who created the record
    date_ingestion = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    model_inputs = relationship("ModelInput", back_populates="employee")


class ModelInput(Base):
    __tablename__ = "model_inputs"

    input_id = Column(Integer, primary_key=True, index=True)
    id_employee = Column(Integer, ForeignKey("employees.id_employee"), nullable=False)
    prediction_timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Example features - these will need to be dynamically generated or carefully managed
    # For simplicity, let's add a generic JSON column for features for now
    # In a real scenario, you might have individual columns for each feature
    features = Column(JSON, nullable=False)

    employee = relationship("Employee", back_populates="model_inputs")
    trace = relationship(
        "PredictionTraceability", back_populates="model_input", uselist=False
    )


class ModelOutput(Base):
    __tablename__ = "model_outputs"

    output_id = Column(Integer, primary_key=True, index=True)
    prediction_proba = Column(Float, nullable=False)
    risk_category = Column(String, nullable=False)
    prediction_label = Column(String, nullable=False)
    log_odds = Column(Float, nullable=False)
    threshold = Column(Float, nullable=True)  # Added threshold column
    prediction_timestamp = Column(DateTime(timezone=True), server_default=func.now())

    trace = relationship(
        "PredictionTraceability", back_populates="model_output", uselist=False
    )


class PredictionTraceability(Base):
    __tablename__ = "predictions_traceability"

    trace_id = Column(Integer, primary_key=True, index=True)
    input_id = Column(Integer, ForeignKey("model_inputs.input_id"), nullable=False)
    output_id = Column(Integer, ForeignKey("model_outputs.output_id"), nullable=False)
    model_version = Column(String, nullable=True)
    prediction_source = Column(String, nullable=True)
    request_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    model_input = relationship("ModelInput", back_populates="trace")
    model_output = relationship("ModelOutput", back_populates="trace")
    shap_analysis = relationship(
        "ShapAnalysis", back_populates="trace", uselist=False
    )


class ShapAnalysis(Base):
    __tablename__ = "shap_analysis"

    shap_id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(Integer, ForeignKey("predictions_traceability.trace_id"), nullable=False, unique=True)
    shap_values = Column(JSON, nullable=False)  # Array of SHAP values
    base_value = Column(Float, nullable=False)  # Expected value from explainer
    feature_names = Column(JSON, nullable=False)  # Array of feature names
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    trace = relationship("PredictionTraceability", back_populates="shap_analysis")
