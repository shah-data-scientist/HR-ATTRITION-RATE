"""Seed the database with sample employee data for testing.

This script creates sample employee records for testing the API and UI.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.database import Base, engine, SessionLocal
from database.models import Employee

# Sample employee data (minimal for testing)
SAMPLE_EMPLOYEES = [
    {
        "id_employee": 10001,
        "age": 35,
        "genre": 1,  # Male (after cleaning)
        "revenu_mensuel": 6000,
        "departement": "IT",
        "poste": "Développeur",
        "statut_marital": "Marié",
        "niveau_education": 3,
        "domaine_etude": "Informatique",
        "ayant_enfants": "Oui",
        "frequence_deplacement": "Rarement",
        "annee_experience_totale": 10,
        "annees_dans_l_entreprise": 5,
        "annees_dans_le_poste_actuel": 3,
        "nombre_experiences_precedentes": 2,
        "nombre_participation_pee": 1,
        "nb_formations_suivies": 2,
        "nombre_employee_sous_responsabilite": 0,
        "distance_domicile_travail": 15,
        "annees_depuis_la_derniere_promotion": 2,
        "annes_sous_responsable_actuel": 2,
        "satisfaction_employee_environnement": 3.0,
        "note_evaluation_precedente": 3.0,
        "niveau_hierarchique_poste": 2.0,
        "satisfaction_employee_nature_travail": 4.0,
        "satisfaction_employee_equipe": 3.0,
        "satisfaction_employee_equilibre_pro_perso": 3.0,
        "note_evaluation_actuelle": 4.0,
        "heures_supplementaires": 0,  # No overtime
        "augmentation_salaire_precedente": 0.07,  # 7%
        "improvement_evaluation": 1.0,
        "total_satisfaction": 36.0,
        "work_mobility": 0.6,
    },
    {
        "id_employee": 10002,
        "age": 42,
        "genre": 0,  # Female
        "revenu_mensuel": 7500,
        "departement": "Sales",
        "poste": "Manager",
        "statut_marital": "Célibataire",
        "niveau_education": 4,
        "domaine_etude": "Marketing",
        "ayant_enfants": "Non",
        "frequence_deplacement": "Fréquemment",
        "annee_experience_totale": 15,
        "annees_dans_l_entreprise": 8,
        "annees_dans_le_poste_actuel": 4,
        "nombre_experiences_precedentes": 3,
        "nombre_participation_pee": 2,
        "nb_formations_suivies": 3,
        "nombre_employee_sous_responsabilite": 5,
        "distance_domicile_travail": 25,
        "annees_depuis_la_derniere_promotion": 1,
        "annes_sous_responsable_actuel": 3,
        "satisfaction_employee_environnement": 4.0,
        "note_evaluation_precedente": 4.0,
        "niveau_hierarchique_poste": 3.0,
        "satisfaction_employee_nature_travail": 4.0,
        "satisfaction_employee_equipe": 4.0,
        "satisfaction_employee_equilibre_pro_perso": 2.0,
        "note_evaluation_actuelle": 4.0,
        "heures_supplementaires": 1,  # Overtime
        "augmentation_salaire_precedente": 0.12,  # 12%
        "improvement_evaluation": 0.0,
        "total_satisfaction": 32.0,
        "work_mobility": 0.5,
    },
    {
        "id_employee": 10003,
        "age": 28,
        "genre": 1,  # Male
        "revenu_mensuel": 4500,
        "departement": "HR",
        "poste": "Analyste",
        "statut_marital": "Marié",
        "niveau_education": 2,
        "domaine_etude": "RH",
        "ayant_enfants": "Non",
        "frequence_deplacement": "Rarement",
        "annee_experience_totale": 5,
        "annees_dans_l_entreprise": 2,
        "annees_dans_le_poste_actuel": 2,
        "nombre_experiences_precedentes": 1,
        "nombre_participation_pee": 0,
        "nb_formations_suivies": 1,
        "nombre_employee_sous_responsabilite": 0,
        "distance_domicile_travail": 10,
        "annees_depuis_la_derniere_promotion": 2,
        "annes_sous_responsable_actuel": 1,
        "satisfaction_employee_environnement": 2.0,
        "note_evaluation_precedente": 2.0,
        "niveau_hierarchique_poste": 1.0,
        "satisfaction_employee_nature_travail": 2.0,
        "satisfaction_employee_equipe": 3.0,
        "satisfaction_employee_equilibre_pro_perso": 3.0,
        "note_evaluation_actuelle": 3.0,
        "heures_supplementaires": 0,
        "augmentation_salaire_precedente": 0.05,  # 5%
        "improvement_evaluation": 1.0,
        "total_satisfaction": 18.0,
        "work_mobility": 1.0,
    },
]


def seed_database(drop_existing: bool = False):
    """Seed the database with sample employee data.

    Args:
        drop_existing: If True, drop all tables before seeding
    """
    print("=== Database Seeding Script ===\n")

    # Create tables if they don't exist
    if drop_existing:
        print("Dropping existing tables...")
        Base.metadata.drop_all(bind=engine)
        print("Tables dropped.\n")

    print("Creating tables if they don't exist...")
    Base.metadata.create_all(bind=engine)
    print("Tables created.\n")

    # Create session
    db = SessionLocal()

    try:
        print(f"Seeding {len(SAMPLE_EMPLOYEES)} sample employees...")

        for emp_data in SAMPLE_EMPLOYEES:
            # Check if employee already exists
            existing = (
                db.query(Employee)
                .filter(Employee.id_employee == emp_data["id_employee"])
                .first()
            )

            if existing:
                print(
                    f"  - Employee {emp_data['id_employee']} already exists, skipping"
                )
            else:
                employee = Employee(
                    **emp_data,
                    date_ingestion=datetime.now(),
                )
                db.add(employee)
                print(f"  - Added employee {emp_data['id_employee']}")

        db.commit()
        print(f"\nSeeding complete! {len(SAMPLE_EMPLOYEES)} employees processed.")

        # Verify
        count = db.query(Employee).count()
        print(f"Total employees in database: {count}")

    except Exception as e:
        db.rollback()
        print(f"\nError seeding database: {e}")
        raise
    finally:
        db.close()

    print("\n=== Seeding Complete ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed database with sample data")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing tables before seeding",
    )
    args = parser.parse_args()

    seed_database(drop_existing=args.drop)
