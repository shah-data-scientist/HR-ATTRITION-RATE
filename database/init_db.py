import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

# Add the project root to sys.path to allow importing modules from the root
# Add the project root to sys.path to allow importing modules from the root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import load_and_merge_data  # Import the merged data loading function

from .database import Base, SessionLocal, engine
from .models import Employee


def init_db():
    """Initialize the database and load initial employee data if the table is empty."""
    # Create all tables defined in Base
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")

    # Load initial data into the employees table
    db: Session = SessionLocal()
    try:
        # Check if employees table is empty
        if db.query(Employee).count() == 0:
            print("Loading initial employee data...")
            # Define paths to your CSV files
            data_dir = Path(__file__).parent.parent / "data"
            eval_file = data_dir / "extrait_eval.csv"
            sirh_file = data_dir / "extrait_sirh.csv"
            sondage_file = data_dir / "extrait_sondage.csv"

            # Load raw dataframes
            eval_df = pd.read_csv(eval_file)
            sirh_df = pd.read_csv(sirh_file)
            sondage_df = pd.read_csv(sondage_file)

            # Merge and clean data using the utility function
            merged_df = load_and_merge_data(eval_df, sirh_df, sondage_df)

            # Prepare data for insertion
            employees_to_add = []
            for _index, row in merged_df.iterrows():
                # Dynamically create a dictionary for employee attributes
                employee_data = {
                    col: row[col] if pd.notna(row[col]) else None
                    for col in row.index
                    if col not in ["id_employee", "a_quitte_l_entreprise"]
                }
                # Ensure id_employee is an integer
                employee_id = (
                    int(row["id_employee"]) if pd.notna(row["id_employee"]) else None
                )

                # Create Employee object
                employee = Employee(
                    id_employee=employee_id,
                    **employee_data,
                    date_ingestion=datetime.now(UTC),
                )
                employees_to_add.append(employee)

            db.add_all(employees_to_add)
            db.commit()
            print(
                f"Successfully loaded {len(employees_to_add)} employees into the database."
            )
        else:
            print("Employees table is not empty. Skipping initial data load.")
    except Exception as e:
        db.rollback()
        print(f"Error during database initialization or data load: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
