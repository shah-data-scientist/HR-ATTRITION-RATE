import os
import pandas as pd
from sqlalchemy.orm import Session
from datetime import datetime
import sys

# Add the project root to sys.path to allow importing modules from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from .database import Base, engine, SessionLocal
from .models import Employee, ModelInput, ModelOutput, PredictionTraceability
from utils import load_and_merge_data  # Import the merged data loading function


def init_db():
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
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
            eval_file = os.path.join(data_dir, "extrait_eval.csv")
            sirh_file = os.path.join(data_dir, "extrait_sirh.csv")
            sondage_file = os.path.join(data_dir, "extrait_sondage.csv")

            # Load raw dataframes
            eval_df = pd.read_csv(eval_file)
            sirh_df = pd.read_csv(sirh_file)
            sondage_df = pd.read_csv(sondage_file)

            # Merge and clean data using the utility function
            merged_df = load_and_merge_data(eval_df, sirh_df, sondage_df)

            # Prepare data for insertion
            employees_to_add = []
            for index, row in merged_df.iterrows():
                # Dynamically create a dictionary for employee attributes
                employee_data = {
                    col: row[col] if pd.notna(row[col]) else None
                    for col in row.index
                    if col != "id_employee"
                    and col
                    != "a_quitte_l_entreprise"  # Exclude id_employee and target variable
                }
                # Ensure id_employee is an integer
                employee_id = (
                    int(row["id_employee"]) if pd.notna(row["id_employee"]) else None
                )

                # Create Employee object
                employee = Employee(
                    id_employee=employee_id,
                    **employee_data,
                    date_ingestion=datetime.now(),
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
