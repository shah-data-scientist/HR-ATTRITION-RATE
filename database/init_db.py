import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

# Add the project root to sys.path to allow importing modules from the root
# Add the project root to sys.path to allow importing modules from the root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils import load_and_merge_data  # Import the merged data loading function

from .database import Base, SessionLocal, engine
from .models import Employee, User


def create_default_users(db: Session):
    """Create default users if users table is empty."""
    try:
        # Check if users table is empty
        if db.query(User).count() == 0:
            print("Creating default users...")

            # Get credentials from environment variables
            admin_username = os.getenv("UI_ADMIN_USERNAME", "admin")
            admin_password = os.getenv("UI_ADMIN_PASSWORD", "Admin@2025!Secure")
            user_username = os.getenv("UI_USER_USERNAME", "analyst")
            user_password = os.getenv("UI_USER_PASSWORD", "Analyst@2025!View")

            # Create admin user
            admin_user = User(
                username=admin_username,
                password_hash=User.hash_password(admin_password),
                role="admin",
                is_active=1
            )

            # Create regular user
            regular_user = User(
                username=user_username,
                password_hash=User.hash_password(user_password),
                role="user",
                is_active=1
            )

            db.add(admin_user)
            db.add(regular_user)
            db.commit()
            print(f"Successfully created 2 default users:")
            print(f"  - Admin user: {admin_username} (role: admin)")
            print(f"  - Regular user: {user_username} (role: user)")
        else:
            print("Users table is not empty. Skipping default users creation.")
    except Exception as e:
        db.rollback()
        print(f"Error creating default users: {e}")
        raise e


def init_db():
    """Initialize the database and load initial employee data if the table is empty."""
    try:
        # Create all tables defined in Base
        print("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("Database tables created.")
    except Exception as e:
        if "psycopg2" in str(e):
            print("\n--- DATABASE CONNECTION ERROR ---")
            print("Could not connect to the PostgreSQL database.")
            print("This script is configured to use PostgreSQL by default.")
            print("\nPlease choose one of the following options:")
            print("1. Make sure you have a PostgreSQL server running and accessible.")
            print(
                "2. To use a local SQLite database instead, open the '.env' file and comment out the DATABASE_URL line."
            )
            print(
                "   The application will automatically create and use a 'hr_attrition.db' file in the project root."
            )
            print("\nOriginal error:", e)
            return
        else:
            raise e

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
                    date_ingestion=datetime.now(timezone.utc),
                )
                employees_to_add.append(employee)

            db.add_all(employees_to_add)
            db.commit()
            print(
                f"Successfully loaded {len(employees_to_add)} employees into the database."
            )
        else:
            print("Employees table is not empty. Skipping initial data load.")

        # Create default users for UI authentication
        create_default_users(db)

    except Exception as e:
        db.rollback()
        print(f"Error during database initialization or data load: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
