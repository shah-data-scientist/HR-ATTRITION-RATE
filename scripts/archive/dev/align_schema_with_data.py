"""
Align database schema with actual CSV data types.
Run this after checking the actual data in CSV files.
"""

import os
from sqlalchemy import create_engine, text


def get_engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL is not set.")
        return None
    try:
        eng = create_engine(url)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return eng
    except Exception as e:
        print(f"ERROR: Could not connect to database: {e}")
        return None


def main():
    engine = get_engine()
    if engine is None:
        return

    print("Aligning schema with CSV data types...")

    # Based on actual CSV inspection:
    # - genre: 'F', 'M' (strings)
    # - ayant_enfants: 'Y', 'N' (strings)
    # - frequence_deplacement: 'Occasionnel', 'Frequent', 'Aucun' (strings)
    # - statut_marital: 'Célibataire', 'Marié(e)', 'Divorcé(e)' (strings)
    # - departement: 'Commercial', 'Consulting', etc. (strings)
    # - poste: 'Cadre Commercial', 'Assistant de Direction', etc. (strings)
    # - domaine_etude: various (strings)
    # - augementation_salaire_precedente: '11 %', '13 %', etc. (strings with %)

    migrations = [
        ("genre", "VARCHAR", "Store raw gender values 'M'/'F'"),
        ("ayant_enfants", "VARCHAR", "Store raw values 'Y'/'N'"),
        (
            "frequence_deplacement",
            "VARCHAR",
            "Store raw values 'Occasionnel'/'Frequent'/'Aucun'",
        ),
        (
            "augementation_salaire_precedente",
            "VARCHAR",
            "Store raw percentage strings like '11 %'",
        ),
    ]

    with engine.begin() as conn:
        for col, target_type, reason in migrations:
            print(f"\nProcessing {col}...")
            print(f"  Reason: {reason}")

            # Check current type
            result = conn.execute(text(f"""
                SELECT data_type 
                FROM information_schema.columns 
                WHERE table_name='employees' AND column_name='{col}'
            """)).fetchone()

            if result:
                current_type = result[0]
                print(f"  Current type: {current_type}")

                if current_type.lower() not in ["character varying", "text", "varchar"]:
                    print(f"  Converting to {target_type}...")
                    conn.execute(text(f"""
                        ALTER TABLE employees 
                        ALTER COLUMN {col} TYPE {target_type} 
                        USING {col}::VARCHAR
                    """))
                    print(f"  ✓ {col} converted to {target_type}")
                else:
                    print(f"  ✓ {col} already VARCHAR")
            else:
                print(f"  ⚠ Column {col} not found")

    print("\n✅ Schema alignment complete!")


if __name__ == "__main__":
    main()
