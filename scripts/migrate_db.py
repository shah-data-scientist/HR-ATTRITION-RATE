import os
from typing import Optional

from sqlalchemy import create_engine, text


def get_engine() -> Optional[object]:
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


def get_column_type(engine, table: str, column: str) -> Optional[str]:
    sql = text("""
        SELECT data_type
        FROM information_schema.columns
        WHERE table_name = :table AND column_name = :column
        """)
    with engine.connect() as conn:
        res = conn.execute(sql, {"table": table, "column": column}).fetchone()
        return res[0] if res else None


def column_exists(engine, table: str, column: str) -> bool:
    sql = text("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = :table AND column_name = :column
        LIMIT 1
        """)
    with engine.connect() as conn:
        res = conn.execute(sql, {"table": table, "column": column}).fetchone()
        return res is not None


def alter_ayant_enfants_to_varchar(engine) -> None:
    current = get_column_type(engine, "employees", "ayant_enfants")
    print(f"employees.ayant_enfants current type: {current}")
    if current and current.lower() in {"integer", "smallint", "bigint"}:
        print("Altering employees.ayant_enfants to VARCHAR ...")
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE employees ALTER COLUMN ayant_enfants TYPE varchar USING ayant_enfants::varchar"
                )
            )
        print("Done.")
    else:
        print("No change required for employees.ayant_enfants.")


def ensure_threshold_column(engine) -> None:
    if not column_exists(engine, "model_outputs", "threshold"):
        print("Adding model_outputs.threshold (FLOAT) ...")
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE model_outputs ADD COLUMN threshold FLOAT"))
        print("Done.")
    else:
        print("Column model_outputs.threshold already exists.")


def ensure_frequence_deplacement_varchar(engine) -> None:
    current = get_column_type(engine, "employees", "frequence_deplacement")
    print(f"employees.frequence_deplacement current type: {current}")
    if current and current.lower() not in {"character varying", "text", "varchar"}:
        print("Altering employees.frequence_deplacement to VARCHAR ...")
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE employees ALTER COLUMN frequence_deplacement TYPE varchar USING frequence_deplacement::varchar"
                )
            )
        print("Done.")
    else:
        print("No change required for employees.frequence_deplacement.")


def ensure_genre_varchar(engine) -> None:
    current = get_column_type(engine, "employees", "genre")
    print(f"employees.genre current type: {current}")
    if current and current.lower() in {"integer", "smallint", "bigint"}:
        print("Altering employees.genre to VARCHAR ...")
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE employees ALTER COLUMN genre TYPE varchar USING genre::varchar"
                )
            )
        print("Done.")
    else:
        print("No change required for employees.genre.")


def main():
    engine = get_engine()
    if engine is None:
        return

    print("Running DB migrations (idempotent)...")
    alter_ayant_enfants_to_varchar(engine)
    ensure_frequence_deplacement_varchar(engine)
    ensure_genre_varchar(engine)
    ensure_threshold_column(engine)
    print("Migrations completed.")


if __name__ == "__main__":
    main()
