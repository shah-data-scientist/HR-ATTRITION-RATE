import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Database connection URL
# Read from environment variable, or use a default for local development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///hr_attrition.db")
DISABLE_DB = os.getenv("DISABLE_DB", "0") == "1"

# Create the SQLAlchemy engine
if DATABASE_URL == "sqlite:///:memory:":
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
elif DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
        pool_recycle=1800,
    )
else:
    # Add safe defaults for Postgres/MySQL connection pooling and keepalives
    connect_args = {}
    if DATABASE_URL.startswith("postgresql"):
        # TCP keepalive settings (server-friendly; ignored if not supported)
        connect_args = {
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args=connect_args,
    )

# Create a SessionLocal class
SessionLocal = (
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
    if not DISABLE_DB
    else None
)

# Base class for declarative models
Base = declarative_base()


# Dependency to get a database session
def get_db():
    """Provide a database session, or yield None when DB is disabled."""
    if DISABLE_DB:
        # Yield a placeholder None to satisfy FastAPI dependency
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
