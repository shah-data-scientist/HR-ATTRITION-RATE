from database.database import SessionLocal
from database.models import Employee, PredictionTraceability, ModelInput, ModelOutput
from datetime import datetime, timedelta

db = SessionLocal()

# Check last 10 minutes
cutoff = datetime.now() - timedelta(minutes=10)

recent_employees = db.query(Employee).filter(Employee.date_ingestion >= cutoff).count()
recent_inputs = (
    db.query(ModelInput).filter(ModelInput.prediction_timestamp >= cutoff).count()
)
recent_outputs = (
    db.query(ModelOutput).filter(ModelOutput.prediction_timestamp >= cutoff).count()
)
recent_traces = (
    db.query(PredictionTraceability)
    .filter(PredictionTraceability.created_at >= cutoff)
    .count()
)

print(f"\n=== New entries in last 10 minutes ===")
print(f"Employees: {recent_employees}")
print(f"Model Inputs: {recent_inputs}")
print(f"Model Outputs: {recent_outputs}")
print(f"Prediction Traces: {recent_traces}")

if recent_inputs > 0:
    latest = (
        db.query(ModelInput).order_by(ModelInput.prediction_timestamp.desc()).first()
    )
    print(f"\nMost recent prediction: {latest.prediction_timestamp}")
    print(f"Time difference from now: {datetime.now() - latest.prediction_timestamp}")
else:
    # Show the most recent entry overall
    latest = (
        db.query(ModelInput).order_by(ModelInput.prediction_timestamp.desc()).first()
    )
    if latest:
        print(f"\nMost recent prediction ever: {latest.prediction_timestamp}")
        print(
            f"Time difference from now: {datetime.now() - latest.prediction_timestamp}"
        )

db.close()
