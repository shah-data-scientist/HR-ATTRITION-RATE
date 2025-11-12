"""Tests for database models and operations."""

import json
from datetime import datetime

import pytest

from database.models import Employee, ModelInput, ModelOutput, PredictionTraceability


class TestEmployeeModel:
    """Test Employee database model."""

    def test_create_employee(self, test_db_session):
        """Test creating an employee record."""
        employee = Employee(
            id_employee=10001,
            age=35,
            genre=1,
            revenu_mensuel=6000,
            departement="IT",
            date_ingestion=datetime.now(),
        )
        test_db_session.add(employee)
        test_db_session.commit()

        retrieved = (
            test_db_session.query(Employee).filter_by(id_employee=10001).first()
        )
        assert retrieved is not None
        assert retrieved.age == 35
        assert retrieved.departement == "IT"

    def test_employee_unique_id(self, test_db_session):
        """Test that id_employee is unique."""
        emp1 = Employee(id_employee=10001, age=30, date_ingestion=datetime.now())
        emp2 = Employee(id_employee=10001, age=40, date_ingestion=datetime.now())

        test_db_session.add(emp1)
        test_db_session.commit()

        test_db_session.add(emp2)
        with pytest.raises(Exception):  # IntegrityError
            test_db_session.commit()

    def test_employee_with_engineered_features(self, test_db_session):
        """Test employee record with engineered features."""
        employee = Employee(
            id_employee=10002,
            age=35,
            improvement_evaluation=1.0,
            total_satisfaction=36.0,
            work_mobility=0.6,
            date_ingestion=datetime.now(),
        )
        test_db_session.add(employee)
        test_db_session.commit()

        retrieved = (
            test_db_session.query(Employee).filter_by(id_employee=10002).first()
        )
        assert retrieved.improvement_evaluation == 1.0
        assert retrieved.total_satisfaction == 36.0
        assert retrieved.work_mobility == 0.6


class TestModelInputOutput:
    """Test ModelInput and ModelOutput models."""

    def test_create_model_input(self, test_db_session, sample_employee_in_db):
        """Test creating a model input record."""
        features = {"age": 35, "departement": "IT", "revenu_mensuel": 6000}
        model_input = ModelInput(
            id_employee=sample_employee_in_db.id_employee,
            features=json.dumps(features),
            prediction_timestamp=datetime.now(),
        )
        test_db_session.add(model_input)
        test_db_session.commit()

        retrieved = test_db_session.query(ModelInput).first()
        assert retrieved is not None
        assert retrieved.id_employee == sample_employee_in_db.id_employee
        assert json.loads(retrieved.features)["age"] == 35

    def test_create_model_output(self, test_db_session):
        """Test creating a model output record."""
        model_output = ModelOutput(
            prediction_proba=0.75,
            risk_category="High",
            prediction_label="Leave",
            log_odds=1.098,
            prediction_timestamp=datetime.now(),
        )
        test_db_session.add(model_output)
        test_db_session.commit()

        retrieved = test_db_session.query(ModelOutput).first()
        assert retrieved is not None
        assert retrieved.prediction_proba == 0.75
        assert retrieved.risk_category == "High"
        assert retrieved.prediction_label == "Leave"

    def test_model_input_foreign_key(self, test_db_session, sample_employee_in_db):
        """Test foreign key relationship between ModelInput and Employee."""
        features = {"test": "data"}
        model_input = ModelInput(
            id_employee=sample_employee_in_db.id_employee,
            features=json.dumps(features),
            prediction_timestamp=datetime.now(),
        )
        test_db_session.add(model_input)
        test_db_session.commit()

        # Access relationship
        assert model_input.employee is not None
        assert model_input.employee.id_employee == sample_employee_in_db.id_employee


class TestPredictionTraceability:
    """Test PredictionTraceability model and relationships."""

    def test_create_full_prediction_trace(
        self, test_db_session, sample_employee_in_db
    ):
        """Test creating a complete prediction trace."""
        # 1. Create ModelInput
        features = {"age": 35}
        model_input = ModelInput(
            id_employee=sample_employee_in_db.id_employee,
            features=json.dumps(features),
            prediction_timestamp=datetime.now(),
        )
        test_db_session.add(model_input)
        test_db_session.flush()

        # 2. Create ModelOutput
        model_output = ModelOutput(
            prediction_proba=0.65,
            risk_category="High",
            prediction_label="Leave",
            log_odds=0.619,
            prediction_timestamp=datetime.now(),
        )
        test_db_session.add(model_output)
        test_db_session.flush()

        # 3. Create Traceability
        trace = PredictionTraceability(
            input_id=model_input.input_id,
            output_id=model_output.output_id,
            model_version="1.0.0",
            prediction_source="API",
            request_metadata={"user_agent": "test", "client_host": "localhost"},
            created_at=datetime.now(),
        )
        test_db_session.add(trace)
        test_db_session.commit()

        # Verify relationships
        assert trace.model_input is not None
        assert trace.model_output is not None
        assert trace.model_input.id_employee == sample_employee_in_db.id_employee
        assert trace.model_output.prediction_proba == 0.65

    def test_trace_metadata_json(self, test_db_session, sample_employee_in_db):
        """Test that request_metadata is stored as JSON."""
        # Create minimal input/output
        model_input = ModelInput(
            id_employee=sample_employee_in_db.id_employee,
            features=json.dumps({}),
        )
        test_db_session.add(model_input)
        test_db_session.flush()

        model_output = ModelOutput(
            prediction_proba=0.5,
            risk_category="Medium",
            prediction_label="Stay",
            log_odds=0.0,
        )
        test_db_session.add(model_output)
        test_db_session.flush()

        # Create trace with metadata
        metadata = {"ip": "127.0.0.1", "endpoint": "/predict"}
        trace = PredictionTraceability(
            input_id=model_input.input_id,
            output_id=model_output.output_id,
            model_version="1.0.0",
            prediction_source="API",
            request_metadata=metadata,
        )
        test_db_session.add(trace)
        test_db_session.commit()

        # Retrieve and verify
        retrieved = test_db_session.query(PredictionTraceability).first()
        assert retrieved.request_metadata == metadata
        assert retrieved.request_metadata["ip"] == "127.0.0.1"


class TestDatabaseIntegrity:
    """Test database constraints and integrity."""

    def test_cascade_relationships(self, test_db_session):
        """Test that deleting input/output doesn't break traceability (if configured)."""
        # This test depends on cascade settings in models
        # For now, just verify the structure exists
        employee = Employee(
            id_employee=10003, age=30, date_ingestion=datetime.now()
        )
        test_db_session.add(employee)
        test_db_session.commit()

        model_input = ModelInput(
            id_employee=employee.id_employee, features=json.dumps({})
        )
        test_db_session.add(model_input)
        test_db_session.commit()

        # Verify relationship
        assert employee.model_inputs is not None
        assert len(employee.model_inputs) == 1

    def test_query_predictions_for_employee(self, test_db_session):
        """Test querying all predictions for a specific employee."""
        employee = Employee(
            id_employee=10004, age=40, date_ingestion=datetime.now()
        )
        test_db_session.add(employee)
        test_db_session.commit()

        # Create 2 predictions
        for i in range(2):
            model_input = ModelInput(
                id_employee=employee.id_employee,
                features=json.dumps({"iteration": i}),
            )
            test_db_session.add(model_input)

        test_db_session.commit()

        # Query all inputs for this employee
        inputs = (
            test_db_session.query(ModelInput)
            .filter_by(id_employee=employee.id_employee)
            .all()
        )
        assert len(inputs) == 2
