"""Core data processing and validation modules."""

from core.data_processing import (
    clean_and_engineer_features,
    clean_raw_input,
    engineer_features,
)
from core.preprocess import enforce_schema, validate_input_ranges
from core.schema import (
    ProcessedBatchPredictionInput,  # Renamed from BatchPredictionInput
    BatchPredictionOutput,
    EmployeeInputSchema,
    PredictionOutput,
    EvalInputSchema,
    SirhInputSchema,
    SondageInputSchema,
    RawBatchPredictionInput,
)
from core.validation import (
    ALL_FEATURE_COLS,
    CATEGORICAL_COLS,
    ENGINEERED_FEATURES,
    EXPECTED_DTYPES,
    NUMERIC_COLS,
    RAW_CATEGORICAL_COLS,
    RAW_INPUT_COLS,
    RAW_NUMERIC_COLS,
    TARGET_COL,
    VALIDATION_RANGES,
)

__all__ = [
    # Data processing
    "clean_and_engineer_features",
    "clean_raw_input",
    "engineer_features",
    # Preprocessing
    "enforce_schema",
    "validate_input_ranges",
    # Schemas
    "ProcessedBatchPredictionInput",
    "BatchPredictionOutput",
    "EmployeeInputSchema",
    "PredictionOutput",
    "EvalInputSchema",
    "SirhInputSchema",
    "SondageInputSchema",
    "RawBatchPredictionInput",
    # Validation constants
    "ALL_FEATURE_COLS",
    "CATEGORICAL_COLS",
    "ENGINEERED_FEATURES",
    "EXPECTED_DTYPES",
    "NUMERIC_COLS",
    "RAW_CATEGORICAL_COLS",
    "RAW_INPUT_COLS",
    "RAW_NUMERIC_COLS",
    "TARGET_COL",
    "VALIDATION_RANGES",
]
