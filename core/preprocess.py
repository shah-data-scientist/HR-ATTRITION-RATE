"""Schema enforcement and preprocessing for model inference.

This module ensures all data sent to the model matches the exact
schema used during training.
"""

import pandas as pd

from core.validation import (
    CATEGORICAL_COLS,
    EXPECTED_DTYPES,
    NUMERIC_COLS,
    VALIDATION_RANGES,
)


def enforce_schema(
    df: pd.DataFrame, feature_order: list[str] | None = None
) -> pd.DataFrame:
    """Enforce canonical schema and column order for model input.

    This function:
    1. Ensures all required columns are present (adds missing with defaults)
    2. Removes unexpected columns
    3. Coerces dtypes to match training schema
    4. Reorders columns to match training order

    Args:
        df: DataFrame to validate and transform
        feature_order: Optional list of columns in exact order (default: NUMERIC + CATEGORICAL)

    Returns:
        DataFrame with enforced schema

    Raises:
        ValueError: If critical columns are missing and cannot be defaulted
    """
    df = df.copy()

    if feature_order is None:
        feature_order = NUMERIC_COLS + CATEGORICAL_COLS

    # Add missing columns with safe defaults
    for col in feature_order:
        if col not in df.columns:
            if col in CATEGORICAL_COLS:
                df[col] = "Unknown"
            elif col in NUMERIC_COLS:
                # Use 0 for numeric defaults; adjust if domain knowledge requires otherwise
                df[col] = 0
            else:
                raise ValueError(f"Column '{col}' is required but missing from input")

    # Select only the required columns in the correct order
    df = df[feature_order]

    # Coerce dtypes to match training schema
    for col in df.columns:
        expected_dtype = EXPECTED_DTYPES.get(col)
        if expected_dtype is None:
            continue

        try:
            if expected_dtype == "object":
                df[col] = df[col].astype(str)
            elif expected_dtype == "Int64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif expected_dtype == "int64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")
            elif expected_dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        except Exception as e:
            raise ValueError(
                f"Failed to coerce column '{col}' to type '{expected_dtype}': {e}"
            ) from e

    return df


def validate_input_ranges(df: pd.DataFrame) -> list[str]:
    """Validate that numeric inputs fall within expected ranges.

    Args:
        df: DataFrame with enforced schema

    Returns:
        List of validation warnings (empty if all valid)
    """
    warnings = []

    for col, (min_val, max_val) in VALIDATION_RANGES.items():
        if col in df.columns:
            # Skip validation if column is not numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            if (df[col] < min_val).any() or (df[col] > max_val).any():
                warnings.append(
                    f"{col} has values outside expected range [{min_val}, {max_val}]"
                )

    return warnings
