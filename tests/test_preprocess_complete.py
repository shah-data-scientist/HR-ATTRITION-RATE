"""Comprehensive tests for core/preprocess.py to achieve 100% coverage."""

import pandas as pd
import pytest


class TestEnforceSchema:
    """Test enforce_schema function comprehensively."""

    def test_enforce_schema_missing_numeric_columns(self):
        """Test that missing numeric columns are added with default value 0."""
        from core.preprocess import enforce_schema

        df = pd.DataFrame({
            "age": [30, 35],
            "genre": [1, 0]
        })

        result = enforce_schema(df)

        # Should have all required columns
        assert "revenu_mensuel" in result.columns
        assert result["revenu_mensuel"].iloc[0] == 0

    def test_enforce_schema_missing_categorical_columns(self):
        """Test that missing categorical columns are added with 'Unknown'."""
        from core.preprocess import enforce_schema

        df = pd.DataFrame({
            "age": [30, 35],
            "revenu_mensuel": [5000, 6000]
        })

        result = enforce_schema(df)

        # Should have categorical columns with default
        assert "statut_marital" in result.columns
        assert result["statut_marital"].iloc[0] == "Unknown"

    def test_enforce_schema_with_custom_feature_order(self):
        """Test enforce_schema with custom feature order."""
        from core.preprocess import enforce_schema

        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": ["a", "b"],
            "col3": [3.0, 4.0]
        })

        feature_order = ["col1", "col2", "col3"]
        result = enforce_schema(df, feature_order=feature_order)

        # Check column order
        assert list(result.columns) == feature_order

    def test_enforce_schema_dtype_coercion_object(self):
        """Test dtype coercion to object (string)."""
        from core.preprocess import enforce_schema
        from core.validation import EXPECTED_DTYPES

        # Patch EXPECTED_DTYPES temporarily
        original_dtypes = EXPECTED_DTYPES.copy()
        EXPECTED_DTYPES["test_col"] = "object"

        try:
            df = pd.DataFrame({
                "test_col": [1, 2, 3]
            })

            result = enforce_schema(df, feature_order=["test_col"])

            assert result["test_col"].dtype == object
            assert result["test_col"].iloc[0] == "1"
        finally:
            # Restore original
            EXPECTED_DTYPES.clear()
            EXPECTED_DTYPES.update(original_dtypes)

    def test_enforce_schema_dtype_coercion_int64(self):
        """Test dtype coercion to int64."""
        from core.preprocess import enforce_schema
        from core.validation import EXPECTED_DTYPES

        original_dtypes = EXPECTED_DTYPES.copy()
        EXPECTED_DTYPES["test_col"] = "int64"

        try:
            df = pd.DataFrame({
                "test_col": ["10", "20", "30"]
            })

            result = enforce_schema(df, feature_order=["test_col"])

            assert result["test_col"].dtype == "int64"
        finally:
            EXPECTED_DTYPES.clear()
            EXPECTED_DTYPES.update(original_dtypes)

    def test_enforce_schema_dtype_coercion_Int64(self):
        """Test dtype coercion to Int64 (nullable)."""
        from core.preprocess import enforce_schema
        from core.validation import EXPECTED_DTYPES

        original_dtypes = EXPECTED_DTYPES.copy()
        EXPECTED_DTYPES["test_col"] = "Int64"

        try:
            df = pd.DataFrame({
                "test_col": ["10", "20", None]
            })

            result = enforce_schema(df, feature_order=["test_col"])

            assert result["test_col"].dtype == pd.Int64Dtype()
        finally:
            EXPECTED_DTYPES.clear()
            EXPECTED_DTYPES.update(original_dtypes)

    def test_enforce_schema_dtype_coercion_float64(self):
        """Test dtype coercion to float64."""
        from core.preprocess import enforce_schema
        from core.validation import EXPECTED_DTYPES

        original_dtypes = EXPECTED_DTYPES.copy()
        EXPECTED_DTYPES["test_col"] = "float64"

        try:
            df = pd.DataFrame({
                "test_col": ["10.5", "20.3", "30.7"]
            })

            result = enforce_schema(df, feature_order=["test_col"])

            assert result["test_col"].dtype == "float64"
        finally:
            EXPECTED_DTYPES.clear()
            EXPECTED_DTYPES.update(original_dtypes)

    def test_enforce_schema_none_dtype_skip(self):
        """Test that columns with None dtype are skipped."""
        from core.preprocess import enforce_schema
        from core.validation import EXPECTED_DTYPES

        original_dtypes = EXPECTED_DTYPES.copy()
        EXPECTED_DTYPES["test_col"] = None

        try:
            df = pd.DataFrame({
                "test_col": [1, 2, 3]
            })

            # Should not raise error, dtype with None value is skipped
            result = enforce_schema(df, feature_order=["test_col"])
            assert "test_col" in result.columns
        finally:
            EXPECTED_DTYPES.clear()
            EXPECTED_DTYPES.update(original_dtypes)

    def test_enforce_schema_removes_extra_columns(self):
        """Test that extra columns not in feature_order are removed."""
        from core.preprocess import enforce_schema

        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": [3, 4],
            "extra_col": [5, 6]
        })

        result = enforce_schema(df, feature_order=["col1", "col2"])

        assert "extra_col" not in result.columns
        assert list(result.columns) == ["col1", "col2"]


class TestValidateInputRanges:
    """Test validate_input_ranges function."""

    def test_validate_input_ranges_all_valid(self):
        """Test validation with all values in range."""
        from core.preprocess import validate_input_ranges
        from core.validation import VALIDATION_RANGES

        # Use a column that exists in VALIDATION_RANGES
        if "age" in VALIDATION_RANGES:
            min_age, max_age = VALIDATION_RANGES["age"]
            df = pd.DataFrame({
                "age": [min_age, (min_age + max_age) // 2, max_age]
            })

            warnings = validate_input_ranges(df)

            assert len(warnings) == 0

    def test_validate_input_ranges_below_minimum(self):
        """Test validation with values below minimum."""
        from core.preprocess import validate_input_ranges
        from core.validation import VALIDATION_RANGES

        if "age" in VALIDATION_RANGES:
            min_age, max_age = VALIDATION_RANGES["age"]
            df = pd.DataFrame({
                "age": [min_age - 10, min_age, max_age]
            })

            warnings = validate_input_ranges(df)

            assert len(warnings) > 0
            assert "age" in warnings[0]

    def test_validate_input_ranges_above_maximum(self):
        """Test validation with values above maximum."""
        from core.preprocess import validate_input_ranges
        from core.validation import VALIDATION_RANGES

        if "age" in VALIDATION_RANGES:
            min_age, max_age = VALIDATION_RANGES["age"]
            df = pd.DataFrame({
                "age": [min_age, max_age, max_age + 10]
            })

            warnings = validate_input_ranges(df)

            assert len(warnings) > 0
            assert "age" in warnings[0]

    def test_validate_input_ranges_non_numeric_column(self):
        """Test that non-numeric columns are skipped in validation."""
        from core.preprocess import validate_input_ranges
        from core.validation import VALIDATION_RANGES

        # Add a validation range for a column we'll make non-numeric
        original_ranges = VALIDATION_RANGES.copy()
        VALIDATION_RANGES["test_col"] = (0, 100)

        try:
            df = pd.DataFrame({
                "test_col": ["a", "b", "c"]  # Non-numeric
            })

            warnings = validate_input_ranges(df)

            # Should not generate warnings for non-numeric columns
            assert all("test_col" not in w for w in warnings)
        finally:
            VALIDATION_RANGES.clear()
            VALIDATION_RANGES.update(original_ranges)

    def test_validate_input_ranges_missing_column(self):
        """Test validation when column in VALIDATION_RANGES is not in dataframe."""
        from core.preprocess import validate_input_ranges

        df = pd.DataFrame({
            "some_other_column": [1, 2, 3]
        })

        # Should not raise error, just skip validation
        warnings = validate_input_ranges(df)

        # No warnings since none of the VALIDATION_RANGES columns are present
        # (or they are but values are valid)
        assert isinstance(warnings, list)
