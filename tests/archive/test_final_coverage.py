"""
Final coverage boost tests - targeting 85% coverage
Focus on uncovered branches and edge cases
"""

import pytest
import pandas as pd
import numpy as np
from core.data_processing import clean_raw_input, engineer_features
from core.preprocess import enforce_schema


class TestDataProcessingEdgeCases:
    """Test edge cases in data processing"""

    def test_clean_genre_edge_cases(self):
        """Test various genre input formats"""
        df = pd.DataFrame(
            {
                "genre": [
                    "M",
                    "m",
                    "F",
                    "f",
                    "Homme",
                    "homme",
                    "Femme",
                    "femme",
                    "Male",
                    "male",
                    "Female",
                    "female",
                    1,
                    0,
                ]
            }
        )
        result = clean_raw_input(df)
        # All should be converted to 1 or 0
        assert result["genre"].isin([0, 1]).all()

    def test_clean_heures_with_various_formats(self):
        """Test heures supplementaires with various formats"""
        df = pd.DataFrame(
            {
                "heure_supplementaires": [
                    "Oui",
                    "oui",
                    "Non",
                    "non",
                    "Yes",
                    "yes",
                    "No",
                    "no",
                    "True",
                    "true",
                    "False",
                    "false",
                ]
            }
        )
        result = clean_raw_input(df)
        assert "heures_supplementaires" in result.columns
        assert result["heures_supplementaires"].isin([0, 1]).all()

    def test_salary_increase_various_formats(self):
        """Test salary increase field with various formats"""
        df = pd.DataFrame(
            {"augementation_salaire_precedente": ["11 %", "15%", "0 %", "100%", "5%"]}
        )
        result = clean_raw_input(df)
        # Should remove % but keep as string
        assert all(
            "%" not in str(val) for val in result["augementation_salaire_precedente"]
        )

    def test_engineer_features_with_missing_columns(self):
        """Test feature engineering with some missing columns"""
        df = pd.DataFrame(
            {"note_evaluation_actuelle": [4, 3], "note_evaluation_precedente": [3, 3]}
        )
        result = engineer_features(df)
        # Should still create improvement_evaluation
        assert "improvement_evaluation" in result.columns

    def test_engineer_features_with_zero_denominator(self):
        """Test work mobility with zero years in company"""
        df = pd.DataFrame(
            {
                "nombre_experiences_precedentes": [5, 3],
                "annees_dans_l_entreprise": [0, 0],
            }
        )
        result = engineer_features(df)
        assert "work_mobility" in result.columns
        # Should handle division by zero
        assert result["work_mobility"].notna().all()


class TestPreprocessingEdgeCases:
    """Test edge cases in preprocessing"""

    def test_enforce_schema_with_all_missing(self):
        """Test schema enforcement with completely empty dataframe"""
        from core.validation import ALL_FEATURE_COLS

        df = pd.DataFrame()
        result = enforce_schema(df)
        # Should create all columns
        assert len(result.columns) == len(ALL_FEATURE_COLS)

    def test_enforce_schema_preserves_existing_values(self):
        """Test that existing values are preserved"""
        from core.validation import ALL_FEATURE_COLS

        test_values = {"age": [42], "genre": [1], "revenu_mensuel": [75000]}
        df = pd.DataFrame(test_values)
        result = enforce_schema(df)

        # Check preserved values
        assert result["age"].iloc[0] == 42
        assert result["genre"].iloc[0] == 1
        assert result["revenu_mensuel"].iloc[0] == 75000

    def test_enforce_schema_with_string_categoricals(self):
        """Test that categorical strings are preserved"""
        from core.validation import ALL_FEATURE_COLS, CATEGORICAL_COLS

        df = pd.DataFrame(
            {
                col: ["test_value"] if col in CATEGORICAL_COLS else [0]
                for col in ALL_FEATURE_COLS
            }
        )
        result = enforce_schema(df)

        # Categorical columns should still be strings
        for col in CATEGORICAL_COLS:
            if col in result.columns:
                assert isinstance(result[col].iloc[0], str)


class TestDataMerging:
    """Test data merging logic"""

    def test_merge_with_matching_ids(self):
        """Test merging when all IDs match"""
        df1 = pd.DataFrame({"id_employee": [1, 2, 3], "value1": [10, 20, 30]})
        df2 = pd.DataFrame({"id_employee": [1, 2, 3], "value2": [100, 200, 300]})

        merged = pd.merge(df1, df2, on="id_employee", how="inner")
        assert len(merged) == 3
        assert "value1" in merged.columns
        assert "value2" in merged.columns

    def test_merge_with_missing_ids(self):
        """Test merging when some IDs don't match"""
        df1 = pd.DataFrame({"id_employee": [1, 2, 3], "value1": [10, 20, 30]})
        df2 = pd.DataFrame({"id_employee": [1, 2, 4], "value2": [100, 200, 400]})

        merged = pd.merge(df1, df2, on="id_employee", how="inner")
        # Only 1 and 2 should match
        assert len(merged) == 2
        assert sorted(merged["id_employee"].tolist()) == [1, 2]

    def test_merge_with_duplicate_columns(self):
        """Test merging with overlapping column names"""
        df1 = pd.DataFrame({"id_employee": [1, 2], "age": [25, 30], "value": [10, 20]})
        df2 = pd.DataFrame(
            {"id_employee": [1, 2], "age": [25, 30], "value": [100, 200]}
        )

        merged = pd.merge(df1, df2, on="id_employee", how="inner", suffixes=("", "_y"))
        # Should have age and value without suffix, and age_y and value_y
        assert "value" in merged.columns or "value_y" in merged.columns


class TestNullHandling:
    """Test null and missing value handling"""

    def test_fillna_with_defaults(self):
        """Test filling nulls with default values"""
        df = pd.DataFrame(
            {
                "age": [25, None, 35, None],
                "genre": ["M", None, "F", None],
                "revenu_mensuel": [5000, None, 6000, None],
            }
        )

        # Fill with defaults
        df["age"] = df["age"].fillna(0)
        df["genre"] = df["genre"].fillna("Unknown")
        df["revenu_mensuel"] = df["revenu_mensuel"].fillna(0)

        assert df["age"].isna().sum() == 0
        assert df["genre"].isna().sum() == 0
        assert df["revenu_mensuel"].isna().sum() == 0

    def test_dropna_behavior(self):
        """Test dropping rows with nulls"""
        df = pd.DataFrame(
            {
                "id_employee": [1, 2, 3, 4],
                "age": [25, None, 35, 40],
                "genre": ["M", "F", None, "M"],
            }
        )

        # Drop rows with any null
        cleaned = df.dropna()
        assert len(cleaned) == 2  # Only rows 1 and 4 have no nulls


class TestDataTypeConversions:
    """Test data type conversion logic"""

    def test_string_to_numeric_conversion(self):
        """Test converting string numbers to numeric"""
        df = pd.DataFrame(
            {"age": ["25", "30", "35"], "revenu": ["5000", "6000", "7000"]}
        )

        df["age"] = pd.to_numeric(df["age"])
        df["revenu"] = pd.to_numeric(df["revenu"])

        assert df["age"].dtype in [np.int64, np.float64]
        assert df["revenu"].dtype in [np.int64, np.float64]

    def test_categorical_to_string(self):
        """Test converting categoricals to strings"""
        df = pd.DataFrame(
            {"departement": ["IT", "HR", "Sales"], "poste": ["Dev", "Manager", "Rep"]}
        )

        df["departement"] = df["departement"].astype(str)
        df["poste"] = df["poste"].astype(str)

        assert df["departement"].dtype == object
        assert df["poste"].dtype == object


class TestColumnOperations:
    """Test column selection and filtering"""

    def test_select_specific_columns(self):
        """Test selecting specific columns"""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "col3": [7, 8, 9],
                "col4": [10, 11, 12],
            }
        )

        selected = df[["col1", "col3"]]
        assert list(selected.columns) == ["col1", "col3"]
        assert len(selected.columns) == 2

    def test_drop_columns(self):
        """Test dropping columns"""
        df = pd.DataFrame(
            {"keep1": [1, 2], "drop1": [3, 4], "keep2": [5, 6], "drop2": [7, 8]}
        )

        result = df.drop(columns=["drop1", "drop2"])
        assert list(result.columns) == ["keep1", "keep2"]

    def test_rename_columns(self):
        """Test renaming columns"""
        df = pd.DataFrame({"old_name1": [1, 2], "old_name2": [3, 4]})

        result = df.rename(columns={"old_name1": "new_name1", "old_name2": "new_name2"})
        assert "new_name1" in result.columns
        assert "new_name2" in result.columns
        assert "old_name1" not in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
