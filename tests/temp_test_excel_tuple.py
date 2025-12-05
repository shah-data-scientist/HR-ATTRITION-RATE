import pandas as pd
import io
import openpyxl  # Ensure openpyxl is installed


def test_excel_with_tuple_in_cell():
    data_with_tuple = {
        "col1": [1, 2, 3],
        "col2": ["A", ("B", "C"), "D"],  # Tuple in a string column
    }
    df_with_tuple = pd.DataFrame(data_with_tuple)

    data_all_strings = {"col1": [1, 2, 3], "col2": ["A", "B,C", "D"]}  # All strings
    df_all_strings = pd.DataFrame(data_all_strings)

    print("Attempting to write DataFrame with tuple in cell:")
    try:
        excel_buffer_tuple = io.BytesIO()
        with pd.ExcelWriter(excel_buffer_tuple, engine="openpyxl") as writer:
            df_with_tuple.to_excel(writer, sheet_name="Sheet1", index=False)
        print(
            "Successfully wrote DataFrame with tuple (converted to string). This might not be the issue."
        )
    except Exception as e:
        print(f"Error writing DataFrame with tuple: {e}")

    print("\nAttempting to write DataFrame with all strings:")
    try:
        excel_buffer_strings = io.BytesIO()
        with pd.ExcelWriter(excel_buffer_strings, engine="openpyxl") as writer:
            df_all_strings.to_excel(writer, sheet_name="Sheet1", index=False)
        print("Successfully wrote DataFrame with all strings.")
    except Exception as e:
        print(f"Error writing DataFrame with all strings: {e}")

    # Test with a dataframe that explicitly tries to put a tuple object where a string is expected by openpyxl
    data_mixed = {
        "Feature": ["feature_1", "feature_2"],
        "Coefficient": [0.5, 0.3],
        "Employee_ID": [1, 2],
        "Prediction": ["Leave", "Stay"],
    }
    df_mixed = pd.DataFrame(data_mixed)

    # What if a feature name *itself* becomes a tuple?
    # This scenario is unlikely with astype(str) but let's consider it.
    df_problematic_feature = pd.DataFrame(
        {
            "Feature": ["f1", ("f2", "tuple"), "f3"],  # Intentionally inserting a tuple
            "Coefficient": [0.1, 0.2, 0.3],
        }
    )
    df_problematic_feature["Feature"] = df_problematic_feature["Feature"].astype(str)
    print("\nAttempting to write DataFrame with a string-converted tuple feature:")
    try:
        excel_buffer_problem = io.BytesIO()
        with pd.ExcelWriter(excel_buffer_problem, engine="openpyxl") as writer:
            df_problematic_feature.to_excel(writer, sheet_name="Problem", index=False)
        print(
            "Successfully wrote DataFrame with string-converted tuple feature. It converts to string."
        )
    except Exception as e:
        print(f"Error writing DataFrame with string-converted tuple feature: {e}")


test_excel_with_tuple_in_cell()
