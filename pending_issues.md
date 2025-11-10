# Pending Issue: TypeError in ML Model Prediction

**Problem:**
The FastAPI `/predict` endpoint is returning a `500 Internal Server Error` with the detail message:
`"Prediction failed: '<' not supported between instances of 'float' and 'str'"`

**Context:**
*   This error occurs during the `model.predict_proba()` call within the `/predict` endpoint in `api/app/main.py`.
*   A `coerce_and_align` function was implemented and integrated into the API to explicitly cast numeric columns to `float` and categorical columns to `string` (or `None` for NaNs) before passing the DataFrame to the model.
*   Debugging with `debug_types.py` confirmed that *after* `coerce_and_align` runs, all columns in the DataFrame (`final_data_for_prediction`) have their expected numeric (`int64`, `float64`) or string (`string[python]`) dtypes. This indicates the type mismatch is not occurring *before* the model's preprocessor.
*   The error message strongly suggests that a column which the model's internal `ColumnTransformer` expects to be numeric (e.g., for `StandardScaler`) is receiving string values, or vice-versa, leading to a comparison operation between `float` and `str`.

**Hypothesis for the Root Cause:**
The most probable cause is a mismatch between the `NUMERIC` and `CATEGORICAL` column definitions hardcoded in `api/app/main.py` (used by `coerce_and_align`) and the actual `num_cols` and `cat_cols` that were used to train and configure the `ColumnTransformer` within the loaded `employee_attrition_pipeline.pkl` model.

**Next Debugging Step (when revisiting this issue):**
To definitively identify the mismatch, the following information is required:
1.  **Output from the FastAPI server logs** (the terminal where `uvicorn` is running) after the application starts up.
2.  Specifically, look for the debug print statements added in the `lifespan` function of `api/app/main.py`, which show the internal column lists of the `ColumnTransformer`:
    ```
    Preprocessor Numeric Columns (from model): [...]
    Preprocessor Categorical Columns (from model): [...]
    ```
    This output will reveal the exact numeric and categorical columns the *trained model* expects. This can then be compared against the `NUMERIC` and `CATEGORICAL` lists defined in `api/app/main.py` to find the discrepancy.

**Current Status:**
Issue documented. Proceeding with other tasks as requested.
