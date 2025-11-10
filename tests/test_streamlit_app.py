import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import io
import sys, importlib
import types

class Bunch(dict):
    """dict + attribute access for st.session_state."""
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v

class FakeStreamlit(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self._calls = []
        self.session_state = Bunch()

    # spy-able method
    def set_page_config(self, **kwargs):
        self._calls.append(("set_page_config", kwargs))

    # minimal UI stubs your app might call
    def title(self, *a, **k): self._calls.append(("title", a, k))
    def subheader(self, *a, **k): self._calls.append(("subheader", a, k))
    def write(self, *a, **k): self._calls.append(("write", a, k))
    def markdown(self, *a, **k): self._calls.append(("markdown", a, k))
    def slider(self, *a, **k): 
        self._calls.append(("slider", a, k))
        return k.get("value", 0.5) # Default threshold
    def file_uploader(self, *a, **k): 
        self._calls.append(("file_uploader", a, k))
        return []
    def button(self, *a, **k): 
        self._calls.append(("button", a, k))
        return False # Default no button click
    def warning(self, *a, **k): self._calls.append(("warning", a, k))
    def success(self, *a, **k): self._calls.append(("success", a, k))
    def columns(self, *args, **kwargs): 
        self._calls.append(("columns", args, kwargs))
        if args and isinstance(args[0], int):
            return [MagicMock() for _ in range(args[0])]
        return [MagicMock()] # Default to one column if no number is specified
    def pyplot(self, *a, **k): self._calls.append(("pyplot", a, k))
    def download_button(self, *a, **k): self._calls.append(("download_button", a, k))
    def components(self): return self # for components.v1
    def v1(self): return self # for components.v1.html
    def html(self, *a, **k): self._calls.append(("html", a, k))


    # helper to assert calls in tests
    def called(self, name):
        return any(c[0] == name for c in self._calls)

    def get_call_args(self, name):
        return [c[1:] for c in self._calls if c[0] == name]


@pytest.fixture(autouse=True)
def mock_streamlit_and_app_setup(monkeypatch, mock_model_and_data):
    fake_st = FakeStreamlit()

    # import app
    import app

    # Directly patch the 'st' and 'st.components.v1' within the app module
    monkeypatch.setattr(app, 'st', fake_st)
    monkeypatch.setattr(app, 'components', fake_st.components())

    mock_pipeline, mock_explainer, mock_df, mock_series = mock_model_and_data

    # Directly assign the mocked objects to the global variables in the app module
    monkeypatch.setattr(app, "model", mock_pipeline, raising=False)
    monkeypatch.setattr(app, "X_train_loaded", mock_df, raising=False) # Using mock_df for X_train_loaded
    monkeypatch.setattr(app, "y_train_loaded", mock_series, raising=False) # Using mock_series for y_train_loaded
    monkeypatch.setattr(app, "X_test_loaded", mock_df, raising=False) # Using mock_df for X_test_loaded
    monkeypatch.setattr(app, "y_test_loaded", mock_series, raising=False) # Using mock_series for y_test_loaded
    monkeypatch.setattr(app, "train_prediction_proba", mock_pipeline.predict_proba(mock_df)[:, 1], raising=False)
    
    # Reset session state before each test
    fake_st.session_state.clear()
    fake_st.session_state.prediction_triggered = False
    fake_st.session_state.report_data = None
    fake_st.session_state.shap_html_content = None
    fake_st.session_state.excel_report_data = None
    fake_st.session_state.processed_data_for_shap = None
    fake_st.session_state.explainer = None
    fake_st.session_state.all_features = None

    yield app.main, app._setup_app, app.clear_prediction_results, fake_st


@pytest.fixture
def mock_model_and_data():
    # Mock the entire pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array(
        [[0.2, 0.8]]
    )  # Example probability

    # Mock preprocessor for get_expected_columns and SHAP
    mock_preprocessor = MagicMock()
    mock_preprocessor.feature_names_in_ = ["feature1", "feature2", "feature3"]
    mock_preprocessor.transform.return_value = np.array(
        [[1, 2, 3]]
    )  # Example transformed data

    # Mock the OneHotEncoder within the preprocessor
    mock_ohe = MagicMock()
    mock_ohe.get_feature_names_out.return_value = ["cat_feature_A", "cat_feature_B"] # Example categorical features

    mock_preprocessor.named_transformers_ = {"cat": mock_ohe} # Add named_transformers_ to preprocessor

    mock_pipeline.named_steps = {
        "preprocessor": mock_preprocessor,
        "model": MagicMock(),
    }

    # Mock explainer for SHAP
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.array(
        [[0.1, 0.2, 0.3]]
    )  # Example SHAP values (2D array for one instance)
    mock_explainer.expected_value = 0.5  # Example expected value

    # Mock dataframes
    mock_df = pd.DataFrame({"feature1": [1], "feature2": [2], "feature3": [3]})
    mock_series = pd.Series([0])

    with (
        patch(
            "app.load_model_and_data",
            return_value=(mock_pipeline, mock_df, mock_series, mock_df, mock_series),
        ),
        patch("shap.LinearExplainer", return_value=mock_explainer),
        patch("shap.plots.waterfall", return_value=None),
        patch("matplotlib.pyplot.gcf", return_value=MagicMock()),
        patch("matplotlib.pyplot.close", return_value=None),
    ):
        yield mock_pipeline, mock_explainer, mock_df, mock_series


@pytest.fixture
def mock_uploaded_files():
    # Create mock file objects
    mock_eval_file = MagicMock()
    mock_eval_file.name = "extrait_eval.csv"
    mock_eval_file.read.return_value = b"id_employee,col_eval\n1,val1"

    mock_sirh_file = MagicMock()
    mock_sirh_file.name = "extrait_sirh.csv"
    mock_sirh_file.read.return_value = b"id_employee,col_sirh\n1,valA"

    mock_sondage_file = MagicMock()
    mock_sondage_file.name = "extrait_sondage.csv"
    mock_sondage_file.read.return_value = b"id_employee,col_sondage\n1,valX"

    return [mock_eval_file, mock_sirh_file, mock_sondage_file]


# --- Tests for main() function ---


    def test_main_initial_load(mock_streamlit_and_app_setup, mock_model_and_data):
        main, _, _, fake_st = mock_streamlit_and_app_setup
        main()
        # Verify initial Streamlit calls
        assert fake_st.called("set_page_config")
        assert fake_st.called("title")
        assert fake_st.get_call_args("title")[0][0][0] == "Employee Attrition Risk"
        assert fake_st.called("subheader")
        assert fake_st.get_call_args("subheader")[0][0][0] == "Understanding Threshold Impact (examples from training data)"
        assert fake_st.called("slider")
        assert fake_st.called("file_uploader")
        # Ensure prediction is not triggered initially
        assert not fake_st.session_state.prediction_triggered

def test_main_predict_button_click(
    mock_streamlit_and_app_setup, mock_model_and_data, mock_uploaded_files
):
    main, _, _, fake_st = mock_streamlit_and_app_setup
    mock_pipeline, mock_explainer, _, _ = mock_model_and_data

    # Simulate file upload
    fake_st.file_uploader = MagicMock(return_value=mock_uploaded_files)

    # Simulate button click
    fake_st.button = MagicMock(return_value=True)

    # Mock pandas read_csv to return actual DataFrames
    with (
        patch(
            "pandas.read_csv",
            side_effect=[
                pd.DataFrame({"id_employee": [1], "col_eval": ["val1"]}),
                pd.DataFrame({"id_employee": [1], "col_sirh": ["valA"]}),
                pd.DataFrame({"id_employee": [1], "col_sondage": ["valX"]}),
            ],
        ),
        patch(
            "app.clean_and_engineer_features",
            return_value=pd.DataFrame(
                {"feature1": [1], "feature2": [2], "feature3": [3], "id_employee": [1]}
            ),
        ),
    ):
        main()

    # Verify prediction was triggered
    assert fake_st.session_state.prediction_triggered
    assert fake_st.session_state.report_data is not None
    assert fake_st.session_state.processed_data_for_shap is not None
    assert fake_st.session_state.explainer is not None
    assert fake_st.session_state.all_features is not None
    assert fake_st.session_state.excel_report_data is not None

    # Verify report generation calls
    assert fake_st.called("download_button")
    assert fake_st.called("html")
    assert fake_st.called("success")
    assert fake_st.get_call_args("success")[0][0][0] == "Reports generated successfully!"


def test_main_file_uploader_partial_files(mock_streamlit_and_app_setup):
    main, _, _, fake_st = mock_streamlit_and_app_setup
    fake_st.file_uploader = MagicMock(
        return_value=[MagicMock(name="extrait_eval.csv")]
    )  # Only one file
    main()
    assert fake_st.called("warning")
    assert fake_st.get_call_args("warning")[0][0][0] == "Please upload all three required CSV files."


def test_main_file_uploader_wrong_names(mock_streamlit_and_app_setup):
    main, _, _, fake_st = mock_streamlit_and_app_setup
    fake_st.file_uploader = MagicMock(
        return_value=[
            MagicMock(name="wrong_eval.csv"),
            MagicMock(name="wrong_sirh.csv"),
            MagicMock(name="wrong_sondage.csv"),
        ]
    )
    main()
    assert fake_st.called("warning")
    assert fake_st.get_call_args("warning")[0][0][0] == "Please make sure to upload the three required files: `extrait_eval.csv`, `extrait_sirh.csv`, and `extrait_sondage.csv`."


def test_clear_prediction_results(mock_streamlit_and_app_setup):
    main, _, clear_prediction_results, fake_st = mock_streamlit_and_app_setup
    # Set some session state values
    fake_st.session_state.prediction_triggered = True
    fake_st.session_state.report_data = "some data"
    clear_prediction_results()
    assert not fake_st.session_state.prediction_triggered
    assert fake_st.session_state.report_data is None


def test_main_no_uploaded_data_no_predict_button_click(mock_streamlit_and_app_setup):
    main, _, _, fake_st = mock_streamlit_and_app_setup
    fake_st.file_uploader = MagicMock(return_value=[])  # No files uploaded
    fake_st.button = MagicMock(return_value=False)  # Button not clicked
    main()
    # Ensure no prediction logic is triggered
    assert not fake_st.session_state.prediction_triggered
    assert not fake_st.called("download_button")
    assert not fake_st.called("html")
    assert not fake_st.called("success")


def test_main_no_uploaded_data_predict_button_click(mock_streamlit_and_app_setup):
    main, _, _, fake_st = mock_streamlit_and_app_setup
    fake_st.file_uploader = MagicMock(return_value=[])  # No files uploaded
    fake_st.button = MagicMock(return_value=True)  # Button clicked, but no data
    main()
    # Ensure no prediction logic is triggered due to missing data
    assert not fake_st.session_state.prediction_triggered
    assert not fake_st.called("download_button")
    assert not fake_st.called("html")
    assert not fake_st.called("success")
