import types
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import app  # Moved import app to the top


class Bunch(dict):
    """A dictionary that allows attribute-style access.

    This is used to mock Streamlit's session_state, which allows access
    to its elements using dot notation (e.g., st.session_state.key).
    """

    __getattr__ = dict.get

    def __setattr__(self, k: str, v: Any) -> None:
        """Set an attribute, which also sets the dictionary item."""
        self[k] = v


class FakeStreamlit(types.SimpleNamespace):
    """A mock Streamlit object for testing purposes."""

    def __init__(self):
        super().__init__()
        self._calls: list[tuple[str, Any, Any]] = []
        self.session_state = Bunch()

    def __getattr__(self, name: str) -> Any:
        """Dynamically capture all Streamlit calls."""
        # If the attribute is already defined, return it
        if name in self.__dict__:
            return self.__dict__[name]

        # Otherwise, create a mock method for the Streamlit function
        def mock_method(*args: Any, **kwargs: Any) -> Any:
            self._calls.append((name, args, kwargs))
            # Provide default return values for common widgets
            if name == "slider":
                return kwargs.get("value", 0.5)
            if name == "file_uploader":
                return []
            if name == "button":
                return False
            if name == "columns":
                if args and isinstance(args[0], int):
                    return [MagicMock() for _ in range(args[0])]
                return [MagicMock()]
            if name == "components":
                return self # for components.v1
            if name == "v1":
                return self # for components.v1.html
            return MagicMock() # Default return for other methods

        return mock_method

    def called(self, name: str) -> bool:
        """Check if a Streamlit method was called."""
        return any(c[0] == name for c in self._calls)

    def get_call_args(self, name: str) -> list[tuple[Any, Any]]:
        """Get arguments of calls to a Streamlit method."""
        return [c[1:] for c in self._calls if c[0] == name]


@pytest.fixture(autouse=True)
def mock_streamlit_and_app_setup(
    monkeypatch: pytest.MonkeyPatch, mock_model_and_data: tuple[Any, ...]
) -> tuple[Any, Any, Any, Any]:
    """Mock Streamlit functions and app global variables for testing."""
    fake_st = FakeStreamlit()

    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "components", fake_st.components())

    mock_pipeline, mock_explainer, mock_df, mock_series = mock_model_and_data

    monkeypatch.setattr(app, "model", mock_pipeline, raising=False)
    monkeypatch.setattr(app, "X_train_loaded", mock_df, raising=False)
    monkeypatch.setattr(app, "y_train_loaded", mock_series, raising=False)
    monkeypatch.setattr(app, "X_test_loaded", mock_df, raising=False)
    monkeypatch.setattr(app, "y_test_loaded", mock_series, raising=False)
    monkeypatch.setattr(
        app,
        "train_prediction_proba",
        mock_pipeline.predict_proba(mock_df)[:, 1],
        raising=False,
    )

    # Explicitly call _setup_app to initialize global variables
    app._setup_app()

    fake_st.session_state.clear()
    fake_st.session_state.prediction_triggered = False
    fake_st.session_state.report_data = None
    fake_st.session_state.shap_html_content = None
    fake_st.session_state.excel_report_data = None
    fake_st.session_state.processed_data_for_shap = None
    fake_st.session_state.explainer = None
    fake_st.session_state.all_features = None

    return app.main, app._setup_app, app.clear_prediction_results, fake_st


@pytest.fixture
def mock_model_and_data() -> tuple[MagicMock, MagicMock, pd.DataFrame, pd.Series]:
    """Mock the ML pipeline, explainer, and data for testing."""
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array(
        [[0.2, 0.8]]
    )  # Example probability

    mock_preprocessor = MagicMock()
    mock_preprocessor.feature_names_in_ = ["feature1", "feature2", "feature3"]
    mock_preprocessor.transform.return_value = pd.DataFrame(
        np.array([[1, 2, 3]]), columns=["feat_A", "feat_B", "feat_C"]
    ) # Example transformed data as DataFrame

    mock_ohe = MagicMock()
    mock_ohe.get_feature_names_out.return_value = [
        "cat_feature_A",
        "cat_feature_B",
    ]  # Example categorical features

    mock_preprocessor.named_transformers_ = {
        "cat": mock_ohe
    }  # Add named_transformers_ to preprocessor

    # Ensure get_feature_names_out returns something with length 3
    mock_preprocessor.get_feature_names_out.return_value = ["feat_A", "feat_B", "feat_C"]

    mock_pipeline.named_steps = {
        "preprocessor": mock_preprocessor,
        "model": MagicMock(),
    }

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.array(
        [[0.1, 0.2, 0.3]]
    )  # Example SHAP values (2D array for one instance)
    mock_explainer.expected_value = 0.5  # Example expected value

    mock_df = pd.DataFrame({"feature1": [1], "feature2": [2], "feature3": [3]})
    mock_series = pd.Series([0])

    with (
        patch(
            "app.load_model_and_data",
            return_value=(mock_pipeline, mock_df, mock_series, mock_df, mock_series),
        ),
        patch("shap.TreeExplainer", return_value=mock_explainer),
        patch("shap.plots.waterfall", return_value=None),
        patch("matplotlib.pyplot.gcf", return_value=MagicMock()),
        patch("matplotlib.pyplot.close", return_value=None),
    ):
        yield mock_pipeline, mock_explainer, mock_df, mock_series


@pytest.fixture
def mock_uploaded_files() -> list[MagicMock]:
    """Mock uploaded file objects for Streamlit's file_uploader."""
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


def test_main_initial_load(
    mock_streamlit_and_app_setup: tuple[Any, Any, Any, Any]
) -> None:
    """Test the initial loading of the Streamlit app."""
    main, _, _, fake_st = mock_streamlit_and_app_setup
    main()
    assert fake_st.called("set_page_config")
    assert fake_st.called("title")




def test_main_file_uploader_partial_files(
    mock_streamlit_and_app_setup: tuple[Any, Any, Any, Any]
) -> None:
    """Test the scenario where only partial files are uploaded."""
    main, _, _, fake_st = mock_streamlit_and_app_setup
    fake_st.file_uploader = MagicMock(
        return_value=[MagicMock(name="extrait_eval.csv")]
    )  # Only one file
    main()
    assert fake_st.called("warning")
    assert (
        fake_st.get_call_args("warning")[0][0][0]
        == "Please upload all three required CSV files."
    )


def test_main_file_uploader_wrong_names(
    mock_streamlit_and_app_setup: tuple[Any, Any, Any, Any]
) -> None:
    """Test the scenario where files with incorrect names are uploaded."""
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
    assert (
        fake_st.get_call_args("warning")[0][0][0]
        == "Please make sure to upload the three required files: "
        "`extrait_eval.csv`, `extrait_sirh.csv`, and `extrait_sondage.csv`."
    )


def test_clear_prediction_results(
    mock_streamlit_and_app_setup: tuple[Any, Any, Any, Any]
) -> None:
    """Test the clearing of prediction results in session state."""
    main, _, clear_prediction_results, fake_st = mock_streamlit_and_app_setup
    fake_st.session_state.prediction_triggered = True
    fake_st.session_state.report_data = "some data"
    clear_prediction_results()
    assert not fake_st.session_state.prediction_triggered
    assert fake_st.session_state.report_data is None


def test_main_no_uploaded_data_no_predict_button_click(
    mock_streamlit_and_app_setup: tuple[Any, Any, Any, Any]
) -> None:
    """Test app behavior when no data is uploaded and predict button is not clicked."""
    main, _, _, fake_st = mock_streamlit_and_app_setup
    fake_st.file_uploader = MagicMock(return_value=[])  # No files uploaded
    fake_st.button = MagicMock(return_value=False)  # Button not clicked
    main()
    assert not fake_st.session_state.prediction_triggered
    assert not fake_st.called("download_button")
    assert not fake_st.called("html")
    assert not fake_st.called("success")


def test_main_no_uploaded_data_predict_button_click(
    mock_streamlit_and_app_setup: tuple[Any, Any, Any, Any]
) -> None:
    """Test app behavior when no data is uploaded but predict button is clicked."""
    main, _, _, fake_st = mock_streamlit_and_app_setup
    fake_st.file_uploader = MagicMock(return_value=[])  # No files uploaded
    fake_st.button = MagicMock(return_value=True)  # Button clicked, but no data
    main()
    assert not fake_st.session_state.prediction_triggered
    assert not fake_st.called("download_button")
    assert not fake_st.called("html")
    assert not fake_st.called("success")
