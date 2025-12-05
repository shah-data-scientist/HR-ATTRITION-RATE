import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import json
import os
import sys

# Add project root to path to allow importing ui.app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ui import app


class TestUIFunctions(unittest.TestCase):

    @patch("ui.app.httpx.get")
    def test_check_api_health_success(self, mock_get):
        mock_get.return_value.status_code = 200
        ok, msg = app._check_api_health("http://test")
        self.assertTrue(ok)
        self.assertEqual(msg, "API is healthy")

    @patch("ui.app.httpx.get")
    def test_check_api_health_failure(self, mock_get):
        mock_get.return_value.status_code = 500
        ok, msg = app._check_api_health("http://test")
        self.assertFalse(ok)
        self.assertIn("500", msg)

    @patch("ui.app.httpx.post")
    def test_call_prediction_api_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"predictions": []}

        result = app._call_prediction_api([], [], [])
        self.assertEqual(result, {"predictions": []})

    def test_apply_threshold(self):
        predictions = [
            {"id_employee": 1, "probability": 0.1},
            {"id_employee": 2, "probability": 0.9},
        ]
        df = app.apply_threshold(predictions, 0.5)

        self.assertEqual(len(df), 2)
        self.assertEqual(df.loc[0, "prediction"], 0)
        self.assertEqual(df.loc[0, "prediction_label"], "Stay")
        self.assertEqual(df.loc[1, "prediction"], 1)
        self.assertEqual(df.loc[1, "prediction_label"], "Leave")

    @patch("ui.app.load_test_set_confusion_matrix")
    def test_create_confusion_matrix(self, mock_load):
        # Mock return: cm, total, attrition
        cm = np.array([[50, 10], [5, 35]])
        mock_load.return_value = (cm, 100, 40)

        fig = app.create_confusion_matrix(0.5)
        self.assertIsNotNone(fig)
        # Check if it's a plotly figure
        self.assertTrue(fig.layout.title.text.startswith("<b>Confusion Matrix"))

    @patch("ui.app.httpx.post")
    def test_call_predict_excel_api(self, mock_post):
        mock_post.return_value.status_code = 200
        # Mock base64 excel
        mock_post.return_value.json.return_value = {
            "excel_base64": "UEsFBgAAAAAAAAAAAAAAAAAAAAAAAA=="
        }  # Dummy zip header

        result = app._call_predict_excel_api({})
        self.assertIsNotNone(result)

    @patch("ui.app.httpx.post")
    def test_call_predict_shap_html_api(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b"<html></html>"

        result = app._call_predict_shap_html_api({})
        self.assertEqual(result, b"<html></html>")

    @patch("ui.app.httpx.post")
    def test_call_predict_shap_images_api(self, mock_post):
        mock_post.return_value.status_code = 200
        # Mock valid SHAP images response
        mock_post.return_value.json.return_value = {
            "shap_images": [
                {
                    "employee_id": 1,
                    "img_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mnk5+fdAgAE/AG6tP5u4wAAAABJRU5ErkJggg==",
                }
            ]
        }

        result = app._call_predict_shap_images_api({})
        self.assertIsNotNone(result)

        # Test empty response
        mock_post.return_value.json.return_value = {"shap_images": []}
        result = app._call_predict_shap_images_api({})
        self.assertIsNone(result)

    @patch("ui.app.httpx.get")
    def test_get_job_status_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        status = app._get_job_status("123")
        self.assertIsNone(status)

    @patch("ui.app.httpx.get")
    def test_fetch_job_result_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        excel, shap = app._fetch_job_result("123")
        self.assertIsNone(excel)
        self.assertIsNone(shap)

    @patch("ui.app.httpx.get")
    def test_get_job_status(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "completed"}

        status = app._get_job_status("123")
        self.assertEqual(status, {"status": "completed"})

    @patch("ui.app.httpx.get")
    def test_fetch_job_result(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "excel_base64": "UEsFBgAAAAAAAAAAAAAAAAAAAAAAAA==",
            "shap_images": [
                {
                    "employee_id": 1,
                    "img_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mnk5+fdAgAE/AG6tP5u4wAAAABJRU5ErkJggg==",
                }
            ],
        }

        excel, shap_zip = app._fetch_job_result("123")
        self.assertIsNotNone(excel)
        self.assertIsNotNone(shap_zip)

    def test_load_template_files(self):
        # Mock file opening
        with patch(
            "builtins.open", unittest.mock.mock_open(read_data="header\nrow1\n" * 10)
        ):
            with patch("os.path.join", return_value="dummy_path"):
                templates = app._load_template_files()
                self.assertEqual(len(templates), 3)

    @patch("ui.app.st")
    def test_handle_file_uploads_predict(self, mock_st):
        # Mock session state
        mock_st.session_state = MagicMock()
        mock_st.session_state.user_id = "test_user"
        mock_st.session_state.threshold = 0.5

        # Mock file uploader returning 3 files
        # We need 3 files with correct names
        f1, f2, f3 = MagicMock(), MagicMock(), MagicMock()
        f1.name = "extrait_eval.csv"
        f2.name = "extrait_sirh.csv"
        f3.name = "extrait_sondage.csv"

        mock_st.file_uploader.return_value = [f1, f2, f3]
        mock_st.button.return_value = True  # Click predict

        # Mock data loading - patch ui.app.pd because that's where it's imported
        with patch("ui.app.pd.read_csv") as mock_read:
            mock_read.return_value = pd.DataFrame({"col": [1, 2]})

            # Mock API call
            with patch("ui.app._call_prediction_api") as mock_api:
                mock_api.return_value = {
                    "predictions": [
                        {
                            "id_employee": 1,
                            "probability": 0.8,
                            "prediction": "Leave",
                            "risk_category": "High",
                        }
                    ]
                }

                # Run the function
                app._handle_file_uploads_and_predict()

                # Verify API was called
                mock_api.assert_called()
                mock_st.success.assert_called()

    @patch("ui.app.st")
    def test_main_flow(self, mock_st):
        # Use a simple class or dict for session state to avoid MagicMock issues
        class SessionState:
            def __init__(self):
                self.threshold = 0.5
                self.prediction_triggered = False
                self.predictions_raw = []
                self.report_data = None
                self.user_id = "demo1"

        mock_st.session_state = SessionState()

        # Mock sidebar
        mock_st.sidebar = MagicMock()

        # Mock columns to return correct number of columns based on input
        def columns_side_effect(spec):
            if isinstance(spec, list):
                return [MagicMock() for _ in spec]
            elif isinstance(spec, int):
                return [MagicMock() for _ in range(spec)]
            return [MagicMock(), MagicMock()]

        mock_st.columns.side_effect = columns_side_effect

        # Mock slider to return a float
        mock_st.slider.return_value = 0.5

        # Run main
        app.main()

        # Verify basic calls
        mock_st.set_page_config.assert_called()
        mock_st.title.assert_called()

    @patch("ui.app.st")
    def test_render_results(self, mock_st):
        # Setup session state with results
        class SessionState:
            def __init__(self):
                self.threshold = 0.5
                self.prediction_triggered = True
                self.predictions_raw = []
                self.report_data = pd.DataFrame(
                    {
                        "id_employee": [1],
                        "prediction": [1],
                        "probability": [80.0],
                        "prediction_label": ["Leave"],
                    }
                )
                self.excel_report_bytes = b"excel"
                self.shap_html_bytes = b"html"
                self.user_id = "demo1"

        mock_st.session_state = SessionState()

        # Mock columns side effect
        def columns_side_effect(spec):
            if isinstance(spec, list):
                return [MagicMock() for _ in spec]
            elif isinstance(spec, int):
                return [MagicMock() for _ in range(spec)]
            return [MagicMock(), MagicMock()]

        mock_st.columns.side_effect = columns_side_effect

        # Mock slider
        mock_st.slider.return_value = 0.5

        # Run main (which calls results rendering)
        app.main()

        # Verify metrics and dataframe
        mock_st.metric.assert_called()
        mock_st.dataframe.assert_called()
        mock_st.download_button.assert_called()  # Excel download

    @patch("ui.app.httpx.post")
    def test_call_prediction_api_errors(self, mock_post):
        # Timeout
        mock_post.side_effect = app.httpx.TimeoutException("Time out")
        res = app._call_prediction_api({}, {}, {})
        self.assertEqual(res, {"predictions": []})

        # Request Error
        mock_post.side_effect = app.httpx.RequestError("Net error")
        res = app._call_prediction_api({}, {}, {})
        self.assertEqual(res, {"predictions": []})

        # HTTP Status Error
        mock_post.side_effect = app.httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock(status_code=404)
        )
        res = app._call_prediction_api({}, {}, {})
        self.assertEqual(res, {"predictions": []})

        # General Error
        mock_post.side_effect = Exception("Boom")
        res = app._call_prediction_api({}, {}, {})
        self.assertEqual(res, {"predictions": []})

    @patch("ui.app.httpx.post")
    def test_call_predict_excel_api_errors(self, mock_post):
        mock_post.side_effect = Exception("Error")
        res = app._call_predict_excel_api({})
        self.assertIsNone(res)

        # Empty response
        mock_post.side_effect = None
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {}
        res = app._call_predict_excel_api({})
        self.assertIsNone(res)

    @patch("ui.app.httpx.post")
    def test_call_predict_shap_images_api_errors(self, mock_post):
        mock_post.side_effect = Exception("Error")
        res = app._call_predict_shap_images_api({})
        self.assertIsNone(res)

    @patch("ui.app.httpx.post")
    def test_call_predict_shap_html_api_errors(self, mock_post):
        mock_post.side_effect = Exception("Error")
        res = app._call_predict_shap_html_api({})
        self.assertIsNone(res)

    @patch("ui.app.httpx.post")
    def test_enqueue_report_job_errors(self, mock_post):
        mock_post.side_effect = Exception("Error")
        res = app._enqueue_report_job({})
        self.assertIsNone(res)

        # No job id
        mock_post.side_effect = None
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {}
        res = app._enqueue_report_job({})
        self.assertIsNone(res)

    def test_clear_prediction_results(self):
        # Setup mocked session state
        session = MagicMock()
        with patch("ui.app.st.session_state", session):
            app.clear_prediction_results()
            self.assertEqual(session.prediction_triggered, False)
            self.assertIsNone(session.report_data)

    @patch("ui.app.httpx.post")
    def test_enqueue_report_job_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"job_id": "123"}

        job_id = app._enqueue_report_job({})
        self.assertEqual(job_id, "123")
