import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure ui module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ui import auth


class TestUIAuth(unittest.TestCase):

    @patch("ui.auth.requests.post")
    def test_authenticate_user_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "user_id": 1,
            "username": "admin",
            "role": "admin",
            "last_login": "2023-01-01T12:00:00",
        }

        success, user = auth.authenticate_user("admin", "pass")
        self.assertTrue(success)
        self.assertEqual(user["username"], "admin")

    @patch("ui.auth.requests.post")
    def test_authenticate_user_exceptions(self, mock_post):
        # Test ConnectionError
        mock_post.side_effect = auth.requests.exceptions.ConnectionError()
        success, user = auth.authenticate_user("u", "p")
        self.assertFalse(success)

        # Test Timeout
        mock_post.side_effect = auth.requests.exceptions.Timeout()
        success, user = auth.authenticate_user("u", "p")
        self.assertFalse(success)

        # Test General Exception
        mock_post.side_effect = Exception("Fail")
        success, user = auth.authenticate_user("u", "p")
        self.assertFalse(success)

    @patch("ui.auth.st")
    def test_show_user_info(self, mock_st):
        mock_st.session_state.authenticated = True
        mock_st.session_state.username = "testuser"
        mock_st.session_state.user_role = "admin"
        mock_st.session_state.user_id = 123

        # Mock button click for logout
        mock_st.sidebar.button.return_value = True

        auth.show_user_info()

        mock_st.sidebar.info.assert_called()
        # Check logout called
        self.assertFalse(mock_st.session_state.authenticated)

    @patch("ui.auth.st")
    def test_check_role(self, mock_st):
        # Use MagicMock for session_state
        mock_session = MagicMock()
        # Configure __contains__ to always return True (key exists)
        mock_session.__contains__.return_value = True
        mock_session.authenticated = True

        # Mock .get() method properly
        def get_side_effect(key, default=None):
            if key == "user_role":
                return mock_session.user_role
            return default

        mock_session.get.side_effect = get_side_effect
        mock_st.session_state = mock_session

        # Test admin role
        mock_session.user_role = "admin"
        self.assertTrue(auth.check_role("user"))
        self.assertTrue(auth.check_role("admin"))

        # Test user role
        mock_session.user_role = "user"
        self.assertTrue(auth.check_role("user"))
        self.assertFalse(auth.check_role("admin"))

    @patch("ui.auth.st")
    def test_logout(self, mock_st):
        mock_st.session_state = MagicMock()
        auth.logout()
        self.assertFalse(mock_st.session_state.authenticated)

    @patch("ui.auth.st")
    def test_show_login_page(self, mock_st):
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.text_input.return_value = "input"
        mock_st.form_submit_button.return_value = True

        # Mock authentication failure path
        with patch("ui.auth.authenticate_user", return_value=(False, None)):
            auth.show_login_page()
            mock_st.error.assert_called_with("Invalid username or password")

        # Mock authentication success path
        with patch(
            "ui.auth.authenticate_user",
            return_value=(
                True,
                {"user_id": 1, "username": "u", "role": "r", "last_login": None},
            ),
        ):
            auth.show_login_page()
            mock_st.success.assert_called()
            mock_st.rerun.assert_called()
