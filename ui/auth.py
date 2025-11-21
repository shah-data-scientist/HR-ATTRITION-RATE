"""Authentication helper module for Streamlit UI."""
import os
import requests
from datetime import datetime
from typing import Optional, Tuple
import streamlit as st

# Get API base URL from environment
API_BASE_URL = os.getenv("API_BASE_URL", "http://fastapi_app:8001")


def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[dict]]:
    """
    Authenticate user via API.

    Args:
        username: Username to authenticate
        password: Password to verify

    Returns:
        Tuple of (success: bool, user_data: dict or None)
        user_data contains: user_id, username, role, last_login
    """
    try:
        # Call the API login endpoint
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            params={"username": username, "password": password},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()

            # Parse the last_login timestamp if present
            last_login = None
            if data.get("last_login"):
                try:
                    last_login = datetime.fromisoformat(data["last_login"])
                except (ValueError, TypeError):
                    last_login = None

            user_data = {
                "user_id": data.get("user_id"),
                "username": data.get("username"),
                "role": data.get("role"),
                "last_login": last_login,
            }
            return True, user_data
        elif response.status_code == 401:
            # Invalid credentials
            return False, None
        else:
            # Other errors
            error_detail = response.json().get("detail", "Unknown error")
            st.error(f"Authentication error: {error_detail}")
            return False, None

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Please ensure the API is running.")
        return False, None
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        return False, None
    except Exception as e:
        st.error(f"Authentication error: {e}")
        return False, None


def check_role(required_role: str = "user") -> bool:
    """
    Check if the current user has the required role.

    Args:
        required_role: Required role ("admin" or "user")

    Returns:
        True if user has required role or higher, False otherwise
    """
    if "authenticated" not in st.session_state or not st.session_state.authenticated:
        return False

    user_role = st.session_state.get("user_role", "user")

    # Admin has access to everything
    if user_role == "admin":
        return True

    # User role check
    if required_role == "user" and user_role == "user":
        return True

    return False


def logout():
    """Logout the current user."""
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.user_role = None
    st.session_state.last_login = None


def require_authentication():
    """
    Decorator/function to require authentication.
    Shows login page if not authenticated.

    Returns:
        True if authenticated, False otherwise
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    return st.session_state.authenticated


def show_login_page():
    """Display the login page."""
    st.set_page_config(
        page_title="HR Attrition Risk - Login",
        page_icon="🔐",
        layout="centered"
    )

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.title("🔐 HR Attrition Risk")
        st.markdown("### Employee Attrition Risk Prediction System")
        st.markdown("---")

        # Login form
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Login", use_container_width=True)

            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    with st.spinner("Authenticating..."):
                        success, user_data = authenticate_user(username, password)

                        if success:
                            # Store user information in session state
                            st.session_state.authenticated = True
                            st.session_state.user_id = user_data["user_id"]
                            st.session_state.username = user_data["username"]
                            st.session_state.user_role = user_data["role"]
                            st.session_state.last_login = user_data["last_login"]

                            st.success(f"Welcome, {username}!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")

        # Information box
        st.markdown("---")
        st.info("""
        **Default Accounts:**

        **Admin Account:**
        - Username: `admin`
        - Access: Full system access

        **Analyst Account:**
        - Username: `analyst`
        - Access: View-only access

        *Note: For security, passwords are set in the environment configuration.*
        """)


def show_user_info():
    """Display current user information in the sidebar."""
    if st.session_state.authenticated:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 User Information")
        st.sidebar.info(f"""
        **Username:** {st.session_state.username}

        **Role:** {st.session_state.user_role.title()}

        **User ID:** {st.session_state.user_id}
        """)

        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()
