# Authentication Integration Guide

This guide explains how to integrate the authentication system into the Streamlit UI.

## Files Created

1. **[ui/auth.py](ui/auth.py)** - Authentication helper module with:
   - `authenticate_user()` - Validates credentials against database
   - `check_role()` - Checks user role permissions
   - `logout()` - Logs out current user
   - `require_authentication()` - Checks if user is authenticated
   - `show_login_page()` - Displays login form
   - `show_user_info()` - Shows user info in sidebar

2. **[ui/app_authenticated.py](ui/app_authenticated.py)** - Authenticated wrapper for the main app

## Integration Steps

### Option 1: Use the Authenticated Wrapper (Recommended)

The easiest way to add authentication is to use `app_authenticated.py` instead of `app.py`:

```bash
# Instead of:
streamlit run ui/app.py

# Use:
streamlit run ui/app_authenticated.py
```

Update Docker configuration to use the authenticated version:
- In `docker/Dockerfile.streamlit`, change the CMD to use `ui/app_authenticated.py`

### Option 2: Manual Integration into app.py

Add these lines at the top of `ui/app.py` after the imports:

```python
# Import authentication functions
try:
    from ui.auth import require_authentication, show_login_page, show_user_info
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
```

Then modify the `main()` function to add authentication check at the very beginning:

```python
def main() -> None:
    """Run the Streamlit application."""
    # Check authentication first
    if AUTH_AVAILABLE:
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False

        if not require_authentication():
            show_login_page()
            return  # Exit early if not authenticated

    # Existing code continues here...
    st.set_page_config(
        page_title="HR Attrition Risk", layout="wide", initial_sidebar_state="collapsed"
    )
    # ... rest of the function
```

Add user info display in the sidebar (after the existing sidebar content):

```python
# In the sidebar section
with st.sidebar:
    # Existing sidebar code...

    # Add user info at the end
    if AUTH_AVAILABLE:
        show_user_info()
```

## User Credentials

Default users are automatically created when the database initializes:

### Admin Account
- **Username**: `admin`
- **Password**: `Admin@2025!Secure` (from .env: UI_ADMIN_PASSWORD)
- **Access**: Full system access

### Analyst Account
- **Username**: `analyst`
- **Password**: `Analyst@2025!View` (from .env: UI_USER_PASSWORD)
- **Access**: View-only access (for future role-based restrictions)

## Role-Based Access Control (Future Enhancement)

The authentication system supports role-based access. To implement:

```python
from ui.auth import check_role

# Check if user is admin
if check_role("admin"):
    # Show admin-only features
    st.button("Admin Feature")
else:
    st.info("This feature is only available to administrators")
```

## Testing Authentication

1. Start the application with authentication:
   ```bash
   streamlit run ui/app_authenticated.py
   ```

2. You should see the login page
3. Login with one of the default accounts
4. After successful login, you'll see the main application
5. User info will be displayed in the sidebar
6. Click "Logout" to return to the login page

## Database Connection

The authentication module automatically:
- Connects to the PostgreSQL database (when using Docker)
- Falls back gracefully if database is unavailable
- Updates last_login timestamp on successful authentication
- Verifies passwords using bcrypt hashing

## Security Features

- Passwords stored using bcrypt with automatic salt generation
- Last login timestamp tracking
- Active user status checking
- Session-based authentication
- Role-based access control ready
