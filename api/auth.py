"""
Authentication and Security Module for HR Attrition API
Provides API key authentication and security utilities
"""

import os
import secrets
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
import bcrypt

# API Key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Get API key from environment
VALID_API_KEY = os.getenv("API_KEY", "demo_api_key_change_in_production")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password to compare against

    Returns:
        True if password matches, False otherwise
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            (
                hashed_password.encode("utf-8")
                if isinstance(hashed_password, str)
                else hashed_password
            ),
        )
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plain text password to hash

    Returns:
        Hashed password string
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def generate_api_key(length: int = 32) -> str:
    """
    Generate a secure random API key.

    Args:
        length: Length of the API key in bytes (default: 32)

    Returns:
        Hex string of the generated API key
    """
    return secrets.token_hex(length)


async def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Validate API key from request header.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        Valid API key string

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )

    return api_key


async def get_optional_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """
    Optional API key validation - allows requests without API key.
    Useful for endpoints that have both public and authenticated modes.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        Valid API key string or None
    """
    if api_key is None:
        return None

    if api_key != VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )

    return api_key


def require_api_key(func):
    """
    Decorator to require API key authentication on endpoints.

    Usage:
        @app.get("/protected")
        @require_api_key
        async def protected_endpoint():
            return {"message": "Protected data"}
    """

    async def wrapper(*args, api_key: str = Security(get_api_key), **kwargs):
        return await func(*args, **kwargs)

    return wrapper
