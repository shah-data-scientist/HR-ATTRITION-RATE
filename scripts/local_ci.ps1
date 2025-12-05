
Write-Host "========================================================"
Write-Host "   LOCAL CI/CD PIPELINE SIMULATION"
Write-Host "========================================================"

$ErrorActionPreference = "Stop"

# --- Step 1: Code Quality ---
Write-Host "`n[1/4] Code Quality Checks..."
Write-Host "  - Running Black..."
poetry run black --check .
if ($LASTEXITCODE -ne 0) { Write-Error "Black formatting check failed." }

Write-Host "  - Running Mypy..."
poetry run mypy --ignore-missing-imports api/ core/ database/ ui/
if ($LASTEXITCODE -ne 0) { Write-Error "Mypy type checking failed." }

Write-Host "  - Running Ruff..."
poetry run ruff check .
if ($LASTEXITCODE -ne 0) { Write-Error "Ruff linting failed." }

Write-Host "✅ Code Quality Checks Passed!"

# --- Step 2: Database Setup ---
Write-Host "`n[2/4] Database Setup..."
# Ensure DB is accessible (Docker must be running)
$db_url = "postgresql://user:password@localhost:5432/hr_attrition_db"
$env:DATABASE_URL = $db_url

Write-Host "  - Initializing Database Schema..."
poetry run python database/init_db.py
if ($LASTEXITCODE -ne 0) { Write-Error "Database initialization failed." }

Write-Host "✅ Database Setup Passed!"

# --- Step 3: Tests ---
Write-Host "`n[3/4] Running Tests..."

# Set environment variables for integration tests against local Docker
$env:API_BASE_URL = "http://localhost:8081"
$env:STREAMLIT_URL = "http://localhost:8581"
# We use the known key that works with the running container (from .env)
$env:API_KEY = "0c1ae40adb7d1d7b6758ffe8697c93f1b451c1ecaaa4cb7a81c26450c7e5f824"

poetry run pytest --ignore=tests/test_automated_e2e.py --ignore=tests/test_ui_automation.py --cov=api.app.main --cov=core --cov=database.models --cov-report=term -v
if ($LASTEXITCODE -ne 0) { Write-Error "Tests failed." }

Write-Host "✅ Tests Passed!"

# --- Step 4: Authentication Test ---
Write-Host "`n[4/4] Authentication Security Test..."
$auth_test_script = @"
import os
os.environ['API_KEY'] = 'test_secure_api_key_for_testing'
from api.auth import verify_password, get_password_hash, generate_api_key

# Test password hashing
password = 'test_password_123'
hashed = get_password_hash(password)
assert verify_password(password, hashed), 'Password verification failed'
assert not verify_password('wrong_password', hashed), 'Should reject wrong password'

# Test API key generation
api_key = generate_api_key()
assert len(api_key) == 64, 'API key should be 64 chars (32 bytes hex)'

print('✅ Auth module verification successful')
"@

$auth_test_file = "temp_auth_test.py"
Set-Content -Path $auth_test_file -Value $auth_test_script
poetry run python $auth_test_file
Remove-Item $auth_test_file

if ($LASTEXITCODE -ne 0) { Write-Error "Authentication security test failed." }

Write-Host "`n========================================================"
Write-Host "🎉 LOCAL CI PIPELINE COMPLETED SUCCESSFULLY!"
Write-Host "========================================================"
