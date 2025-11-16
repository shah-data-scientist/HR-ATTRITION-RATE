#!/usr/bin/env python3
"""
Verification script to test the HR Attrition Rate application setup.

This script performs the following checks:
1. Import checks for all modules
2. File existence checks (model, data files)
3. Configuration validation
4. Database connection (optional)
"""

import os
import sys
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✓{RESET} {msg}")

def print_error(msg):
    print(f"{RED}✗{RESET} {msg}")

def print_warning(msg):
    print(f"{YELLOW}⚠{RESET} {msg}")

def check_imports():
    """Test that all required modules can be imported."""
    print("\n=== Testing Module Imports ===")
    
    # Add project root to Python path
    import sys
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    modules_to_test = [
        ("core.data_processing", "Core data processing"),
        ("core.preprocess", "Core preprocessing"),
        ("core.schema", "Core schemas"),
        ("core.validation", "Core validation"),
        ("database.models", "Database models"),
        ("database.database", "Database connection"),
        ("api.app.main", "API main application"),
        ("ui.app", "Streamlit UI"),
    ]
    
    all_passed = True
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print_success(f"{description}: {module_name}")
        except Exception as e:
            print_error(f"{description}: {module_name} - {str(e)}")
            all_passed = False
    
    return all_passed

def check_files():
    """Check that required files exist."""
    print("\n=== Checking Required Files ===")
    
    files_to_check = [
        ("outputs/employee_attrition_pipeline.pkl", "Trained model"),
        ("outputs/X_train.parquet", "Training data for SHAP"),
        ("data/extrait_eval.csv", "Sample evaluation data"),
        ("data/extrait_sirh.csv", "Sample SIRH data"),
        ("data/extrait_sondage.csv", "Sample survey data"),
        (".env.example", "Environment template"),
        ("docker-compose.yml", "Docker orchestration"),
        ("Dockerfile.api", "API Dockerfile"),
        ("Dockerfile.streamlit", "UI Dockerfile"),
    ]
    
    all_passed = True
    for filepath, description in files_to_check:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            if size > 0:
                print_success(f"{description}: {filepath} ({size:,} bytes)")
            else:
                print_warning(f"{description}: {filepath} (empty file)")
        else:
            print_error(f"{description}: {filepath} (not found)")
            all_passed = False
    
    return all_passed

def check_scripts():
    """Check that startup scripts exist and are executable."""
    print("\n=== Checking Startup Scripts ===")
    
    scripts = [
        ("scripts/start-api.sh", True),
        ("scripts/start-ui.sh", True),
        ("scripts/start-api.bat", False),
        ("scripts/start-ui.bat", False),
    ]
    
    all_passed = True
    for script_path, should_be_executable in scripts:
        if os.path.exists(script_path):
            if should_be_executable and os.name != 'nt':  # Not Windows
                if os.access(script_path, os.X_OK):
                    print_success(f"{script_path} (executable)")
                else:
                    print_warning(f"{script_path} (not executable - run: chmod +x {script_path})")
            else:
                print_success(f"{script_path}")
        else:
            print_error(f"{script_path} (not found)")
            all_passed = False
    
    return all_passed

def check_documentation():
    """Check that documentation files exist."""
    print("\n=== Checking Documentation ===")
    
    docs = [
        ("README.md", "Main documentation"),
        ("QUICKSTART.md", "Quick start guide"),
        ("DEVELOPMENT.md", "Development guide"),
        ("DEPLOYMENT.md", "Deployment guide"),
        ("docs/ARCHITECTURE.md", "Architecture documentation"),
    ]
    
    all_passed = True
    for doc_path, description in docs:
        if os.path.exists(doc_path):
            print_success(f"{description}: {doc_path}")
        else:
            print_error(f"{description}: {doc_path} (not found)")
            all_passed = False
    
    return all_passed

def check_environment():
    """Check environment configuration."""
    print("\n=== Checking Environment Configuration ===")
    
    if os.path.exists(".env"):
        print_success(".env file exists")
        print_warning("Remember: Never commit .env to version control!")
    else:
        print_warning(".env file not found (copy .env.example to .env)")
    
    if os.path.exists(".env.example"):
        print_success(".env.example template exists")
    else:
        print_error(".env.example template not found")
        return False
    
    return True

def check_api_configuration():
    """Verify API can be configured."""
    print("\n=== Checking API Configuration ===")
    
    try:
        # Add project root to path
        import sys
        project_root = os.getcwd()
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        from api.app.main import app
        print_success(f"API Title: {app.title}")
        print_success(f"API Version: {app.version}")
        print_success("API can be imported and configured")
        return True
    except Exception as e:
        print_error(f"API configuration failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("HR Attrition Rate - Setup Verification")
    print("=" * 60)
    
    # Change to project root if needed
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == 'tests' else script_dir
    os.chdir(project_root)
    print(f"\nProject root: {os.getcwd()}")
    
    results = {
        "Imports": check_imports(),
        "Files": check_files(),
        "Scripts": check_scripts(),
        "Documentation": check_documentation(),
        "Environment": check_environment(),
        "API Configuration": check_api_configuration(),
    }
    
    print("\n" + "=" * 60)
    print("=== Verification Summary ===")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results.items():
        if passed:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print_success("\n🎉 All checks passed! Your setup looks good.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure (if not done)")
        print("2. Start PostgreSQL: docker-compose up db -d")
        print("3. Initialize database: poetry run python database/init_db.py")
        print("4. Start API: ./scripts/start-api.sh (or .bat on Windows)")
        print("5. Start UI: ./scripts/start-ui.sh (or .bat on Windows)")
        print("\nFor more info, see QUICKSTART.md")
        return 0
    else:
        print_error("\n⚠ Some checks failed. Please review the errors above.")
        print("\nFor help, see:")
        print("- QUICKSTART.md for setup instructions")
        print("- DEVELOPMENT.md for development details")
        print("- docs/archive/ for additional troubleshooting")
        return 1

if __name__ == "__main__":
    sys.exit(main())
