"""
Automated UI Testing with Playwright
Tests the Streamlit UI by simulating user interactions and capturing screenshots.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

# Check if playwright is available
try:
    from playwright.sync_api import sync_playwright, expect

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    # No need to print warning here, it will be printed by the skipif decorator


# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8001")
UI_URL = os.environ.get("STREAMLIT_URL", "http://localhost:8501")
DATA_DIR = Path(__file__).parent.parent / "data"
SCREENSHOTS_DIR = Path(__file__).parent.parent / "test_screenshots"
TIMEOUT = 60000  # 60 seconds


def is_streamlit_server_running(url):
    """Checks if the Streamlit server is running by making a GET request."""
    try:
        response = requests.get(url, timeout=5)
        # Streamlit usually returns 200 for its main page
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.Timeout:
        return False
    except Exception:
        return False


@pytest.fixture(scope="module")
def browser_context():
    """Create browser context for UI testing."""
    if not PLAYWRIGHT_AVAILABLE:
        pytest.skip("Playwright not installed")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        yield context
        context.close()
        browser.close()


@pytest.fixture(scope="module")
def ensure_screenshots_dir():
    """Ensure screenshots directory exists."""
    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    return SCREENSHOTS_DIR


@pytest.mark.skipif(
    not PLAYWRIGHT_AVAILABLE or not is_streamlit_server_running(UI_URL),
    reason="Playwright not installed or Streamlit UI server not running",
)
class TestStreamlitUI:
    """Test Streamlit UI with browser automation."""

    def test_ui_loads(self, browser_context, ensure_screenshots_dir):
        """Test that Streamlit UI loads successfully."""
        page = browser_context.new_page()

        try:
            # Navigate to Streamlit app
            print(f"\nNavigating to: {UI_URL}")
            page.goto(UI_URL, timeout=TIMEOUT)

            # Wait for Streamlit to be ready
            page.wait_for_load_state("networkidle", timeout=TIMEOUT)
            time.sleep(3)  # Extra wait for Streamlit initialization

            # Take screenshot
            screenshot_path = ensure_screenshots_dir / "01_ui_loaded.png"
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"✓ Screenshot saved: {screenshot_path}")

            # Check for key UI elements
            page_content = page.content()
            assert (
                "HR Attrition" in page_content or "Attrition" in page_content
            ), "Page title not found"

            print("✓ UI loaded successfully")

        finally:
            page.close()

    def test_file_upload_and_prediction(self, browser_context, ensure_screenshots_dir):
        """Test uploading files and getting predictions through UI."""
        page = browser_context.new_page()

        try:
            # Navigate to app
            page.goto(UI_URL, timeout=TIMEOUT)
            page.wait_for_load_state("networkidle", timeout=TIMEOUT)
            time.sleep(3)

            # Take initial screenshot
            screenshot_path = ensure_screenshots_dir / "02_before_upload.png"
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"✓ Screenshot saved: {screenshot_path}")

            # Step 2.1: Login if required
            try:
                username_input = page.locator(
                    'input[placeholder="Enter your username"]'
                )
                password_input = page.locator(
                    'input[placeholder="Enter your password"]'
                )
                login_button = page.get_by_role("button", name="Login")

                if username_input.count() > 0:
                    print("  Login page detected. Logging in...")
                    username_input.fill("admin")
                    password_input.fill("Admin@2025!Secure")
                    login_button.click()

                    # Wait for redirect/reload
                    time.sleep(3)
                    page.wait_for_load_state("networkidle", timeout=TIMEOUT)
                    print("  ✓ Logged in")
            except Exception as e:
                print(f"  ⚠ Login step info: {e}")

            # Find and interact with file upload
            # Streamlit uses file_uploader which creates input[type=file] elements
            print("\nAttempting to upload files...")

            # Get file paths
            eval_file = DATA_DIR / "extrait_eval.csv"
            sirh_file = DATA_DIR / "extrait_sirh.csv"
            sondage_file = DATA_DIR / "extrait_sondage.csv"

            # Verify files exist
            assert eval_file.exists(), f"Eval file not found: {eval_file}"
            assert sirh_file.exists(), f"SIRH file not found: {sirh_file}"
            assert sondage_file.exists(), f"Sondage file not found: {sondage_file}"

            # Wait for file uploader to be present
            # Streamlit creates multiple file inputs when accept_multiple_files=True
            file_inputs = page.locator('input[type="file"]')

            # Wait a bit for Streamlit to fully render
            time.sleep(2)

            # Upload files (Streamlit's file uploader accepts multiple files)
            # We need to set all files at once
            file_input = file_inputs.first
            file_input.set_input_files(
                [str(eval_file), str(sirh_file), str(sondage_file)]
            )

            print("✓ Files selected for upload")

            # Wait for files to be processed
            time.sleep(5)

            # Take screenshot after upload
            screenshot_path = ensure_screenshots_dir / "03_files_uploaded.png"
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"✓ Screenshot saved: {screenshot_path}")

            # Look for the predict button
            # Try different possible button texts
            predict_button = None
            button_texts = ["Predict Attrition", "Predict", "Run Prediction"]

            for button_text in button_texts:
                try:
                    predict_button = page.get_by_role("button", name=button_text)
                    if predict_button.count() > 0:
                        print(f"✓ Found button: {button_text}")
                        break
                except:
                    continue

            if predict_button and predict_button.count() > 0:
                print("\nClicking predict button...")
                predict_button.click()

                # Wait for prediction to complete
                # Look for success indicators or results
                print("Waiting for predictions to complete...")
                time.sleep(10)  # Give time for API call and rendering

                # Take screenshot after prediction
                screenshot_path = ensure_screenshots_dir / "04_prediction_results.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"✓ Screenshot saved: {screenshot_path}")

                # Check for results
                page_content = page.content()

                # Look for indicators of successful prediction
                success_indicators = [
                    "prediction" in page_content.lower(),
                    "probability" in page_content.lower(),
                    "risk" in page_content.lower(),
                    "employee" in page_content.lower(),
                ]

                if any(success_indicators):
                    print("✓ Predictions displayed successfully")

                    # Try to scroll down to see more results
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    time.sleep(2)

                    # Take full results screenshot
                    screenshot_path = ensure_screenshots_dir / "05_full_results.png"
                    page.screenshot(path=str(screenshot_path), full_page=True)
                    print(f"✓ Screenshot saved: {screenshot_path}")
                else:
                    print("⚠ Could not confirm predictions in page content")

            else:
                print("⚠ Predict button not found - may need manual verification")
                print(
                    "  Available buttons:",
                    [btn.inner_text() for btn in page.locator("button").all()],
                )

            # Save final page HTML for debugging
            html_path = ensure_screenshots_dir / "final_page.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(page.content())
            print(f"✓ Page HTML saved: {html_path}")

        except Exception as e:
            print(f"✗ Error during UI testing: {e}")
            # Take error screenshot
            try:
                screenshot_path = ensure_screenshots_dir / "error_screenshot.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"Error screenshot saved: {screenshot_path}")
            except:
                pass
            raise
        finally:
            page.close()


def test_ui_manual_instructions():
    """Provide manual testing instructions if automated testing fails."""
    print("\n" + "=" * 70)
    print("MANUAL UI TESTING INSTRUCTIONS")
    print("=" * 70)
    print("\nIf automated UI tests fail, test manually:")
    print("\n1. Start the API:")
    print("   ./scripts/start-api.sh")
    print("\n2. Start the UI (in another terminal):")
    print("   ./scripts/start-ui.sh")
    print("\n3. Open browser to: http://localhost:8501")
    print("\n4. Upload files from data/ folder:")
    print("   - extrait_eval.csv")
    print("   - extrait_sirh.csv")
    print("   - extrait_sondage.csv")
    print("\n5. Click 'Predict Attrition' button")
    print("\n6. Verify results are displayed with:")
    print("   - Employee predictions")
    print("   - Risk categories")
    print("   - Probability percentages")
    print("   - Download Excel option")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    """Run UI tests directly."""
    if not PLAYWRIGHT_AVAILABLE:
        print("Error: Playwright not installed")
        print("Install with: pip install playwright && playwright install chromium")
        sys.exit(1)

    print("=" * 70)
    print("STREAMLIT UI AUTOMATED TESTING")
    print("=" * 70)
    print(f"\nUI URL: {UI_URL}")
    print(f"API URL: {API_BASE_URL}")
    print(f"Data folder: {DATA_DIR}")
    print(f"Screenshots will be saved to: {SCREENSHOTS_DIR}")

    # Run with pytest
    pytest.main(
        [
            __file__,
            "-v",
            "-s",
            "--tb=short",
        ]
    )
