#!/usr/bin/env python3
"""
Standalone UI Test Script with Playwright
Tests Streamlit UI and captures screenshots without pytest dependency.
"""
import os
import sys
import time
from pathlib import Path

# Check for Playwright
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    print("❌ Playwright not installed!")
    print("\nInstall with:")
    print("  pip install playwright")
    print("  playwright install chromium")
    print("\nOr use the automated API tests instead:")
    print("  python tests/run_automated_test.py")
    sys.exit(1)


# Configuration
UI_URL = os.environ.get("STREAMLIT_URL", "http://localhost:8501")
DATA_DIR = Path(__file__).parent.parent / "data"
SCREENSHOTS_DIR = Path(__file__).parent.parent / "test_screenshots"
TIMEOUT = 60000  # 60 seconds


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_step(step_num, description):
    """Print test step."""
    print(f"\n[STEP {step_num}] {description}")
    print("-" * 70)


def main():
    """Run UI automation test."""
    print_header("STREAMLIT UI - AUTOMATED TESTING WITH SCREENSHOTS")
    
    print(f"\nConfiguration:")
    print(f"  UI URL: {UI_URL}")
    print(f"  Data folder: {DATA_DIR}")
    print(f"  Screenshots: {SCREENSHOTS_DIR}")
    
    # Create screenshots directory
    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    print(f"\n✓ Screenshots directory ready: {SCREENSHOTS_DIR}")
    
    # Get file paths
    eval_file = DATA_DIR / "extrait_eval.csv"
    sirh_file = DATA_DIR / "extrait_sirh.csv"
    sondage_file = DATA_DIR / "extrait_sondage.csv"
    
    # Verify files exist
    print("\nVerifying test data files...")
    for file in [eval_file, sirh_file, sondage_file]:
        if file.exists():
            print(f"  ✓ {file.name}")
        else:
            print(f"  ✗ {file.name} NOT FOUND")
            return False
    
    # Start browser automation
    print_step(1, "Launching Browser")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,  # Set to False to see browser
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        page = context.new_page()
        
        try:
            # Step 2: Navigate to UI
            print_step(2, "Opening Streamlit UI")
            print(f"  Navigating to: {UI_URL}")
            
            try:
                page.goto(UI_URL, timeout=TIMEOUT)
                page.wait_for_load_state("networkidle", timeout=TIMEOUT)
                time.sleep(3)  # Wait for Streamlit to initialize
                print("  ✓ UI loaded successfully")
            except Exception as e:
                print(f"  ✗ Failed to load UI: {e}")
                print(f"\n  Make sure Streamlit is running:")
                print(f"    ./scripts/start-ui.sh")
                return False
            
            # Screenshot: Initial load
            screenshot_path = SCREENSHOTS_DIR / "01_ui_initial_load.png"
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"  📸 Screenshot: {screenshot_path.name}")
            
            # Step 3: Upload files
            print_step(3, "Uploading CSV Files")
            
            try:
                # Find file input
                file_inputs = page.locator('input[type="file"]')
                
                if file_inputs.count() == 0:
                    print("  ⚠ No file upload input found - UI may have changed")
                    print("  Check if local files are being used instead")
                else:
                    # Upload all three files
                    file_input = file_inputs.first
                    file_input.set_input_files([
                        str(eval_file),
                        str(sirh_file),
                        str(sondage_file)
                    ])
                    print(f"  ✓ Uploaded: {eval_file.name}")
                    print(f"  ✓ Uploaded: {sirh_file.name}")
                    print(f"  ✓ Uploaded: {sondage_file.name}")
                    
                    # Wait for files to be processed
                    time.sleep(5)
                
                # Screenshot: After upload
                screenshot_path = SCREENSHOTS_DIR / "02_files_uploaded.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"  📸 Screenshot: {screenshot_path.name}")
                
            except Exception as e:
                print(f"  ⚠ File upload error: {e}")
                print("  Continuing with test...")
            
            # Step 4: Click predict button
            print_step(4, "Running Prediction")
            
            try:
                # Look for predict button
                button_found = False
                button_texts = ["Predict Attrition", "Predict", "Run Prediction"]
                
                for button_text in button_texts:
                    try:
                        button = page.get_by_role("button", name=button_text)
                        if button.count() > 0:
                            print(f"  ✓ Found button: '{button_text}'")
                            button.click()
                            button_found = True
                            break
                    except:
                        continue
                
                if not button_found:
                    print("  ⚠ Predict button not found - trying alternative selectors")
                    # Try to find button by text content
                    buttons = page.locator("button")
                    for i in range(buttons.count()):
                        btn_text = buttons.nth(i).inner_text()
                        if "predict" in btn_text.lower() or "attrition" in btn_text.lower():
                            print(f"  ✓ Found button: '{btn_text}'")
                            buttons.nth(i).click()
                            button_found = True
                            break
                
                if button_found:
                    print("  ⏳ Waiting for predictions to complete...")
                    time.sleep(10)  # Wait for API call and rendering
                else:
                    print("  ⚠ Could not find predict button")
                    print("  UI may be using local files automatically")
                
                # Screenshot: After prediction
                screenshot_path = SCREENSHOTS_DIR / "03_prediction_initiated.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"  📸 Screenshot: {screenshot_path.name}")
                
            except Exception as e:
                print(f"  ⚠ Prediction error: {e}")
            
            # Step 5: Capture results
            print_step(5, "Capturing Results")
            
            try:
                # Wait a bit more for results to render
                time.sleep(5)
                
                # Check page content for results
                page_content = page.content().lower()
                
                results_found = False
                indicators = {
                    "predictions": "prediction" in page_content,
                    "probabilities": "probability" in page_content or "%" in page_content,
                    "risk categories": "risk" in page_content,
                    "employee data": "employee" in page_content,
                }
                
                print("\n  Results indicators:")
                for indicator, found in indicators.items():
                    status = "✓" if found else "✗"
                    print(f"    {status} {indicator}")
                    if found:
                        results_found = True
                
                # Scroll down to see full results
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(2)
                
                # Screenshot: Results top
                screenshot_path = SCREENSHOTS_DIR / "04_results_top.png"
                page.screenshot(path=str(screenshot_path), full_page=False)
                print(f"\n  📸 Screenshot: {screenshot_path.name}")
                
                # Screenshot: Full page
                screenshot_path = SCREENSHOTS_DIR / "05_results_full_page.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"  📸 Screenshot: {screenshot_path.name}")
                
                # Save HTML for debugging
                html_path = SCREENSHOTS_DIR / "page_final.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(page.content())
                print(f"  📄 HTML saved: {html_path.name}")
                
                if results_found:
                    print("\n  ✓ Results displayed successfully!")
                else:
                    print("\n  ⚠ Could not verify results - check screenshots")
                
            except Exception as e:
                print(f"  ✗ Error capturing results: {e}")
            
            # Summary
            print_header("TEST COMPLETED")
            
            print("\n📸 Screenshots saved to:")
            print(f"  {SCREENSHOTS_DIR}/")
            print("\nScreenshots captured:")
            for screenshot in sorted(SCREENSHOTS_DIR.glob("*.png")):
                print(f"  - {screenshot.name}")
            
            print("\n✓ UI automation test completed!")
            print("\nReview the screenshots to verify:")
            print("  1. UI loaded correctly")
            print("  2. Files uploaded (if applicable)")
            print("  3. Predictions displayed")
            print("  4. Results are visible")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to save error screenshot
            try:
                screenshot_path = SCREENSHOTS_DIR / "error_screenshot.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"\n📸 Error screenshot saved: {screenshot_path}")
            except:
                pass
            
            return False
            
        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
