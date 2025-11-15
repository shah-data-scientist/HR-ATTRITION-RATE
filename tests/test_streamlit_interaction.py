
import unittest
import subprocess
import os
import sys
from streamlit.testing.v1 import AppTest

# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestStreamlitInteraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Run the script to generate synthetic data
        subprocess.run(["poetry", "run", "python", "create_synthetic_data.py"], check=True)

    def test_app_flow_with_synthetic_data(self):
        """
        Tests the full app flow from loading data to clicking 'Predict'
        and checks for exceptions.
        """
        # Initialize the AppTest
        at = AppTest.from_file("ui/app.py")

        # Run the app
        at.run(timeout=30)

        # The app should automatically load local files.
        # Now, find and click the 'Predict Attrition' button.
        predict_button = at.button(key="Predict Attrition")
        self.assertIsNotNone(predict_button)
        predict_button.click().run(timeout=30)

        # Check if any exceptions occurred during the run
        self.assertFalse(at.exception, f"App raised an exception: {at.exception}")

if __name__ == '__main__':
    unittest.main()
