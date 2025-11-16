import unittest
import os
import pandas as pd
from streamlit.testing.v1 import AppTest
import sys
import io

# Add the project root to the sys.path to allow importing modules from the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming create_synthetic_data.py is in the root of the project
# from create_synthetic_data import create_synthetic_data

class TestStreamlitInteraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure the data directory exists
        cls.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(cls.data_dir, exist_ok=True)

        # Create synthetic data files for testing
        # For now, we'll manually create simple dataframes
        # In a real scenario, you might call create_synthetic_data()
        cls.eval_df = pd.DataFrame({
            'eval_number': ['E_1', 'E_2', 'E_3'],
            'augmentation_salaire_precedente': ['10%', '6%', '12%'],
            'heures_supplementaires': ['Oui', 'Oui', 'Non'],
            'note_evaluation_actuelle': [3, 1, 4],
            'note_evaluation_precedente': [4, 3, 4],
            'anciennete': [7, 4, 9],
            'id_employee': [1, 2, 3] # Added id_employee for merging
        })
        cls.sirh_df = pd.DataFrame({
            'id_employee': [1, 2, 3],
            'genre': ['m', 'm', 'f'],
            'nombre_heures_travailless': [180, 170, 189],
            'departement': ['IT', 'HR', 'IT'],
            'salaire': [42612, 79285, 30625]
        })
        cls.sondage_df = pd.DataFrame({
            'code_sondage': [1, 2, 3],
            'satisfaction_employee_nature_travail': [2, 2, 3],
            'satisfaction_employee_equipe': [1, 3, 3],
            'satisfaction_employee_equilibre_pro_perso': [1, 4, 3],
            'annees_dans_le_poste_actuel': [4, 4, 4],
            'annees_dans_l_entreprise': [6, 3, 6],
            'annees_sous_responsable_actuel': [1, 2, 4],
            'id_employee': [1, 2, 3] # Added id_employee for merging
        })

        # Save to BytesIO objects to simulate file uploads
        cls.eval_file = io.BytesIO()
        cls.eval_df.to_csv(cls.eval_file, index=False)
        cls.eval_file.seek(0)

        cls.sirh_file = io.BytesIO()
        cls.sirh_df.to_csv(cls.sirh_file, index=False)
        cls.sirh_file.seek(0)

        cls.sondage_file = io.BytesIO()
        cls.sondage_df.to_csv(cls.sondage_file, index=False)
        cls.sondage_file.seek(0)


    def test_app_flow_with_synthetic_data(self):
        """
        Tests the full app flow from loading data to clicking 'Predict'
        and checks for exceptions.
        """
        at = AppTest.from_file("ui/app.py")
        at_initial_run = at.run(timeout=30) # Run once to render widgets

        # Simulate file uploads
        file_uploader_widget = at_initial_run.get("file_uploader")[0]
        self.assertIsNotNone(file_uploader_widget, "File uploader widget not found.")
        at.set_uploaded_files(file_uploader_widget, [
            self.eval_file,
            self.sirh_file,
            self.sondage_file
        ], names=["extrait_eval.csv", "extrait_sirh.csv", "extrait_sondage.csv"])

        # Run the app again to process the uploaded files
        at_after_upload_run = at.run(timeout=30)

        # Find and click the 'Predict Attrition' button.
        predict_button = at_after_upload_run.button(key="Predict Attrition")
        self.assertIsNotNone(predict_button, "Predict Attrition button not found.")
        
        # Run the app after clicking the predict button
        predict_button_run_result = predict_button.click().run(timeout=30);

        # Check if any exceptions occurred during the run
        self.assertFalse(predict_button_run_result.exception, f"App raised an exception: {predict_button_run_result.exception}")

        # Check for st.error messages in the output
        error_messages = [element.body for element in predict_button_run_result.error]
        self.assertEqual(len(error_messages), 0, f"Found error messages in app output: {error_messages}")

        # Optionally, check for success messages or specific content
        self.assertIn("Predictions received successfully!", predict_button_run_result.success[0].body)

        print("\n--- Captured stdout from AppTest ---")
        print("\n".join(predict_button_run_result.stdout))
        print("------------------------------------")
        print("\n--- Captured stderr from AppTest ---")
        print("\n".join(predict_button_run_result.stderr))
        print("------------------------------------")

if __name__ == '__main__':
    unittest.main()
