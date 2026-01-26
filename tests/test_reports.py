import unittest
from unittest.mock import MagicMock
import sys
import os

# Create a more robust mock for streamlit
mock_st = MagicMock()

# Mock session_state as a dict-like object
class MockSessionState(dict):
    pass
mock_st.session_state = MockSessionState()

# Mock st.columns to return a list of mocks that can be unpacked
def mock_columns(n):
    return [MagicMock() for _ in range(n)]
mock_st.columns.side_effect = mock_columns

# Mock other streamlit functions used at module level
mock_st.sidebar = MagicMock()
mock_st.sidebar.__enter__ = MagicMock(return_value=MagicMock())
mock_st.sidebar.__exit__ = MagicMock(return_value=None)

# Mock st.secrets
mock_st.secrets = {}

sys.modules["streamlit"] = mock_st

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import the function
from new.app import generate_text_report

class TestReports(unittest.TestCase):
    def test_generate_text_report_basic(self):
        """Test generating report with minimum data."""
        recommendations = {
            "name": "John Doe",
            "topic": "Diabetes",
            "dietary_plan": {
                "daily_calories": "2000 kcal",
                "foods_to_eat": ["Broccoli", "Fish"],
                "foods_to_avoid": ["Sweets"]
            }
        }
        report = generate_text_report(recommendations)
        
        self.assertIn("John Doe", report)
        self.assertIn("Diabetes", report)
        self.assertIn("2000 kcal", report)
        self.assertIn("Broccoli", report)
        self.assertIn("Sweets", report)

    def test_generate_text_report_empty(self):
        """Test generating report with empty recommendations."""
        recommendations = {}
        report = generate_text_report(recommendations)
        
        self.assertIn("Patient Name: N/A", report)
        self.assertIn("Condition: N/A", report)
        self.assertIn("HEALTH MANAGEMENT PLAN", report)

    def test_generate_text_report_full(self):
        """Test generating report with all sections."""
        recommendations = {
            "name": "Jane Smith",
            "topic": "Hypertension",
            "medications": {
                "medication_details": [
                    {"name": "Amlodipine", "dosage": "5mg", "frequency": "Daily", "duration": "Indefinite"}
                ]
            },
            "doctor_visitation": {
                "urgency": "Routine",
                "specialist_type": "Cardiologist"
            },
            "precautions": {
                "lifestyle_changes": ["Reduce salt"],
                "warning_signs": ["Dizziness"]
            },
            "exercise_recommendations": {
                "duration": "30 mins",
                "frequency": "5 times/week",
                "recommended_exercises": ["Walking"]
            }
        }
        report = generate_text_report(recommendations)
        
        self.assertIn("Jane Smith", report)
        self.assertIn("Amlodipine", report)
        self.assertIn("Cardiologist", report)
        self.assertIn("Reduce salt", report)
        self.assertIn("Walking", report)

if __name__ == "__main__":
    unittest.main()
