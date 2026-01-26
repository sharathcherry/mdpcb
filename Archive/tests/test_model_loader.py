import unittest
from unittest.mock import patch, mock_open
import pickle
import sys
import os

# Add the root directory to sys.path to import from 'new'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from new.services.model_loader import load_models

class TestModelLoader(unittest.TestCase):
    def test_load_models_success(self):
        """Test successful loading of models."""
        model_files = {"test_model": "path/to/model.sav"}
        mock_model_data = "mocked model"
        
        with patch("builtins.open", mock_open(read_data=b"dummy data")):
            with patch("pickle.load", return_value=mock_model_data):
                loaded, failed = load_models(model_files)
                
                self.assertIn("test_model", loaded)
                self.assertEqual(loaded["test_model"], mock_model_data)
                self.assertEqual(len(failed), 0)

    def test_load_models_file_not_found(self):
        """Test behavior when a model file is not found."""
        model_files = {"missing_model": "nonexistent/path.sav"}
        
        with patch("builtins.open", side_effect=FileNotFoundError):
            loaded, failed = load_models(model_files)
            
            self.assertEqual(len(loaded), 0)
            self.assertIn("missing_model", failed)
            self.assertEqual(failed["missing_model"], "file not found")

    def test_load_models_exception(self):
        """Test behavior when an unexpected exception occurs during loading."""
        model_files = {"bad_model": "path/to/bad_model.sav"}
        
        with patch("builtins.open", mock_open(read_data=b"dummy data")):
            with patch("pickle.load", side_effect=Exception("Pickle error")):
                loaded, failed = load_models(model_files)
                
                self.assertEqual(len(loaded), 0)
                self.assertIn("bad_model", failed)
                self.assertEqual(failed["bad_model"], "Pickle error")

if __name__ == "__main__":
    unittest.main()
