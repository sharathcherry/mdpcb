import unittest
import sys
import os

# Add the root directory to sys.path to import from 'new'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from new.services.llm import _clean_response

class TestLLMUtils(unittest.TestCase):
    def test_clean_response_markdown_json(self):
        """Test cleaning a response wrapped in ```json ... ```."""
        raw = "```json\n{\"key\": \"value\"}\n```"
        expected = "{\"key\": \"value\"}"
        self.assertEqual(_clean_response(raw), expected)

    def test_clean_response_markdown_plain(self):
        """Test cleaning a response wrapped in ``` ... ```."""
        raw = "```\n{\"key\": \"value\"}\n```"
        expected = "{\"key\": \"value\"}"
        self.assertEqual(_clean_response(raw), expected)

    def test_clean_response_no_markdown(self):
        """Test cleaning a response with no markdown formatting."""
        raw = "  {\"key\": \"value\"}  "
        expected = "{\"key\": \"value\"}"
        self.assertEqual(_clean_response(raw), expected)

    def test_clean_response_complex(self):
        """Test cleaning with multiple whitespace and newlines."""
        raw = "\n\n```json\n\n{\"key\": \"value\"}\n\n```\n\n"
        expected = "{\"key\": \"value\"}"
        self.assertEqual(_clean_response(raw), expected)

if __name__ == "__main__":
    unittest.main()
