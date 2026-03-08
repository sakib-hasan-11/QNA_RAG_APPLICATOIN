"""
Unit tests for document reader module
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, Mock, patch

from src.scripts.doc_reader import read_doc


class TestDocReader:
    """Test cases for document reading functionality"""

    def test_read_doc_success(self):
        """Test successful PDF reading"""
        with patch("src.scripts.doc_reader.PyPDFDirectoryLoader") as mock_loader:
            mock_instance = Mock()
            mock_instance.load.return_value = [Mock(page_content="Test content")]
            mock_loader.return_value = mock_instance

            result = read_doc("test_dir")
            assert len(result) > 0
            mock_instance.load.assert_called_once()

    def test_read_doc_file_not_found(self):
        """Test handling of missing files"""
        with patch("src.scripts.doc_reader.PyPDFDirectoryLoader") as mock_loader:
            mock_instance = Mock()
            mock_instance.load.side_effect = FileNotFoundError("File not found")
            mock_loader.return_value = mock_instance

            with pytest.raises(FileNotFoundError):
                read_doc("nonexistent_dir")

    def test_read_doc_general_exception(self):
        """Test handling of general exceptions"""
        with patch("src.scripts.doc_reader.PyPDFDirectoryLoader") as mock_loader:
            mock_instance = Mock()
            mock_instance.load.side_effect = Exception("General error")
            mock_loader.return_value = mock_instance

            with pytest.raises(Exception):
                read_doc("test_dir")

    def test_read_doc_empty_directory(self):
        """Test reading from empty directory"""
        with patch("src.scripts.doc_reader.PyPDFDirectoryLoader") as mock_loader:
            mock_instance = Mock()
            mock_instance.load.return_value = []
            mock_loader.return_value = mock_instance

            result = read_doc("empty_dir")
            assert len(result) == 0
