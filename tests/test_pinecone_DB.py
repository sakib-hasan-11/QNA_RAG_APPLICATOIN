"""
Unit tests for Pinecone database module
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch

from src.utils.pinecone_DB import create_DB


class TestPineconeDB:
    """Test cases for Pinecone database connectivity"""

    @patch.dict(
        os.environ, {"PINECONE_API_KEY": "test-key", "PINECONE_HOST": "test-host"}
    )
    @patch("src.utils.pinecone_DB.Pinecone")
    def test_create_db_success(self, mock_pinecone):
        """Test successful Pinecone connection"""
        mock_pc = Mock()
        mock_index = Mock()
        mock_index.describe_index_stats.return_value = {"dimension": 1536}
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc

        pc, index = create_DB()
        assert pc is not None
        assert index is not None

    @patch.dict(os.environ, {}, clear=True)
    def test_create_db_missing_api_key(self):
        """Test connection with missing API key"""
        with pytest.raises(KeyError):
            create_DB()

    @patch.dict(os.environ, {"PINECONE_API_KEY": "test-key"}, clear=True)
    def test_create_db_missing_host(self):
        """Test connection with missing host"""
        with pytest.raises(KeyError):
            create_DB()

    @patch.dict(
        os.environ, {"PINECONE_API_KEY": "test-key", "PINECONE_HOST": "test-host"}
    )
    @patch("src.utils.pinecone_DB.Pinecone")
    def test_create_db_connection_error(self, mock_pinecone):
        """Test connection error handling"""
        mock_pinecone.side_effect = Exception("Connection failed")

        with pytest.raises(Exception):
            create_DB()

    @patch.dict(
        os.environ, {"PINECONE_API_KEY": "test-key", "PINECONE_HOST": "test-host"}
    )
    @patch("src.utils.pinecone_DB.Pinecone")
    def test_create_db_invalid_credentials(self, mock_pinecone):
        """Test invalid credentials"""
        mock_pinecone.side_effect = Exception("Invalid API key")

        with pytest.raises(Exception):
            create_DB()
