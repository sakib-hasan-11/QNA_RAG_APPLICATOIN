"""
Unit tests for retrieve module
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch

from src.scripts.retrieve import retrieve_query


class TestRetrieve:
    """Test cases for document retrieval"""

    def test_retrieve_query_success(self):
        """Test successful query retrieval"""
        mock_embedding_model = Mock()
        mock_embedding_model.embed_query.return_value = [0.1] * 1536

        mock_index = Mock()
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "doc_0",
                    "score": 0.95,
                    "metadata": {
                        "text": "Test content",
                        "source": "test.pdf",
                        "page": 1,
                    },
                }
            ]
        }

        results = retrieve_query("test query", mock_embedding_model, mock_index, k=5)

        assert len(results) == 1
        assert results[0]["score"] == 0.95
        assert results[0]["text"] == "Test content"

    def test_retrieve_query_multiple_results(self):
        """Test retrieval of multiple results"""
        mock_embedding_model = Mock()
        mock_embedding_model.embed_query.return_value = [0.1] * 1536

        mock_index = Mock()
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": f"doc_{i}",
                    "score": 0.9 - i * 0.1,
                    "metadata": {
                        "text": f"Content {i}",
                        "source": "test.pdf",
                        "page": i,
                    },
                }
                for i in range(5)
            ]
        }

        results = retrieve_query("test query", mock_embedding_model, mock_index, k=5)

        assert len(results) == 5
        assert results[0]["score"] > results[1]["score"]

    def test_retrieve_query_no_results(self):
        """Test query with no results"""
        mock_embedding_model = Mock()
        mock_embedding_model.embed_query.return_value = [0.1] * 1536

        mock_index = Mock()
        mock_index.query.return_value = {"matches": []}

        results = retrieve_query("test query", mock_embedding_model, mock_index, k=5)

        assert len(results) == 0

    def test_retrieve_query_embedding_error(self):
        """Test handling of embedding errors"""
        mock_embedding_model = Mock()
        mock_embedding_model.embed_query.side_effect = Exception("Embedding failed")
        mock_index = Mock()

        with pytest.raises(Exception):
            retrieve_query("test query", mock_embedding_model, mock_index, k=5)

    def test_retrieve_query_index_error(self):
        """Test handling of index query errors"""
        mock_embedding_model = Mock()
        mock_embedding_model.embed_query.return_value = [0.1] * 1536

        mock_index = Mock()
        mock_index.query.side_effect = Exception("Index error")

        with pytest.raises(Exception):
            retrieve_query("test query", mock_embedding_model, mock_index, k=5)

    def test_retrieve_query_different_k_values(self):
        """Test retrieval with different k values"""
        mock_embedding_model = Mock()
        mock_embedding_model.embed_query.return_value = [0.1] * 1536

        for k in [1, 3, 5, 10]:
            mock_index = Mock()
            mock_index.query.return_value = {
                "matches": [
                    {
                        "id": f"doc_{i}",
                        "score": 0.9,
                        "metadata": {
                            "text": "Content",
                            "source": "test.pdf",
                            "page": i,
                        },
                    }
                    for i in range(k)
                ]
            }

            results = retrieve_query(
                "test query", mock_embedding_model, mock_index, k=k
            )
            assert len(results) <= k
