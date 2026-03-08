"""
Unit tests for RAG query module
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch

from src.scripts.rag_query import rag_query


class TestRAGQuery:
    """Test cases for RAG query execution"""

    @patch("src.scripts.rag_query.retrieve_query")
    @patch("src.scripts.rag_query.create_prompt")
    def test_rag_query_success(self, mock_create_prompt, mock_retrieve):
        """Test successful RAG query"""
        mock_retrieve.return_value = [
            {
                "id": "doc_0",
                "score": 0.95,
                "text": "Test content",
                "source": "test.pdf",
                "page": 1,
            }
        ]
        mock_create_prompt.return_value = "Test prompt"

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Test answer")

        mock_embedding_model = Mock()
        mock_index = Mock()

        result = rag_query(
            "test query", mock_embedding_model, mock_index, mock_llm, k=3, verbose=False
        )

        assert "answer" in result
        assert "sources" in result
        assert "retrieved_docs" in result
        assert len(result["sources"]) == 1

    @patch("src.scripts.rag_query.retrieve_query")
    @patch("src.scripts.rag_query.create_prompt")
    def test_rag_query_multiple_sources(self, mock_create_prompt, mock_retrieve):
        """Test RAG query with multiple sources"""
        mock_retrieve.return_value = [
            {
                "id": f"doc_{i}",
                "score": 0.9 - i * 0.1,
                "text": f"Content {i}",
                "source": f"test{i}.pdf",
                "page": i,
            }
            for i in range(5)
        ]
        mock_create_prompt.return_value = "Test prompt"

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Test answer")

        result = rag_query("test query", Mock(), Mock(), mock_llm, k=5, verbose=False)

        assert len(result["sources"]) == 5
        assert len(result["retrieved_docs"]) == 5

    @patch("src.scripts.rag_query.retrieve_query")
    def test_rag_query_retrieval_error(self, mock_retrieve):
        """Test handling of retrieval errors"""
        mock_retrieve.side_effect = Exception("Retrieval failed")

        with pytest.raises(Exception):
            rag_query("test query", Mock(), Mock(), Mock(), k=3, verbose=False)

    @patch("src.scripts.rag_query.retrieve_query")
    @patch("src.scripts.rag_query.create_prompt")
    def test_rag_query_llm_error(self, mock_create_prompt, mock_retrieve):
        """Test handling of LLM errors"""
        mock_retrieve.return_value = [
            {
                "id": "doc_0",
                "score": 0.95,
                "text": "Test",
                "source": "test.pdf",
                "page": 1,
            }
        ]
        mock_create_prompt.return_value = "Test prompt"

        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM failed")

        with pytest.raises(Exception):
            rag_query("test query", Mock(), Mock(), mock_llm, k=3, verbose=False)

    @patch("src.scripts.rag_query.retrieve_query")
    @patch("src.scripts.rag_query.create_prompt")
    def test_rag_query_no_results(self, mock_create_prompt, mock_retrieve):
        """Test RAG query with no retrieval results"""
        mock_retrieve.return_value = []
        mock_create_prompt.return_value = "Test prompt"

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="I don't have information")

        result = rag_query("test query", Mock(), Mock(), mock_llm, k=3, verbose=False)

        assert len(result["sources"]) == 0
        assert "answer" in result
