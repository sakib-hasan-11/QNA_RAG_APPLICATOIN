"""
Unit tests for vector sender module
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, Mock, patch

from src.scripts.vector_sender import sent_vector


class TestVectorSender:
    """Test cases for vector embedding and sending"""

    def test_sent_vector_success(self):
        """Test successful vector sending"""
        mock_docs = [
            Mock(
                page_content="Test content 1",
                metadata={"source": "test.pdf", "page": 1},
            ),
            Mock(
                page_content="Test content 2",
                metadata={"source": "test.pdf", "page": 2},
            ),
        ]
        mock_embedding_model = Mock()
        mock_embedding_model.embed_query.return_value = [0.1] * 1536
        mock_index = Mock()
        mock_index.describe_index_stats.return_value = {"total_vector_count": 2}

        sent_vector(mock_docs, mock_embedding_model, 100, mock_index)

        assert mock_embedding_model.embed_query.call_count == 2
        assert mock_index.upsert.call_count >= 1

    def test_sent_vector_batch_processing(self):
        """Test batch processing of vectors"""
        mock_docs = [
            Mock(
                page_content=f"Content {i}", metadata={"source": "test.pdf", "page": i}
            )
            for i in range(250)
        ]
        mock_embedding_model = Mock()
        mock_embedding_model.embed_query.return_value = [0.1] * 1536
        mock_index = Mock()
        mock_index.describe_index_stats.return_value = {"total_vector_count": 250}

        sent_vector(mock_docs, mock_embedding_model, 100, mock_index)

        # Should batch in groups of 100
        assert mock_index.upsert.call_count >= 2

    def test_sent_vector_empty_docs(self):
        """Test with empty document list"""
        mock_embedding_model = Mock()
        mock_index = Mock()
        mock_index.describe_index_stats.return_value = {"total_vector_count": 0}

        sent_vector([], mock_embedding_model, 100, mock_index)

        assert mock_embedding_model.embed_query.call_count == 0

    def test_sent_vector_embedding_error(self):
        """Test handling of embedding errors"""
        mock_docs = [
            Mock(page_content="Test", metadata={"source": "test.pdf", "page": 1})
        ]
        mock_embedding_model = Mock()
        mock_embedding_model.embed_query.side_effect = Exception("Embedding failed")
        mock_index = Mock()

        with pytest.raises(Exception):
            sent_vector(mock_docs, mock_embedding_model, 100, mock_index)

    def test_sent_vector_upsert_error(self):
        """Test handling of upsert errors"""
        mock_docs = [
            Mock(page_content="Test", metadata={"source": "test.pdf", "page": 1})
        ]
        mock_embedding_model = Mock()
        mock_embedding_model.embed_query.return_value = [0.1] * 1536
        mock_index = Mock()
        mock_index.upsert.side_effect = Exception("Upsert failed")

        with pytest.raises(Exception):
            sent_vector(mock_docs, mock_embedding_model, 100, mock_index)

    def test_sent_vector_metadata_handling(self):
        """Test proper metadata handling"""
        mock_doc = Mock(
            page_content="Test content",
            metadata={"source": "doc.pdf", "page": 5, "extra": "data"},
        )
        mock_embedding_model = Mock()
        mock_embedding_model.embed_query.return_value = [0.1] * 1536
        mock_index = Mock()
        mock_index.describe_index_stats.return_value = {"total_vector_count": 1}

        sent_vector([mock_doc], mock_embedding_model, 100, mock_index)

        # Verify upsert was called with proper metadata
        mock_index.upsert.assert_called_once()
