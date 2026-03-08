"""
Unit tests for chunks module
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch

from src.scripts.chunks import divide_chunks


class TestChunks:
    """Test cases for document chunking functionality"""

    def test_divide_chunks_success(self):
        """Test successful chunking"""
        mock_docs = [
            Mock(page_content="A" * 1000, metadata={"source": "test.pdf", "page": 1}),
            Mock(page_content="B" * 1000, metadata={"source": "test.pdf", "page": 2}),
        ]

        result = divide_chunks(mock_docs, chunk_size=500, chunk_overlap=50)
        assert len(result) >= 2

    def test_divide_chunks_custom_size(self):
        """Test chunking with custom size"""
        mock_docs = [
            Mock(page_content="X" * 2000, metadata={"source": "test.pdf", "page": 1})
        ]

        result = divide_chunks(mock_docs, chunk_size=800, chunk_overlap=80)
        assert len(result) > 0

    def test_divide_chunks_empty_docs(self):
        """Test chunking with empty documents"""
        result = divide_chunks([], chunk_size=800, chunk_overlap=80)
        assert len(result) == 0

    def test_divide_chunks_small_document(self):
        """Test chunking document smaller than chunk_size"""
        mock_docs = [
            Mock(page_content="Small", metadata={"source": "test.pdf", "page": 1})
        ]

        result = divide_chunks(mock_docs, chunk_size=800, chunk_overlap=80)
        assert len(result) > 0

    def test_divide_chunks_zero_overlap(self):
        """Test chunking with zero overlap"""
        mock_docs = [
            Mock(page_content="A" * 1000, metadata={"source": "test.pdf", "page": 1})
        ]

        result = divide_chunks(mock_docs, chunk_size=500, chunk_overlap=0)
        assert len(result) > 0

    def test_divide_chunks_exception_handling(self):
        """Test exception handling in chunking"""
        with pytest.raises(Exception):
            divide_chunks(None, chunk_size=800, chunk_overlap=80)
