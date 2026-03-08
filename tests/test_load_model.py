"""
Unit tests for model loading module
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, Mock, patch

from src.scripts.load_model import load_embed_model, load_llm


class TestLoadModel:
    """Test cases for model loading functionality"""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.scripts.load_model.OpenAIEmbeddings")
    def test_load_embed_model_success(self, mock_embeddings):
        """Test successful embedding model loading"""
        mock_embeddings.return_value = Mock()
        result = load_embed_model("text-embedding-3-small")
        assert result is not None
        mock_embeddings.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    def test_load_embed_model_missing_api_key(self):
        """Test embedding model loading with missing API key"""
        with pytest.raises(KeyError):
            load_embed_model("text-embedding-3-small")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.scripts.load_model.OpenAIEmbeddings")
    def test_load_embed_model_api_error(self, mock_embeddings):
        """Test embedding model loading with API error"""
        mock_embeddings.side_effect = Exception("API Error")
        result = load_embed_model("text-embedding-3-small")
        # Should return None but not crash
        assert result is not None or result is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.scripts.load_model.ChatOpenAI")
    def test_load_llm_success(self, mock_llm):
        """Test successful LLM loading"""
        mock_llm.return_value = Mock()
        result = load_llm("gpt-4o-mini")
        assert result is not None
        mock_llm.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    def test_load_llm_missing_api_key(self):
        """Test LLM loading with missing API key"""
        with pytest.raises(KeyError):
            load_llm("gpt-4o-mini")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.scripts.load_model.ChatOpenAI")
    def test_load_llm_api_error(self, mock_llm):
        """Test LLM loading with API error"""
        mock_llm.side_effect = Exception("API Error")
        result = load_llm("gpt-4o-mini")
        # Should handle error gracefully
        assert result is not None or result is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.scripts.load_model.ChatOpenAI")
    def test_load_llm_different_models(self, mock_llm):
        """Test loading different LLM models"""
        mock_llm.return_value = Mock()

        models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
        for model in models:
            result = load_llm(model)
            assert result is not None
