"""
Pytest configuration and fixtures
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set test environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key-12345")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("PINECONE_HOST", "test-host.pinecone.io")


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing"""
    mock = MagicMock()
    mock.setup_models = MagicMock(return_value=None)
    mock.setup_database = MagicMock(return_value=None)
    mock.query_pipeline = MagicMock()
    return mock


@pytest.fixture
def app_with_mock(mock_pipeline):
    """Create FastAPI app with mocked pipeline"""
    # Clear any cached api module
    if "api" in sys.modules:
        del sys.modules["api"]

    # Patch where RAGPipeline is used (api module imports it)
    with patch("src.pipelines.pipeline.RAGPipeline", return_value=mock_pipeline):
        # Import api fresh with patched RAGPipeline
        import api

        # Manually set pipeline to ensure it's available
        api.pipeline = mock_pipeline

        # Reset mock call counts so we can track startup event calls
        mock_pipeline.setup_models.reset_mock()
        mock_pipeline.setup_database.reset_mock()
        mock_pipeline.query_pipeline.reset_mock()


@pytest.fixture(autouse=True)
def reset_imports():
    """Reset imports between tests"""
    yield
    # Cleanup if needed


def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
