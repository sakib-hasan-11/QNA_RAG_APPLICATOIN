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
    with patch("src.pipelines.pipeline.RAGPipeline", return_value=mock_pipeline):
        # Clear any cached api module
        if "api" in sys.modules:
            del sys.modules["api"]

        # Import api fresh
        import api

        # Manually set the pipeline to our mock
        api.pipeline = mock_pipeline

        # Yield both app and mock for tests
        yield api.app, mock_pipeline


@pytest.fixture
def client(app_with_mock):
    """Create TestClient with mocked pipeline"""
    from fastapi.testclient import TestClient

    app, mock_pipeline = app_with_mock
    return TestClient(app)


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
