"""
Pytest configuration and fixtures
"""

import os
import sys

import pytest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set test environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key-12345")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("PINECONE_HOST", "test-host.pinecone.io")


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
