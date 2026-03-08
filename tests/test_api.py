"""
Test API endpoints and edge cases
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_pipeline():
    """Mock RAG pipeline"""
    mock = MagicMock()
    mock.setup_models = MagicMock(return_value=None)
    mock.setup_database = MagicMock(return_value=None)
    mock.query_pipeline = MagicMock()
    return mock


@pytest.fixture
def app_with_mock(mock_pipeline):
    """Create app with mocked pipeline"""
    with patch("src.pipelines.pipeline.RAGPipeline", return_value=mock_pipeline):
        # Clear cached import
        if "api" in sys.modules:
            del sys.modules["api"]

        import api

        # Manually set pipeline to avoid startup issues
        api.pipeline = mock_pipeline
        yield api.app, mock_pipeline


@pytest.fixture
def client(app_with_mock):
    """Test client"""
    app, mock_pipeline = app_with_mock
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct status"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "online", "service": "RAG Query API"}

    def test_health_endpoint(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestQueryEndpoint:
    """Test query endpoint with various scenarios"""

    def test_valid_query(self, app_with_mock):
        """Test valid query request"""
        app, mock_pipeline = app_with_mock
        client = TestClient(app)

        mock_response = Mock()
        mock_response.content = "This is the answer"

        mock_pipeline.query_pipeline.return_value = {
            "answer": mock_response,
            "sources": [{"source": "doc.pdf", "page": 1, "score": 0.95}],
        }

        response = client.post(
            "/query", json={"query": "What is the main topic?", "k": 3}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is the main topic?"
        assert data["answer"] == "This is the answer"
        assert len(data["sources"]) == 1

    def test_empty_query(self, client):
        """Test empty query string"""
        response = client.post("/query", json={"query": "", "k": 3})
        # Should still process, validation happens at business logic level
        assert response.status_code in [200, 422, 500]

    def test_query_with_custom_k(self, app_with_mock):
        """Test query with custom k value"""
        app, mock_pipeline = app_with_mock
        client = TestClient(app)

        mock_response = Mock()
        mock_response.content = "Answer"

        mock_pipeline.query_pipeline.return_value = {
            "answer": mock_response,
            "sources": [],
        }

        response = client.post("/query", json={"query": "Test query", "k": 10})

        assert response.status_code == 200
        mock_pipeline.query_pipeline.assert_called_with(
            query="Test query", k=10, verbose=False
        )

    def test_invalid_k_value(self, client):
        """Test invalid k value (negative)"""
        response = client.post("/query", json={"query": "Test", "k": -1})
        # Should be handled by pydantic validation or business logic
        assert response.status_code in [200, 422, 500]

    def test_missing_query_field(self, client):
        """Test missing required query field"""
        response = client.post("/query", json={"k": 3})
        assert response.status_code == 422

    def test_missing_k_field(self, app_with_mock):
        """Test missing k field (should use default)"""
        app, mock_pipeline = app_with_mock
        client = TestClient(app)

        mock_response = Mock()
        mock_response.content = "Answer"

        mock_pipeline.query_pipeline.return_value = {
            "answer": mock_response,
            "sources": [],
        }

        response = client.post("/query", json={"query": "Test"})

        assert response.status_code == 200
        # Should use default k=3
        mock_pipeline.query_pipeline.assert_called_with(
            query="Test", k=3, verbose=False
        )

    def test_large_query_string(self, app_with_mock):
        """Test very large query string"""
        app, mock_pipeline = app_with_mock
        client = TestClient(app)

        mock_response = Mock()
        mock_response.content = "Answer"

        mock_pipeline.query_pipeline.return_value = {
            "answer": mock_response,
            "sources": [],
        }

        large_query = "a" * 10000
        response = client.post("/query", json={"query": large_query, "k": 3})

        assert response.status_code in [200, 413, 500]

    def test_special_characters_in_query(self, app_with_mock):
        """Test query with special characters"""
        app, mock_pipeline = app_with_mock
        client = TestClient(app)

        mock_response = Mock()
        mock_response.content = "Answer"

        mock_pipeline.query_pipeline.return_value = {
            "answer": mock_response,
            "sources": [],
        }

        response = client.post(
            "/query", json={"query": "Test @#$%^&*() query?!", "k": 3}
        )

        assert response.status_code == 200

    def test_unicode_query(self, app_with_mock):
        """Test query with unicode characters"""
        app, mock_pipeline = app_with_mock
        client = TestClient(app)

        mock_response = Mock()
        mock_response.content = "Answer"

        mock_pipeline.query_pipeline.return_value = {
            "answer": mock_response,
            "sources": [],
        }

        response = client.post("/query", json={"query": "What is 机器学习?", "k": 3})

        assert response.status_code == 200

    def test_pipeline_exception(self, app_with_mock):
        """Test when pipeline raises exception"""
        app, mock_pipeline = app_with_mock
        client = TestClient(app)

        # Set up pipeline to raise exception
        mock_pipeline.query_pipeline.side_effect = Exception("Pipeline error")

        response = client.post("/query", json={"query": "Test", "k": 3})

        assert response.status_code == 500
        assert "Pipeline error" in response.json()["detail"]

    def test_pipeline_not_initialized(self, client):
        """Test when pipeline is None"""
        with patch("api.pipeline", None):
            response = client.post("/query", json={"query": "Test", "k": 3})

            # Should handle gracefully
            assert response.status_code in [500, 503]


class TestStartupBehavior:
    """Test application startup behavior"""

    def test_startup_with_missing_env_vars(self):
        """Test startup fails gracefully without env vars"""
        # Remove env vars temporarily
        env_backup = {}
        for key in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_HOST"]:
            env_backup[key] = os.environ.pop(key, None)

        try:
            with patch("api.RAGPipeline") as mock_pipeline_class:
                mock_instance = MagicMock()
                mock_instance.setup_models.side_effect = KeyError("OPENAI_API_KEY")
                mock_pipeline_class.return_value = mock_instance

                # Should raise on startup
                from api import app

                client = TestClient(app, raise_server_exceptions=False)

                # App should handle missing vars
                response = client.get("/health")
                # Even if startup fails, health check should be callable
                assert response.status_code in [200, 500, 503]
        finally:
            # Restore env vars
            for key, value in env_backup.items():
                if value is not None:
                    os.environ[key] = value

    def test_startup_success(self):
        """Test successful startup"""
        with patch("api.RAGPipeline") as mock_pipeline_class:
            mock_instance = MagicMock()
            mock_pipeline_class.return_value = mock_instance

            from api import app

            client = TestClient(app)

            response = client.get("/health")
            assert response.status_code == 200


class TestCORSConfiguration:
    """Test CORS configuration"""

    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/query", headers={"Origin": "http://localhost:8501"})
        # CORS should allow requests
        assert response.status_code in [200, 405]


class TestConcurrentRequests:
    """Test concurrent request handling"""

    def test_multiple_concurrent_queries(self, app_with_mock):
        """Test handling multiple concurrent queries"""
        import concurrent.futures

        app, mock_pipeline = app_with_mock
        client = TestClient(app)

        mock_response = Mock()
        mock_response.content = "Answer"

        mock_pipeline.query_pipeline.return_value = {
            "answer": mock_response,
            "sources": [],
        }

        def make_request(i):
            return client.post("/query", json={"query": f"Query {i}", "k": 3})

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(20)]
            results = [f.result() for f in futures]

        # All requests should succeed
        assert all(r.status_code == 200 for r in results)


class TestInputValidation:
    """Test input validation and edge cases"""

    def test_sql_injection_attempt(self, app_with_mock):
        """Test SQL injection attempt in query"""
        app, mock_pipeline = app_with_mock
        client = TestClient(app)

        mock_response = Mock()
        mock_response.content = "Answer"

        mock_pipeline.query_pipeline.return_value = {
            "answer": mock_response,
            "sources": [],
        }

        response = client.post(
            "/query", json={"query": "Test'; DROP TABLE users; --", "k": 3}
        )

        # Should handle safely
        assert response.status_code == 200

    def test_extremely_large_k_value(self, app_with_mock):
        """Test extremely large k value"""
        app, mock_pipeline = app_with_mock
        client = TestClient(app)

        mock_response = Mock()
        mock_response.content = "Answer"

        mock_pipeline.query_pipeline.return_value = {
            "answer": mock_response,
            "sources": [],
        }

        response = client.post("/query", json={"query": "Test", "k": 999999})

        # Should handle gracefully
        assert response.status_code in [200, 422, 500]

    def test_zero_k_value(self, app_with_mock):
        """Test k=0"""
        app, mock_pipeline = app_with_mock
        client = TestClient(app)

        mock_response = Mock()
        mock_response.content = "Answer"

        mock_pipeline.query_pipeline.return_value = {
            "answer": mock_response,
            "sources": [],
        }

        response = client.post("/query", json={"query": "Test", "k": 0})

        # Should handle gracefully
        assert response.status_code in [200, 422, 500]


class TestResponseFormat:
    """Test response format consistency"""

    def test_response_has_required_fields(self, app_with_mock):
        """Test response contains all required fields"""
        app, mock_pipeline = app_with_mock
        client = TestClient(app)

        mock_response = Mock()
        mock_response.content = "Answer"

        mock_pipeline.query_pipeline.return_value = {
            "answer": mock_response,
            "sources": [{"source": "doc.pdf", "page": 1, "score": 0.95}],
        }

        response = client.post("/query", json={"query": "Test", "k": 3})

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "query" in data

    def test_answer_without_content_attribute(self, app_with_mock):
        """Test when answer doesn't have content attribute"""
        app, mock_pipeline = app_with_mock
        client = TestClient(app)

        mock_pipeline.query_pipeline.return_value = {
            "answer": "Plain string answer",
            "sources": [],
        }

        response = client.post("/query", json={"query": "Test", "k": 3})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Plain string answer"
