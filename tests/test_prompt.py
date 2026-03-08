"""
Unit tests for prompt module
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scripts.prompt import create_prompt


class TestPrompt:
    """Test cases for prompt creation"""

    def test_create_prompt_success(self):
        """Test successful prompt creation"""
        query = "What is the main topic?"
        context_docs = [
            {"text": "This is about AI", "source": "ai.pdf", "page": 1},
            {"text": "This is about ML", "source": "ml.pdf", "page": 2},
        ]

        prompt = create_prompt(query, context_docs)

        assert query in prompt
        assert "This is about AI" in prompt
        assert "This is about ML" in prompt
        assert "Context 1:" in prompt
        assert "Context 2:" in prompt

    def test_create_prompt_single_context(self):
        """Test prompt creation with single context"""
        query = "Test question"
        context_docs = [{"text": "Single context", "source": "test.pdf", "page": 1}]

        prompt = create_prompt(query, context_docs)

        assert query in prompt
        assert "Single context" in prompt

    def test_create_prompt_empty_context(self):
        """Test prompt creation with empty context"""
        query = "Test question"
        context_docs = []

        prompt = create_prompt(query, context_docs)

        assert query in prompt
        assert "Context:" in prompt

    def test_create_prompt_long_context(self):
        """Test prompt creation with long context"""
        query = "What are the findings?"
        context_docs = [
            {"text": "A" * 1000, "source": f"doc{i}.pdf", "page": i} for i in range(10)
        ]

        prompt = create_prompt(query, context_docs)

        assert query in prompt
        assert len(prompt) > 0

    def test_create_prompt_special_characters(self):
        """Test prompt with special characters"""
        query = "What's the cost of $100?"
        context_docs = [
            {"text": "Cost is $100 & tax 10%", "source": "prices.pdf", "page": 1}
        ]

        prompt = create_prompt(query, context_docs)

        assert query in prompt
        assert "$100" in prompt
