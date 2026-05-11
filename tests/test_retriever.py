"""
tests/test_retriever.py
───────────────────────
Tests for the retrieval pipeline.
LanceDB and Gemini API are mocked — no network or disk required.
"""

import pytest
from unittest.mock import patch, MagicMock

from agent import retriever as retriever_module
from agent.retriever import (
    Retriever,
    _build_query_from_messages,
    _rerank,
)


# ── Query builder ─────────────────────────────────────────────────────────────
class TestBuildQueryFromMessages:
    def test_single_user_message(self):
        msgs = [{"role": "user", "content": "Java developer"}]
        result = _build_query_from_messages(msgs)
        assert result == "Java developer"

    def test_combines_all_user_messages(self):
        msgs = [
            {"role": "user", "content": "Java developer"},
            {"role": "assistant", "content": "What level?"},
            {"role": "user", "content": "Mid-level, 4 years experience"},
        ]
        result = _build_query_from_messages(msgs)
        assert "Java developer" in result
        assert "Mid-level" in result
        assert "What level?" not in result  # assistant message excluded

    def test_skips_assistant_messages(self):
        msgs = [
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "Senior analyst role"},
        ]
        result = _build_query_from_messages(msgs)
        assert "How can I help?" not in result
        assert "Senior analyst" in result

    def test_empty_messages(self):
        result = _build_query_from_messages([])
        assert result == ""

    def test_no_user_messages(self):
        msgs = [{"role": "assistant", "content": "Hello"}]
        result = _build_query_from_messages(msgs)
        assert result == ""


# ── Reranking ─────────────────────────────────────────────────────────────────
class TestRerank:
    def test_empty_candidates(self):
        result = _rerank("query", [], top_k=5)
        assert result == []

    def test_rerank_returns_top_k(self):
        candidates = [
            {"name": f"Assessment {i}", "embed_text": f"text {i}", "description": f"desc {i}"}
            for i in range(5)
        ]
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.9, 0.3, 0.7, 0.1, 0.5]

        with patch.object(retriever_module, "_cross_encoder", mock_encoder):
            result = _rerank("Java developer", candidates, top_k=3)
            assert len(result) == 3
            # Should be sorted by score descending: 0.9, 0.7, 0.5
            assert result[0]["name"] == "Assessment 0"  # score 0.9
            assert result[1]["name"] == "Assessment 2"  # score 0.7
            assert result[2]["name"] == "Assessment 4"  # score 0.5

    def test_rerank_fewer_candidates_than_top_k(self):
        candidates = [
            {"name": "A", "embed_text": "text A", "description": "desc A"},
        ]
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.8]

        with patch.object(retriever_module, "_cross_encoder", mock_encoder):
            result = _rerank("query", candidates, top_k=10)
            assert len(result) == 1


# ── Retriever class ───────────────────────────────────────────────────────────
class TestRetriever:
    def test_retrieve_calls_pipeline(self):
        """Test that retrieve() goes through the full pipeline."""
        mock_embedding = [0.1] * 768
        mock_candidates = [
            {
                "name": "Java 8 (New)",
                "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
                "test_type": "K",
                "description": "Java programming assessment",
                "embed_text": "Assessment: Java 8 (New)",
                "job_levels": "[]",
                "languages": "[]",
                "remote_testing": True,
                "adaptive": False,
                "duration": "17 minutes",
            }
        ]
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.9]

        with patch("agent.retriever._generate_hypothetical_document", return_value="hypothetical doc"), \
             patch("agent.retriever._embed_query", return_value=mock_embedding), \
             patch("agent.retriever._dense_retrieve", return_value=mock_candidates), \
               patch.object(retriever_module, "_cross_encoder", mock_encoder):

            retriever = Retriever()
            results = retriever.retrieve(
                query="Java developer",
                messages=[{"role": "user", "content": "Java developer"}],
                use_hyde=True,
            )
            assert len(results) == 1
            assert results[0]["name"] == "Java 8 (New)"

    def test_retrieve_without_hyde(self):
        """Test retrieval works when HyDE is disabled."""
        mock_embedding = [0.1] * 768
        mock_candidates = [
            {
                "name": "Test Assessment",
                "url": "https://www.shl.com/test",
                "test_type": "A",
                "description": "A test",
                "embed_text": "Assessment: Test",
                "job_levels": "[]",
                "languages": "[]",
                "remote_testing": False,
                "adaptive": False,
                "duration": "",
            }
        ]
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.5]

        with patch("agent.retriever._embed_query", return_value=mock_embedding) as mock_embed, \
             patch("agent.retriever._dense_retrieve", return_value=mock_candidates), \
             patch("agent.retriever._generate_hypothetical_document") as mock_hyde, \
               patch.object(retriever_module, "_cross_encoder", mock_encoder):

            retriever = Retriever()
            results = retriever.retrieve(
                query="data analyst",
                use_hyde=False,
            )
            # HyDE should NOT have been called
            mock_hyde.assert_not_called()
            assert len(results) == 1

    def test_hyde_failure_falls_back_to_raw_query(self):
        """If HyDE fails, retrieval should fall back to raw query."""
        mock_embedding = [0.1] * 768
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = []

        with patch("agent.retriever._generate_hypothetical_document", side_effect=Exception("API error")), \
             patch("agent.retriever._embed_query", return_value=mock_embedding) as mock_embed, \
             patch("agent.retriever._dense_retrieve", return_value=[]), \
             patch.object(retriever_module, "_cross_encoder", mock_encoder):

            retriever = Retriever()
            # Should not raise — should fall back gracefully
            results = retriever.retrieve(query="Java developer", use_hyde=True)
            # embed_query was called with raw query (fallback), not hypothetical
            mock_embed.assert_called_once_with("Java developer")

    def test_empty_retrieval_returns_empty(self):
        mock_embedding = [0.1] * 768

        with patch("agent.retriever._generate_hypothetical_document", return_value="hypo"), \
             patch("agent.retriever._embed_query", return_value=mock_embedding), \
             patch("agent.retriever._dense_retrieve", return_value=[]):

            retriever = Retriever()
            results = retriever.retrieve(query="obscure role nobody hires for")
            assert results == []