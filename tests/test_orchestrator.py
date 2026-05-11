"""
tests/test_orchestrator.py
──────────────────────────
Tests for the main orchestrator pipeline.
Gemini API, LanceDB, and CrossEncoder are all mocked.
Tests each intent path end-to-end through the orchestrator.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from agent.orchestrator import (
    Orchestrator,
    AgentResponse,
    Assessment,
    _validate_urls_against_catalog,
    _extract_assessment_names_for_compare,
)
from agent.intent import Intent


# ── Fixtures ──────────────────────────────────────────────────────────────────
VALID_ASSESSMENT = Assessment(
    name="Java 8 (New)",
    url="https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
    test_type="K",
)

VALID_CATALOG_URLS = frozenset([
    "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
    "https://www.shl.com/solutions/products/product-catalog/view/opq32r/",
    "https://www.shl.com/solutions/products/product-catalog/view/verify-g-plus/",
])

MOCK_RETRIEVED = [
    {
        "name": "Java 8 (New)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
        "test_type": "K",
        "description": "Measures Java programming skills",
        "job_levels": '["Mid-Professional"]',
        "languages": '["English"]',
        "remote_testing": True,
        "adaptive": False,
        "duration": "17 minutes",
    }
]


# ── URL Validation ────────────────────────────────────────────────────────────
class TestValidateUrls:
    def test_valid_url_passes(self):
        recs = [VALID_ASSESSMENT]
        with patch("agent.orchestrator._load_catalog_urls", return_value=VALID_CATALOG_URLS):
            result = _validate_urls_against_catalog(recs)
            assert len(result) == 1
            assert result[0].name == "Java 8 (New)"

    def test_hallucinated_url_filtered(self):
        hallucinated = Assessment(
            name="Fake Assessment",
            url="https://www.shl.com/fake/assessment/that/doesnt/exist/",
            test_type="K",
        )
        with patch("agent.orchestrator._load_catalog_urls", return_value=VALID_CATALOG_URLS):
            result = _validate_urls_against_catalog([hallucinated])
            assert len(result) == 0

    def test_mixed_valid_and_hallucinated(self):
        hallucinated = Assessment(
            name="Fake",
            url="https://www.shl.com/fake/",
            test_type="K",
        )
        recs = [VALID_ASSESSMENT, hallucinated]
        with patch("agent.orchestrator._load_catalog_urls", return_value=VALID_CATALOG_URLS):
            result = _validate_urls_against_catalog(recs)
            assert len(result) == 1
            assert result[0].name == "Java 8 (New)"

    def test_empty_catalog_falls_back_to_domain_check(self):
        """When catalog is empty, validate domain only."""
        recs = [VALID_ASSESSMENT]
        with patch("agent.orchestrator._load_catalog_urls", return_value=frozenset()):
            result = _validate_urls_against_catalog(recs)
            # Domain starts with https://www.shl.com — passes domain check
            assert len(result) == 1


# ── Compare name extraction ───────────────────────────────────────────────────
class TestExtractNamesForCompare:
    def test_extracts_quoted_names(self):
        msgs = [{"role": "user", "content": 'Compare "OPQ32r" and "Verify G+"'}]
        result = _extract_assessment_names_for_compare(msgs)
        assert "OPQ32r" in result
        assert "Verify G+" in result

    def test_extracts_capitalized_names(self):
        msgs = [{"role": "user", "content": "What is the difference between OPQ and GSA?"}]
        result = _extract_assessment_names_for_compare(msgs)
        assert len(result) > 0

    def test_empty_messages_returns_empty(self):
        result = _extract_assessment_names_for_compare([])
        assert result == []


# ── Orchestrator.run — intent paths ──────────────────────────────────────────
class TestOrchestratorRun:
    def _make_mock_agent_response(self, intent: str) -> AgentResponse:
        if intent == "CLARIFY":
            return AgentResponse(
                reply="What role are you hiring for?",
                recommendations=[],
                end_of_conversation=False,
            )
        elif intent == "RECOMMEND":
            return AgentResponse(
                reply="Here are my recommendations.",
                recommendations=[VALID_ASSESSMENT],
                end_of_conversation=False,
            )
        elif intent == "OFF_SCOPE":
            return AgentResponse(
                reply="I can only help with SHL assessment selection.",
                recommendations=[],
                end_of_conversation=False,
            )
        return AgentResponse(reply="OK", recommendations=[], end_of_conversation=False)

    @pytest.mark.asyncio
    async def test_empty_messages_returns_greeting(self):
        orch = Orchestrator()
        response = await orch.run([])
        assert "SHL" in response.reply or "assessment" in response.reply.lower()
        assert response.recommendations == []
        assert response.end_of_conversation is False

    @pytest.mark.asyncio
    async def test_clarify_intent_returns_empty_recommendations(self):
        mock_response = self._make_mock_agent_response("CLARIFY")
        mock_retriever = MagicMock()
        mock_retriever.retrieve = MagicMock(return_value=[])

        with patch("agent.orchestrator.classify_intent", return_value=Intent.CLARIFY), \
            patch("agent.orchestrator._call_groq", new=AsyncMock(return_value=mock_response)), \
            patch("agent.orchestrator._load_catalog_urls", return_value=VALID_CATALOG_URLS), \
            patch.object(Orchestrator, "__init__", lambda self: setattr(self, "retriever", mock_retriever)):

            orch = Orchestrator()
            msgs = [{"role": "user", "content": "I need an assessment"}]
            response = await orch.run(msgs)

            # CLARIFY must always return empty recommendations
            assert response.recommendations == []
            orch.retriever.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_off_scope_returns_empty_recommendations(self):
        mock_response = self._make_mock_agent_response("OFF_SCOPE")

        with patch("agent.orchestrator.classify_intent", return_value=Intent.OFF_SCOPE), \
            patch("agent.orchestrator._call_groq", new=AsyncMock(return_value=mock_response)), \
            patch("agent.orchestrator._load_catalog_urls", return_value=VALID_CATALOG_URLS), \
            patch.object(Orchestrator, "__init__", lambda self: setattr(self, "retriever", MagicMock(retrieve=MagicMock(return_value=[])))):

            orch = Orchestrator()
            msgs = [{"role": "user", "content": "What is the salary for a Java developer?"}]
            response = await orch.run(msgs)

            assert response.recommendations == []

    @pytest.mark.asyncio
    async def test_recommend_intent_returns_recommendations(self):
        mock_response = self._make_mock_agent_response("RECOMMEND")

        mock_retriever = MagicMock()
        mock_retriever.retrieve = MagicMock(return_value=MOCK_RETRIEVED)

        with patch("agent.orchestrator.classify_intent", return_value=Intent.RECOMMEND), \
            patch("agent.orchestrator._call_groq", new=AsyncMock(return_value=mock_response)), \
            patch("agent.orchestrator._load_catalog_urls", return_value=VALID_CATALOG_URLS), \
            patch.object(Orchestrator, "__init__", lambda self: setattr(self, "retriever", mock_retriever)):

            orch = Orchestrator()
            msgs = [
                {"role": "user", "content": "I am hiring a mid-level Java developer"},
                {"role": "assistant", "content": "What seniority level?"},
                {"role": "user", "content": "Mid-level, 4 years experience"},
            ]
            response = await orch.run(msgs)

            assert len(response.recommendations) >= 1

    @pytest.mark.asyncio
    async def test_agent_failure_returns_graceful_error(self):
        """If the LLM call throws, orchestrator should return a helpful message."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = MagicMock(return_value=MOCK_RETRIEVED)

        with patch("agent.orchestrator.classify_intent", return_value=Intent.RECOMMEND), \
            patch("agent.orchestrator._call_groq", new=AsyncMock(side_effect=Exception("Groq API error"))), \
            patch("agent.orchestrator._load_catalog_urls", return_value=VALID_CATALOG_URLS), \
            patch.object(Orchestrator, "__init__", lambda self: setattr(self, "retriever", mock_retriever)):

            orch = Orchestrator()
            msgs = [{"role": "user", "content": "Hire a Java developer"}]
            response = await orch.run(msgs)

            # Should return a helpful fallback, not crash
            assert response.reply != ""
            assert response.recommendations == []
            assert response.end_of_conversation is False