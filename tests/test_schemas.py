"""
tests/test_schemas.py
─────────────────────
Tests for API request/response Pydantic schemas.
Covers valid inputs, edge cases, and all validation rules.

Updated to cover strip_sentinel behaviour in from_agent_response().
"""

import pytest
from pydantic import ValidationError

from agent.intent import RECS_SENTINEL
from api.schemas import (
    ChatRequest,
    ChatResponse,
    Message,
    RecommendationItem,
    Role,
)


# ── Message ────────────────────────────────────────────────────────────────────
class TestMessage:
    def test_valid_user_message(self):
        m = Message(role="user", content="Hello")
        assert m.role == Role.USER
        assert m.content == "Hello"

    def test_valid_assistant_message(self):
        m = Message(role="assistant", content="Hi there")
        assert m.role == Role.ASSISTANT

    def test_content_stripped(self):
        m = Message(role="user", content="  hello  ")
        assert m.content == "hello"

    def test_empty_content_raises(self):
        with pytest.raises(ValidationError):
            Message(role="user", content="")

    def test_whitespace_only_content_raises(self):
        with pytest.raises(ValidationError):
            Message(role="user", content="   ")

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            Message(role="system", content="Hello")


# ── ChatRequest ────────────────────────────────────────────────────────────────
class TestChatRequest:
    def test_valid_single_user_message(self):
        req = ChatRequest(messages=[{"role": "user", "content": "I need an assessment"}])
        assert len(req.messages) == 1

    def test_valid_multi_turn(self):
        req = ChatRequest(messages=[
            {"role": "user", "content": "Hiring a Java developer"},
            {"role": "assistant", "content": "What seniority level?"},
            {"role": "user", "content": "Mid-level"},
        ])
        assert len(req.messages) == 3

    def test_last_message_must_be_user(self):
        with pytest.raises(ValidationError):
            ChatRequest(messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ])

    def test_first_message_must_be_user(self):
        with pytest.raises(ValidationError):
            ChatRequest(messages=[
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Hello"},
            ])

    def test_empty_messages_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(messages=[])

    def test_exceeding_16_messages_raises(self):
        messages = []
        for i in range(9):
            messages.append({"role": "user", "content": f"User message {i}"})
            messages.append({"role": "assistant", "content": f"Assistant reply {i}"})
        with pytest.raises(ValidationError):
            ChatRequest(messages=messages)

    def test_to_dict_list(self):
        req = ChatRequest(messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Help me"},
        ])
        result = req.to_dict_list()
        assert result == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Help me"},
        ]


# ── RecommendationItem ─────────────────────────────────────────────────────────
class TestRecommendationItem:
    def test_valid_item(self):
        item = RecommendationItem(
            name="Java 8 (New)",
            url="https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
            test_type="K",
        )
        assert item.name == "Java 8 (New)"

    def test_non_shl_url_raises(self):
        with pytest.raises(ValidationError):
            RecommendationItem(
                name="Fake Test",
                url="https://www.fake.com/test",
                test_type="K",
            )

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            RecommendationItem(
                name="",
                url="https://www.shl.com/test",
                test_type="K",
            )

    def test_whitespace_name_raises(self):
        with pytest.raises(ValidationError):
            RecommendationItem(
                name="   ",
                url="https://www.shl.com/test",
                test_type="K",
            )

    def test_http_url_raises(self):
        with pytest.raises(ValidationError):
            RecommendationItem(
                name="Test",
                url="http://www.shl.com/test",
                test_type="K",
            )


# ── ChatResponse ───────────────────────────────────────────────────────────────
class TestChatResponse:
    def test_valid_response_no_recommendations(self):
        resp = ChatResponse(
            reply="What role are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )
        assert resp.reply == "What role are you hiring for?"
        assert resp.recommendations == []
        assert resp.end_of_conversation is False

    def test_valid_response_with_recommendations(self):
        resp = ChatResponse(
            reply="Here are 2 assessments.",
            recommendations=[
                {
                    "name": "Java 8 (New)",
                    "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
                    "test_type": "K",
                }
            ],
            end_of_conversation=True,
        )
        assert len(resp.recommendations) == 1
        assert resp.end_of_conversation is True

    def test_more_than_10_recommendations_raises(self):
        items = [
            {
                "name": f"Test {i}",
                "url": "https://www.shl.com/test",
                "test_type": "K",
            }
            for i in range(11)
        ]
        with pytest.raises(ValidationError):
            ChatResponse(reply="Here are some tests.", recommendations=items)

    def test_empty_reply_raises(self):
        with pytest.raises(ValidationError):
            ChatResponse(reply="", recommendations=[], end_of_conversation=False)

    def test_defaults(self):
        resp = ChatResponse(reply="Hello")
        assert resp.recommendations == []
        assert resp.end_of_conversation is False

    def test_from_agent_response_strips_sentinel(self):
        from agent.orchestrator import AgentResponse, Assessment

        agent_resp = AgentResponse(
            reply=f"Here is a recommendation. {RECS_SENTINEL}",
            recommendations=[
                Assessment(
                    name="Java 8 (New)",
                    url="https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
                    test_type="K",
                )
            ],
            end_of_conversation=False,
        )
        chat_resp = ChatResponse.from_agent_response(agent_resp, strip_sentinel=True)
        assert RECS_SENTINEL not in chat_resp.reply
        assert chat_resp.reply == "Here is a recommendation."
        assert len(chat_resp.recommendations) == 1

    def test_from_agent_response_keeps_sentinel_when_not_stripped(self):
        from agent.orchestrator import AgentResponse, Assessment

        agent_resp = AgentResponse(
            reply=f"Here is a recommendation. {RECS_SENTINEL}",
            recommendations=[
                Assessment(
                    name="Java 8 (New)",
                    url="https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
                    test_type="K",
                )
            ],
            end_of_conversation=False,
        )
        chat_resp = ChatResponse.from_agent_response(agent_resp, strip_sentinel=False)
        assert RECS_SENTINEL in chat_resp.reply

    def test_from_agent_response_basic(self):
        from agent.orchestrator import AgentResponse, Assessment

        agent_resp = AgentResponse(
            reply="Here is a recommendation.",
            recommendations=[
                Assessment(
                    name="Java 8 (New)",
                    url="https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
                    test_type="K",
                )
            ],
            end_of_conversation=False,
        )
        chat_resp = ChatResponse.from_agent_response(agent_resp)
        assert chat_resp.reply == "Here is a recommendation."
        assert len(chat_resp.recommendations) == 1
        assert chat_resp.recommendations[0].name == "Java 8 (New)"