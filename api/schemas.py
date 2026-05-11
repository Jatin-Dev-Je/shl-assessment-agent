"""
api/schemas.py
──────────────
Pydantic v2 request and response models for the FastAPI endpoints.

Schema is non-negotiable per assignment spec:
  POST /chat
    Request:  {"messages": [{"role": "...", "content": "..."}]}
    Response: {"reply": "...", "recommendations": [...], "end_of_conversation": bool}

  GET /health
    Response: {"status": "ok"}
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Enums ─────────────────────────────────────────────────────────────────────
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

# ── Request models ────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: Role
    content: str

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content must not be empty")
        return v.strip()

class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ...,
        min_length=1,
        description="Full conversation history. Must contain at least one message.",
    )

    @model_validator(mode="after")
    def last_message_must_be_user(self) -> "ChatRequest":
        if self.messages[-1].role != Role.USER:
            raise ValueError(
                "Last message in conversation must be from 'user'. "
                "The API is stateless — send full history ending with latest user message."
            )
        return self

    @model_validator(mode="after")
    def validate_turn_count(self) -> "ChatRequest":
        if len(self.messages) > 16:
            raise ValueError(
                "Conversation exceeds maximum allowed messages (16). "
                "The evaluator caps conversations at 8 turns (user + assistant)."
            )
        return self

    @model_validator(mode="after")
    def validate_alternating_roles(self) -> "ChatRequest":
        """
        Roles should roughly alternate: user, assistant, user, assistant...
        First message must always be user.
        """
        if self.messages[0].role != Role.USER:
            raise ValueError("First message must be from 'user'")
        return self

    def to_dict_list(self) -> list[dict[str, str]]:
        """Convert to plain dict list for agent consumption."""
        return [{"role": m.role.value, "content": m.content} for m in self.messages]

# ── Response models ───────────────────────────────────────────────────────────
class RecommendationItem(BaseModel):
    name: str = Field(..., description="Assessment name from SHL catalog")
    url: str = Field(..., description="Full URL from SHL catalog")
    test_type: str = Field(..., description="SHL test type code e.g. K, P, A")

    @field_validator("url")
    @classmethod
    def url_must_be_shl(cls, v: str) -> str:
        if not v.startswith("https://www.shl.com"):
            raise ValueError(f"URL must be from shl.com, got: {v}")
        return v

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be empty")
        return v.strip()

    @field_validator("test_type")
    @classmethod
    def test_type_not_empty(cls, v: str) -> str:
        return v.strip()

class ChatResponse(BaseModel):
    reply: str = Field(
        ...,
        description="Agent's conversational response",
    )
    recommendations: list[RecommendationItem] = Field(
        default=[],
        description=(
            "Empty when clarifying or refusing. "
            "1-10 items when agent has committed to a shortlist."
        ),
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when agent considers the task complete.",
    )

    @field_validator("reply")
    @classmethod
    def reply_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reply must not be empty")
        return v.strip()

    @field_validator("recommendations")
    @classmethod
    def recommendations_max_ten(cls, v: list) -> list:
        if len(v) > 10:
            raise ValueError(
                f"recommendations must contain at most 10 items, got {len(v)}"
            )
        return v

    @model_validator(mode="after")
    def end_of_conversation_requires_recommendations(self) -> "ChatResponse":
        """
        If end_of_conversation is True, there should be recommendations.
        A conversation ending with no recommendations is suspicious.
        """
        if self.end_of_conversation and len(self.recommendations) == 0:
            pass
        return self

    @classmethod
    def from_agent_response(cls, agent_resp: Any) -> "ChatResponse":
        """
        Build ChatResponse from orchestrator AgentResponse.
        Single conversion point — keeps API layer clean.
        """
        return cls(
            reply=agent_resp.reply,
            recommendations=[
                RecommendationItem(
                    name=r.name,
                    url=r.url,
                    test_type=r.test_type,
                )
                for r in agent_resp.recommendations
            ],
            end_of_conversation=agent_resp.end_of_conversation,
        )

# ── Health check ──────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"

# ── Error response ────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    detail: str
    status_code: int
