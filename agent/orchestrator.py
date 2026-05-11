"""
agent/orchestrator.py
─────────────────────
Core conversational agent using direct Groq AsyncAPI.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import groq
from pydantic import BaseModel, field_validator, model_validator

from agent.intent import Intent, classify_intent
from agent.prompts import build_system_prompt
from agent.retriever import Retriever
from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


# ── Pydantic output schema ────────────────────────────────────────────────────
class Assessment(BaseModel):
    name: str
    url: str
    test_type: str

    @field_validator("url")
    @classmethod
    def url_must_be_shl(cls, v: str) -> str:
        if not v.startswith("https://www.shl.com"):
            raise ValueError(f"URL must be from shl.com catalog, got: {v}")
        return v

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Assessment name must not be empty")
        return v.strip()


class AgentResponse(BaseModel):
    reply: str
    recommendations: list[Assessment] = []
    end_of_conversation: bool = False

    @model_validator(mode="after")
    def validate_recommendations_count(self) -> "AgentResponse":
        if len(self.recommendations) > settings.MAX_RECOMMENDATIONS:
            raise ValueError(f"Too many recommendations: {len(self.recommendations)}")
        return self

    @field_validator("reply")
    @classmethod
    def reply_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reply must not be empty")
        return v.strip()


# ── Full catalog URL set ──────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_catalog_urls() -> frozenset[str]:
    catalog_path = Path(settings.CATALOG_PATH)
    if not catalog_path.exists():
        logger.warning("catalog.json not found")
        return frozenset()
    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        urls = frozenset(item["url"] for item in catalog if item.get("url"))
        logger.info("Catalog URLs loaded", extra={"count": len(urls)})
        return urls
    except Exception as e:
        logger.error("Failed to load catalog URLs", extra={"error": str(e)})
        return frozenset()


# ── URL validator ─────────────────────────────────────────────────────────────
def _validate_urls_against_catalog(
    recommendations: list[Assessment],
) -> list[Assessment]:
    catalog_urls = _load_catalog_urls()
    if not catalog_urls:
        return [r for r in recommendations if r.url.startswith("https://www.shl.com")]
    validated = []
    for rec in recommendations:
        if rec.url in catalog_urls:
            validated.append(rec)
        else:
            logger.warning(
                "Hallucinated URL removed",
                extra={"assessment_name": rec.name, "url": rec.url},
            )
    return validated


# ── Compare helper ────────────────────────────────────────────────────────────
def _extract_assessment_names_for_compare(
    messages: list[dict[str, Any]],
) -> list[str]:
    latest = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            latest = msg.get("content", "")
            break
    quoted = re.findall(r'["]([^\"]+)["]|\'([^\']+)\'', latest)
    if quoted:
        return [q[0] or q[1] for q in quoted if q[0] or q[1]]
    tokens = re.findall(r'\b[A-Z][A-Za-z0-9+\-\.]*(?:\s[A-Z][A-Za-z0-9+\-\.]*)*\b', latest)
    if tokens:
        return tokens[:3]
    return []


# ── Async Groq LLM call ───────────────────────────────────────────────────────
async def _call_groq(system_prompt: str, user_prompt: str) -> AgentResponse:
    """Call Groq AsyncAPI and parse structured JSON response."""

    full_system = system_prompt + """

IMPORTANT: You MUST respond with ONLY valid JSON in this exact format:
{
  "reply": "your conversational response here",
  "recommendations": [
    {"name": "assessment name", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
No markdown, no code blocks, no explanation outside JSON.
"""

    raw = ""
    try:
        client = groq.AsyncGroq(api_key=settings.GROQ_API_KEY)
        response = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content
        logger.info("Groq response received", extra={"preview": raw[:100]})

        data = json.loads(raw)

        return AgentResponse(
            reply=data.get("reply", "I need more information to help you."),
            recommendations=[
                Assessment(
                    name=r.get("name", ""),
                    url=r.get("url", "https://www.shl.com/"),
                    test_type=r.get("test_type", ""),
                )
                for r in data.get("recommendations", [])
                if r.get("name") and r.get("url", "").startswith("https://www.shl.com")
            ],
            end_of_conversation=bool(data.get("end_of_conversation", False)),
        )

    except json.JSONDecodeError as e:
        logger.error("JSON parse error", extra={"error": str(e), "raw": raw[:200]})
        raise
    except Exception as e:
        logger.error("Groq API error", extra={"error": str(e)})
        raise


# ── Core orchestrator ─────────────────────────────────────────────────────────
class Orchestrator:
    """Main agent orchestrator. Stateless — full history on every call."""

    def __init__(self) -> None:
        self.retriever = Retriever()
        _load_catalog_urls()

    async def run(
        self,
        messages: list[dict[str, Any]],
    ) -> AgentResponse:
        if not messages:
            return AgentResponse(
                reply=(
                    "Hello! Tell me about the role you're hiring for and "
                    "I'll recommend the right SHL assessments."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # ── Step 1: Classify intent ───────────────────────────────────────────
        intent = classify_intent(messages)
        logger.info("Intent classified", extra={"intent": intent.value})

        # ── Step 2: Retrieve ──────────────────────────────────────────────────
        retrieved: list[dict[str, Any]] = []

        if intent in (Intent.CLARIFY, Intent.OFF_SCOPE):
            retrieved = []
        elif intent == Intent.COMPARE:
            names = _extract_assessment_names_for_compare(messages)
            if names:
                retrieved = self.retriever.retrieve_for_compare(names)
                logger.info("Compare retrieval", extra={"names": names, "found": len(retrieved)})
            if not retrieved:
                retrieved = self.retriever.retrieve(
                    query=messages[-1].get("content", ""),
                    messages=messages,
                )
        else:
            retrieved = self.retriever.retrieve(
                query=messages[-1].get("content", ""),
                messages=messages,
            )

        logger.info("Retrieval complete", extra={"count": len(retrieved)})

        # ── Step 3: Build system prompt ───────────────────────────────────────
        system_prompt = build_system_prompt(
            intent=intent.value,
            retrieved_assessments=retrieved,
        )

        # ── Step 4: Build user prompt ─────────────────────────────────────────
        latest_user_content = messages[-1].get("content", "")

        if len(messages) > 1:
            history_summary = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in messages[:-1]
            )
            user_prompt = (
                f"Conversation so far:\n{history_summary}\n\n"
                f"Latest message: {latest_user_content}"
            )
        else:
            user_prompt = latest_user_content

        # ── Step 5: Call Groq ─────────────────────────────────────────────────
        try:
            response = await _call_groq(system_prompt, user_prompt)
        except Exception as e:
            logger.error("Groq call failed", extra={"error": str(e)})
            return AgentResponse(
                reply=(
                    "I encountered an issue processing your request. "
                    "Could you please rephrase or provide more details about the role?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # ── Step 6: Validate URLs ─────────────────────────────────────────────
        if response.recommendations:
            validated_recs = _validate_urls_against_catalog(response.recommendations)
            response = AgentResponse(
                reply=response.reply,
                recommendations=validated_recs,
                end_of_conversation=response.end_of_conversation,
            )

        # ── Step 7: Force empty recs for CLARIFY / OFF_SCOPE ─────────────────
        if intent in (Intent.CLARIFY, Intent.OFF_SCOPE):
            response = AgentResponse(
                reply=response.reply,
                recommendations=[],
                end_of_conversation=response.end_of_conversation,
            )

        logger.info(
            "Orchestrator response ready",
            extra={
                "intent": intent.value,
                "recommendations": len(response.recommendations),
                "end_of_conversation": response.end_of_conversation,
            },
        )
        return response


# ── Singleton ─────────────────────────────────────────────────────────────────
_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator