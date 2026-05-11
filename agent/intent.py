"""
agent/intent.py
───────────────
Classifies user intent from conversation history.

Two-stage classification:
  Stage 1: Rule-based fast path (handles obvious cases instantly)
  Stage 2: LLM fallback (handles ambiguous cases)

Intents:
  CLARIFY    — too vague, need more info
  RECOMMEND  — enough context to recommend
  REFINE     — updating existing recommendation
  COMPARE    — comparing two or more assessments
  OFF_SCOPE  — outside SHL assessment domain

Design: rule-based first to save latency + LLM quota.
LLM only called when rules cannot confidently classify.
"""

from __future__ import annotations

import asyncio
import re
from enum import Enum
from typing import Any

import google.generativeai as genai

from agent.prompts import INTENT_CLASSIFIER_PROMPT
from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

# ── Intent enum ───────────────────────────────────────────────────────────────
class Intent(str, Enum):
    CLARIFY = "CLARIFY"
    RECOMMEND = "RECOMMEND"
    REFINE = "REFINE"
    COMPARE = "COMPARE"
    OFF_SCOPE = "OFF_SCOPE"


# ── Sentinel injected by orchestrator into assistant messages ─────────────────
# When the orchestrator produces recommendations it appends this tag to the
# assistant content string so the intent classifier can detect prior recs
# without inspecting the structured recommendations array (which is not
# available in the plain messages list).
RECS_SENTINEL = "<HAS_RECS/>"


# ── In-memory reply cache ─────────────────────────────────────────────────────
# The public API strips RECS_SENTINEL before returning to the client. To keep
# REFINE detection working in a stateless request flow, we also remember the
# stripped assistant reply text for recommendation turns.
_RECOMMENDATION_REPLY_CACHE: set[str] = set()


def register_recommendation_reply(reply_text: str) -> None:
    """Register a shortlist-bearing assistant reply for future REFINE turns."""
    normalized = reply_text.replace(RECS_SENTINEL, "").strip()
    if normalized:
        _RECOMMENDATION_REPLY_CACHE.add(normalized)

# ── Rule patterns ─────────────────────────────────────────────────────────────
_COMPARE_PATTERNS = re.compile(
    r"\b(compare|difference between|vs\.?|versus|which is better|how does .+ differ)\b",
    re.IGNORECASE,
)

_REFINE_PATTERNS = re.compile(
    r"\b(add|also include|remove|exclude|instead|change|update|actually|"
    r"what about|can you add|drop|replace|plus|and also)\b",
    re.IGNORECASE,
)

_OFF_SCOPE_PATTERNS = re.compile(
    r"\b(salary|compensation|legal|law|lawsuit|gdpr|compliance|interview question|"
    r"hire|fire|terminate|performance review|background check|reference|"
    r"ignore previous|forget|pretend|jailbreak|as an? (ai|language model)|"
    r"act as|you are now|override|bypass)\b",
    re.IGNORECASE,
)

_STRONG_RECOMMEND_PATTERNS = re.compile(
    r"\b(job description|jd|job spec|hiring for|looking for|we need|"
    r"role is|position is|candidate should|years? (of )?experience|"
    r"senior|junior|mid.?level|entry.?level|manager|engineer|analyst|"
    r"developer|designer|sales|marketing|finance|operations)\b",
    re.IGNORECASE,
)

_VAGUE_PATTERNS = re.compile(
    r"^(i need (an?|some) assessment|give me (an?|some) assessment|"
    r"recommend (something|an assessment)|what assessments|help me|"
    r"suggest something|any assessment)s?\.?$",
    re.IGNORECASE,
)

# ── Helper ────────────────────────────────────────────────────────────────────
def _conversation_has_recommendations(messages: list[dict[str, Any]]) -> bool:
    """Check if the agent has already returned recommendations in this conversation."""
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if RECS_SENTINEL in content:
                return True
            if content.strip() in _RECOMMENDATION_REPLY_CACHE:
                return True
            if "shl.com" in content.lower():
                return True
    return False

def _get_latest_user_message(messages: list[dict[str, Any]]) -> str:
    """Extract the most recent user message content."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""

def _count_turns(messages: list[dict[str, Any]]) -> int:
    """Count total turns in conversation."""
    return len(messages)

# ── Rule-based classifier ─────────────────────────────────────────────────────
def _rule_based_classify(
    messages: list[dict[str, Any]],
    latest_msg: str,
) -> Intent | None:
    """
    Fast rule-based classification.
    Returns Intent if confident, None if LLM should decide.
    """
    # OFF_SCOPE — highest priority, check first
    if _OFF_SCOPE_PATTERNS.search(latest_msg):
        logger.info("Intent: OFF_SCOPE (rule)", extra={"trigger": "off_scope_pattern"})
        return Intent.OFF_SCOPE
    # COMPARE
    if _COMPARE_PATTERNS.search(latest_msg):
        logger.info("Intent: COMPARE (rule)", extra={"trigger": "compare_pattern"})
        return Intent.COMPARE
    # REFINE — only if recommendations already exist
    if _REFINE_PATTERNS.search(latest_msg) and _conversation_has_recommendations(messages):
        logger.info("Intent: REFINE (rule)", extra={"trigger": "refine_pattern"})
        return Intent.REFINE
    # CLARIFY — obviously vague first message
    if len(messages) <= 1 and _VAGUE_PATTERNS.match(latest_msg):
        logger.info("Intent: CLARIFY (rule)", extra={"trigger": "vague_pattern"})
        return Intent.CLARIFY
    # RECOMMEND — strong signals present
    if _STRONG_RECOMMEND_PATTERNS.search(latest_msg):
        logger.info("Intent: RECOMMEND (rule)", extra={"trigger": "strong_recommend_pattern"})
        return Intent.RECOMMEND
    # Force RECOMMEND near turn limit to avoid infinite clarification
    turn_count = _count_turns(messages)
    if turn_count >= 5:
        logger.info(
            "Intent: RECOMMEND (rule — turn limit approaching)",
            extra={"turns": turn_count},
        )
        return Intent.RECOMMEND
    # Ambiguous — defer to LLM
    return None

# ── LLM classifier ────────────────────────────────────────────────────────────
def _llm_classify(messages: list[dict[str, Any]]) -> Intent:
    """
    LLM-based intent classification fallback.
    Called only when rules are not confident.
    """
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=settings.GEMINI_LLM_MODEL,
        system_instruction=INTENT_CLASSIFIER_PROMPT,
    )
    history_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content'].replace(RECS_SENTINEL, '').strip()}"
        for msg in messages[-6:]
    )
    prompt = f"Conversation history:\n{history_text}\n\nClassify the intent:"
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=10,
            ),
        )
        raw = response.text.strip().upper()
        valid = {i.value for i in Intent}
        if raw in valid:
            logger.info("Intent: LLM classification", extra={"intent": raw})
            return Intent(raw)
        logger.warning(
            "LLM returned unexpected intent, defaulting to CLARIFY",
            extra={"raw": raw},
        )
        return Intent.CLARIFY
    except Exception as e:
        logger.error(
            "LLM intent classification failed, defaulting to CLARIFY",
            extra={"error": str(e)},
        )
        return Intent.CLARIFY


async def _llm_classify_async(messages: list[dict[str, Any]]) -> Intent:
    """
    Async wrapper — runs the blocking Gemini SDK call in a thread pool
    so it never blocks the FastAPI event loop.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _llm_classify, messages)

# ── Public API ────────────────────────────────────────────────────────────────
def classify_intent(messages: list[dict[str, Any]]) -> Intent:
    """
    Classify the intent of the latest user message.
    Two-stage:
      1. Rule-based fast path
      2. LLM fallback if rules are not confident
    Args:
        messages: Full conversation history in
                  [{"role": "user"|"assistant", "content": "..."}] format
    Returns:
        Intent enum value
    """
    if not messages:
        return Intent.CLARIFY
    latest_msg = _get_latest_user_message(messages)
    if not latest_msg:
        return Intent.CLARIFY
    rule_result = _rule_based_classify(messages, latest_msg)
    if rule_result is not None:
        return rule_result
    logger.info("Intent: deferring to LLM classifier", extra={"msg_preview": latest_msg[:80]})
    return _llm_classify(messages)


async def classify_intent_async(messages: list[dict[str, Any]]) -> Intent:
    """
    Async entry point. Runs rule-based path inline; offloads LLM to executor.
    Use this from async orchestrator to avoid blocking the event loop.
    """
    if not messages:
        return Intent.CLARIFY
    latest_msg = _get_latest_user_message(messages)
    if not latest_msg:
        return Intent.CLARIFY
    rule_result = _rule_based_classify(messages, latest_msg)
    if rule_result is not None:
        return rule_result
    logger.info(
        "Intent: deferring to LLM classifier (async)",
        extra={"msg_preview": latest_msg[:80]},
    )
    return await _llm_classify_async(messages)
