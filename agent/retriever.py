"""
agent/retriever.py
──────────────────
Two-stage retrieval pipeline:
  Stage 1: Dense retrieval via LanceDB + Gemini embeddings
  Stage 2: CrossEncoder reranking (ms-marco-MiniLM-L-6-v2)

HyDE (Hypothetical Document Embeddings):
  Instead of embedding the raw user query, we first generate a
  hypothetical ideal assessment description, then embed that.
  This bridges the vocabulary gap between user language and
  catalog language — directly improves Recall@10 on vague queries.

Usage:
    retriever = Retriever()
    results = retriever.retrieve(query="Java developer mid-level")
"""

from __future__ import annotations

import json
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import google.generativeai as genai
import lancedb
from sentence_transformers import CrossEncoder

from agent.prompts import HYDE_PROMPT_TEMPLATE
from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_MODEL = "models/gemini-embedding-2"   # must match build_index.py

# ── Metadata type boosting ────────────────────────────────────────────────────
# Maps SHL test_type codes to keywords that signal user preference for that type.
# Applied after reranking to surface type-matching assessments without hard filtering.
_TYPE_KEYWORDS: dict[str, re.Pattern] = {
    "P": re.compile(
        r"\b(personality|behaviour|behavior|OPQ|trait|style|motivation|preference|"
        r"cultural fit|values|attitude|interpersonal|leadership style)\b",
        re.IGNORECASE,
    ),
    "A": re.compile(
        r"\b(cognitive|ability|aptitude|numerical|verbal|reasoning|inductive|"
        r"deductive|Verify|abstract|critical thinking|problem.?solving|logical)\b",
        re.IGNORECASE,
    ),
    "K": re.compile(
        r"\b(technical|programming|coding|Java|Python|SQL|software|knowledge|"
        r"skill test|computer|IT|data|engineering)\b",
        re.IGNORECASE,
    ),
    "B": re.compile(
        r"\b(situational|judgement|judgment|SJT|scenario|workplace scenario|"
        r"decision.?making|real.?world)\b",
        re.IGNORECASE,
    ),
    "S": re.compile(
        r"\b(simulation|work sample|exercise|assessment centre|in.?tray|"
        r"inbox|case study)\b",
        re.IGNORECASE,
    ),
}


def _artifact_path(path_str: str) -> Path:
    """Resolve runtime artifact paths deterministically from the repo root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


# ── CrossEncoder cache state ─────────────────────────────────────────────────
_cross_encoder: CrossEncoder | None = None
_cross_encoder_loading = False
_cross_encoder_lock = threading.Lock()


# ── Cached loaders ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_cross_encoder() -> CrossEncoder:
    """Load CrossEncoder once and cache in memory."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    with _cross_encoder_lock:
        if _cross_encoder is not None:
            return _cross_encoder
        logger.info("Loading CrossEncoder model", extra={"model": CROSS_ENCODER_MODEL})
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        return _cross_encoder


def _warm_cross_encoder_async() -> None:
    """Warm the CrossEncoder in a background thread without blocking requests."""
    global _cross_encoder_loading

    with _cross_encoder_lock:
        if _cross_encoder is not None or _cross_encoder_loading:
            return
        _cross_encoder_loading = True

    def _worker() -> None:
        global _cross_encoder_loading
        try:
            _load_cross_encoder()
        except Exception as e:
            logger.error("CrossEncoder warmup failed", extra={"error": str(e)})
        finally:
            with _cross_encoder_lock:
                _cross_encoder_loading = False

    threading.Thread(target=_worker, daemon=True).start()


@lru_cache(maxsize=1)
def _load_lancedb_table() -> lancedb.table.Table:
    """Connect to LanceDB and return table. Cached after first call."""
    lancedb_path = _artifact_path(settings.LANCEDB_PATH)
    logger.info(
        "Connecting to LanceDB",
        extra={"path": str(lancedb_path), "table": settings.LANCEDB_TABLE},
    )
    if not lancedb_path.exists():
        raise RuntimeError(
            f"LanceDB path not found: {lancedb_path}. "
            "Build the index locally with: python catalog/build_index.py"
        )

    db = lancedb.connect(str(lancedb_path))
    if settings.LANCEDB_TABLE not in db.table_names():
        raise RuntimeError(
            f"LanceDB table '{settings.LANCEDB_TABLE}' not found in {lancedb_path}. "
            "Build the index locally with: python catalog/build_index.py"
        )
    return db.open_table(settings.LANCEDB_TABLE)


# ── Embedding ─────────────────────────────────────────────────────────────────
def _embed_query(text: str) -> list[float]:
    """Embed a query string with RETRIEVAL_QUERY task type."""
    genai.configure(api_key=settings.GEMINI_API_KEY)
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="RETRIEVAL_QUERY",
    )
    return result["embedding"]


# ── HyDE ──────────────────────────────────────────────────────────────────────
def _generate_hypothetical_document(query: str) -> str:
    """
    HyDE: Generate a hypothetical ideal assessment description for the query.

    Instead of searching with the user's raw words ("Java developer"),
    we generate what a matching assessment description would look like,
    then embed that. This closes the vocabulary gap between
    recruiter language and catalog language.
    """
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name=settings.GEMINI_LLM_MODEL)
    prompt = HYDE_PROMPT_TEMPLATE.format(query=query)

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=200,
        ),
    )
    hypothetical = response.text.strip()
    logger.info(
        "HyDE document generated",
        extra={"query": query[:80], "hypothetical_preview": hypothetical[:120]},
    )
    return hypothetical


# ── Dense retrieval ───────────────────────────────────────────────────────────
def _dense_retrieve(
    query_embedding: list[float],
    top_k: int = settings.TOP_K_RETRIEVE,
) -> list[dict[str, Any]]:
    """
    Retrieve top_k candidates from LanceDB using cosine similarity.
    Returns list of assessment dicts.
    """
    table = _load_lancedb_table()

    # Fetch extra candidates to absorb any job solutions that slip through
    results = (
        table.search(query_embedding)
        .metric("cosine")
        .limit(top_k + 15)
        .to_list()
    )

    # Deserialize JSON-serialized list fields
    for r in results:
        for field in ("job_levels", "languages"):
            if isinstance(r.get(field), str):
                try:
                    r[field] = json.loads(r[field])
                except Exception:
                    r[field] = []

    # Filter pre-packaged Job Solutions that may have been indexed before
    # the scraper filter was applied. Names ending in "Solution" are out-of-scope.
    before = len(results)
    results = [r for r in results if not r.get("name", "").strip().endswith("Solution")]
    results = results[:top_k]
    if len(results) < before:
        logger.info(
            "Filtered job solutions from dense results",
            extra={"removed": before - len(results) - max(0, before - top_k - 15)},
        )

    logger.info(
        "Dense retrieval complete",
        extra={"top_k": top_k, "returned": len(results)},
    )
    return results


# ── CrossEncoder reranking ────────────────────────────────────────────────────
def _rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = settings.TOP_K_RERANK,
) -> list[dict[str, Any]]:
    """
    Rerank candidates using CrossEncoder.

    CrossEncoder scores each (query, document) pair jointly —
    more accurate than cosine similarity alone.
    We use the original user query here (not the HyDE document)
    because the CrossEncoder should score against user intent.
    """
    if not candidates:
        return []

    if _cross_encoder is None:
        _warm_cross_encoder_async()
        logger.info(
            "CrossEncoder not ready yet — returning dense candidates",
            extra={"candidates_in": len(candidates), "top_k": top_k},
        )
        return candidates[:top_k]

    cross_encoder = _load_cross_encoder()

    pairs = [
        (query, candidate.get("embed_text", candidate.get("description", "")))
        for candidate in candidates
    ]

    scores = cross_encoder.predict(pairs)

    scored = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    reranked = [candidate for _, candidate in scored[:top_k]]

    logger.info(
        "Reranking complete",
        extra={
            "candidates_in": len(candidates),
            "top_k": top_k,
            "top_score": float(scored[0][0]) if scored else 0,
        },
    )
    return reranked


# ── Query builder ─────────────────────────────────────────────────────────────
def _build_query_from_messages(messages: list[dict[str, Any]]) -> str:
    """
    Condense full conversation history into a single retrieval query.
    Combines all user messages to capture evolving context.
    """
    user_messages = [
        msg["content"]
        for msg in messages
        if msg.get("role") == "user" and msg.get("content")
    ]
    return " ".join(user_messages)


# ── Metadata type helpers ─────────────────────────────────────────────────────
def _detect_preferred_types(query: str) -> set[str]:
    """Return the set of SHL test_type codes signalled by keywords in the query."""
    return {code for code, pat in _TYPE_KEYWORDS.items() if pat.search(query)}


def _boost_by_type(
    candidates: list[dict[str, Any]],
    preferred_types: set[str],
) -> list[dict[str, Any]]:
    """
    Surface type-matching assessments to the top of the list without hard-filtering.

    Splits the reranked list into preferred / rest, then concatenates.
    The overall length is unchanged — no candidates are dropped.
    """
    if not preferred_types:
        return candidates
    preferred = [c for c in candidates if c.get("test_type", "") in preferred_types]
    rest = [c for c in candidates if c.get("test_type", "") not in preferred_types]
    boosted = preferred + rest
    logger.info(
        "Type boosting applied",
        extra={"preferred_types": list(preferred_types), "boosted": len(preferred)},
    )
    return boosted


# ── Specific assessment lookup ────────────────────────────────────────────────
def get_assessments_by_name(names: list[str]) -> list[dict[str, Any]]:
    """
    Fetch specific assessments by name from LanceDB.
    Used for COMPARE intent — exact lookup, no embedding needed.
    """
    table = _load_lancedb_table()
    results = []

    for name in names:
        try:
            rows = (
                table.search()
                .where(f"lower(name) LIKE '%{name.lower()}%'")
                .limit(1)
                .to_list()
            )
            if rows:
                row = rows[0]
                for field in ("job_levels", "languages"):
                    if isinstance(row.get(field), str):
                        try:
                            row[field] = json.loads(row[field])
                        except Exception:
                            row[field] = []
                results.append(row)
                logger.info("Assessment found by name", extra={"assessment_name": name})
            else:
                logger.warning("Assessment not found by name", extra={"assessment_name": name})
        except Exception as e:
            logger.error(
                "Error looking up assessment by name",
                extra={"assessment_name": name, "error": str(e)},
            )

    return results


# ── Public API ────────────────────────────────────────────────────────────────
class Retriever:
    """
    Main retrieval interface.

    Usage:
        retriever = Retriever()
        results = retriever.retrieve(
            query="Java developer mid-level stakeholder communication",
            messages=conversation_history,
        )
    """

    def retrieve(
        self,
        query: str,
        messages: list[dict[str, Any]] | None = None,
        use_hyde: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        Full retrieval pipeline: HyDE → embed → dense search → rerank → type boost.

        Args:
            query:     Condensed query string
            messages:  Full conversation history (used to build richer query)
            use_hyde:  Override HyDE setting (defaults to settings.HYDE_ENABLED)

        Returns:
            List of top-k reranked (and type-boosted) assessment dicts
        """
        use_hyde = settings.HYDE_ENABLED if use_hyde is None else use_hyde

        if messages:
            full_query = _build_query_from_messages(messages)
        else:
            full_query = query

        # ── Adaptive HyDE ─────────────────────────────────────────────────────
        # After 3+ user turns the concatenated query is dense enough that HyDE's
        # vocabulary bridging adds latency (~1-2s) without meaningful recall gain.
        if use_hyde and messages:
            user_turn_count = sum(1 for m in messages if m.get("role") == "user")
            if user_turn_count >= 3:
                use_hyde = False
                logger.info(
                    "Adaptive HyDE: disabled — conversation already rich",
                    extra={"user_turns": user_turn_count},
                )

        logger.info(
            "Starting retrieval",
            extra={"query_preview": full_query[:100], "hyde": use_hyde},
        )

        # ── Stage 1: HyDE ─────────────────────────────────────────────────────
        if use_hyde:
            try:
                hypothetical_doc = _generate_hypothetical_document(full_query)
                embed_input = hypothetical_doc
            except Exception as e:
                logger.warning(
                    "HyDE generation failed, falling back to raw query",
                    extra={"error": str(e)},
                )
                embed_input = full_query
        else:
            embed_input = full_query

        # ── Stage 2: Embed ────────────────────────────────────────────────────
        try:
            query_embedding = _embed_query(embed_input)
        except Exception as e:
            logger.error("Embedding failed", extra={"error": str(e)})
            raise

        # ── Stage 3: Dense retrieval ──────────────────────────────────────────
        candidates = _dense_retrieve(query_embedding, top_k=settings.TOP_K_RETRIEVE)

        if not candidates:
            logger.warning("No candidates returned from dense retrieval")
            return []

        # ── Stage 4: Rerank ───────────────────────────────────────────────────
        reranked = _rerank(full_query, candidates, top_k=settings.TOP_K_RERANK)

        # ── Stage 5: Type boost ───────────────────────────────────────────────
        # Surface type-matching assessments to the top when the user's query
        # explicitly signals a preference (personality, cognitive, technical…).
        # Does not hard-filter — all reranked candidates are preserved.
        preferred_types = _detect_preferred_types(full_query)
        if preferred_types:
            reranked = _boost_by_type(reranked, preferred_types)

        logger.info(
            "Retrieval pipeline complete",
            extra={"final_count": len(reranked), "type_boost": list(preferred_types)},
        )
        return reranked

    def retrieve_for_compare(self, names: list[str]) -> list[dict[str, Any]]:
        """
        Fetch specific assessments by name for comparison.
        Skips embedding — direct name lookup.
        """
        return get_assessments_by_name(names)