from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent

class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file=".env",
		env_file_encoding="utf-8",
		case_sensitive=False,
		extra="ignore",
	)

	# ── Gemini ────────────────────────────────────────────────────────────────
	GEMINI_API_KEY: str
	GEMINI_LLM_MODEL: str = "gemini-2.0-flash"
	# NOTE: Embedding model is hardcoded in retriever.py and build_index.py as
	# "models/gemini-embedding-2" (3072-dim). The old GEMINI_EMBEDDING_MODEL
	# config field pointed to the wrong model string and was never used by
	# those files, so it has been removed to avoid confusion.

	# ── Groq ─────────────────────────────────────────────────────────────────
	GROQ_API_KEY: str
	GROQ_MODEL: str = "llama-3.3-70b-versatile"

	# ── Retrieval ─────────────────────────────────────────────────────────────
	LANCEDB_PATH: str = str(PROJECT_ROOT / "lancedb_index")
	LANCEDB_TABLE: str = "shl_assessments"
	CATALOG_PATH: str = str(PROJECT_ROOT / "catalog" / "catalog.json")
	TOP_K_RETRIEVE: int = 20
	TOP_K_RERANK: int = 10

	# ── HyDE ──────────────────────────────────────────────────────────────────
	HYDE_ENABLED: bool = True

	# ── API ───────────────────────────────────────────────────────────────────
	API_HOST: str = "0.0.0.0"
	API_PORT: int = 8000
	REQUEST_TIMEOUT: int = 25

	# ── Agent ─────────────────────────────────────────────────────────────────
	MAX_TURNS: int = 8
	MAX_RECOMMENDATIONS: int = 10
	MIN_RECOMMENDATIONS: int = 1

	# ── Validators ────────────────────────────────────────────────────────────
	@field_validator("GEMINI_API_KEY")
	@classmethod
	def api_key_must_not_be_empty(cls, v: str) -> str:
		if not v or not v.strip():
			raise ValueError(
				"GEMINI_API_KEY is missing or empty. "
				"Add it to your .env file: GEMINI_API_KEY=your_key_here"
			)
		return v.strip()

	@field_validator("TOP_K_RERANK")
	@classmethod
	def rerank_lte_retrieve(cls, v: int, info: Any) -> int:
		retrieve = info.data.get("TOP_K_RETRIEVE", 20)
		if v > retrieve:
			raise ValueError(
				f"TOP_K_RERANK ({v}) must be <= TOP_K_RETRIEVE ({retrieve})"
			)
		return v

	@field_validator("MAX_RECOMMENDATIONS")
	@classmethod
	def max_recs_within_bounds(cls, v: int) -> int:
		if not (1 <= v <= 10):
			raise ValueError(
				"MAX_RECOMMENDATIONS must be between 1 and 10 "
				"(SHL assignment schema requirement)"
			)
		return v

	@field_validator("REQUEST_TIMEOUT")
	@classmethod
	def timeout_under_hard_limit(cls, v: int) -> int:
		if v >= 30:
			raise ValueError(
				"REQUEST_TIMEOUT must be < 30 seconds "
				"(evaluator hard limit is 30s)"
			)
		return v

# Single instance — import like: from config import settings
settings = Settings()
