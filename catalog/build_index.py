"""
catalog/build_index.py
──────────────────────
Loads catalog.json, generates Gemini embeddings for each assessment,
and builds a LanceDB vector index for semantic retrieval.

Usage:
    python catalog/build_index.py
    make build-index
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai
import lancedb
import pyarrow as pa

from catalog.scraper import load_catalog
from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 3072
EMBED_DELAY = 0.15
BATCH_SIZE = 20
EMBEDDING_MODEL = "models/gemini-embedding-2"


# ── Schema ────────────────────────────────────────────────────────────────────
def get_lancedb_schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.int32()),
        pa.field("name", pa.utf8()),
        pa.field("url", pa.utf8()),
        pa.field("test_type", pa.utf8()),
        pa.field("description", pa.utf8()),
        pa.field("job_levels", pa.utf8()),
        pa.field("languages", pa.utf8()),
        pa.field("remote_testing", pa.bool_()),
        pa.field("adaptive", pa.bool_()),
        pa.field("duration", pa.utf8()),
        pa.field("embed_text", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    ])


# ── Text preparation ──────────────────────────────────────────────────────────
def build_embed_text(assessment: dict[str, Any]) -> str:
    """
    Build the text string to embed for each assessment.
    Format matches what HyDE will generate — alignment improves recall.
    """
    job_levels = assessment.get("job_levels", [])
    if isinstance(job_levels, str):
        try:
            job_levels = json.loads(job_levels)
        except Exception:
            job_levels = [job_levels] if job_levels else []

    languages = assessment.get("languages", [])
    if isinstance(languages, str):
        try:
            languages = json.loads(languages)
        except Exception:
            languages = []

    parts = [
        f"Assessment: {assessment.get('name', '')}",
        f"Type: {assessment.get('test_type', '')}",
        f"Description: {assessment.get('description', '')}",
        f"Job Levels: {', '.join(job_levels) if job_levels else 'Not specified'}",
        f"Duration: {assessment.get('duration', 'Not specified')}",
        f"Remote Testing: {'Yes' if assessment.get('remote_testing') else 'No'}",
        f"Adaptive: {'Yes' if assessment.get('adaptive') else 'No'}",
        f"Languages: {', '.join(languages) if languages else 'Not specified'}",
    ]
    return " | ".join(p for p in parts if p.split(": ", 1)[-1].strip())


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_text(text: str) -> list[float]:
    """Embed a single text for indexing."""
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="RETRIEVAL_DOCUMENT",
        )
        return result["embedding"]
    except Exception as e:
        logger.error(
            "Embedding failed",
            extra={"error": str(e), "text_preview": text[:100]},
        )
        raise


def embed_query(text: str) -> list[float]:
    """Embed a query string at retrieval time. Used in retriever.py."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="RETRIEVAL_QUERY",
    )
    return result["embedding"]


# ── Index builder ─────────────────────────────────────────────────────────────
def build_index(
    catalog: list[dict[str, Any]],
    lancedb_path: str = settings.LANCEDB_PATH,
    table_name: str = settings.LANCEDB_TABLE,
    overwrite: bool = True,
) -> lancedb.table.Table:
    """
    Embed all assessments and write to LanceDB.

    Args:
        catalog:      List of assessment dicts from catalog.json
        lancedb_path: Directory path for LanceDB storage
        table_name:   LanceDB table name
        overwrite:    Drop and recreate table if it exists

    Returns:
        LanceDB Table object
    """
    logger.info(
        "Starting index build",
        extra={
            "count": len(catalog),
            "lancedb_path": lancedb_path,
            "table": table_name,
        },
    )

    genai.configure(api_key=settings.GEMINI_API_KEY)
    Path(lancedb_path).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(lancedb_path)

    if overwrite and table_name in db.table_names():
        db.drop_table(table_name)
        logger.info("Dropped existing table", extra={"table": table_name})

    records: list[dict[str, Any]] = []
    failed: list[str] = []

    for i, assessment in enumerate(catalog):
        assessment_name = assessment.get("name", f"assessment_{i}")
        try:
            embed_text_str = build_embed_text(assessment)
            embedding = embed_text(embed_text_str)

            job_levels = assessment.get("job_levels", [])
            languages = assessment.get("languages", [])

            record = {
                "id": i,
                "name": assessment.get("name", ""),
                "url": assessment.get("url", ""),
                "test_type": assessment.get("test_type", ""),
                "description": assessment.get("description", ""),
                "job_levels": json.dumps(job_levels if isinstance(job_levels, list) else []),
                "languages": json.dumps(languages if isinstance(languages, list) else []),
                "remote_testing": bool(assessment.get("remote_testing", False)),
                "adaptive": bool(assessment.get("adaptive", False)),
                "duration": assessment.get("duration", ""),
                "embed_text": embed_text_str,
                "vector": embedding,
            }
            records.append(record)

            if (i + 1) % BATCH_SIZE == 0:
                logger.info(
                    "Embedding progress",
                    extra={"done": i + 1, "total": len(catalog)},
                )
                print(f"  Embedded {i + 1}/{len(catalog)}...")

            time.sleep(EMBED_DELAY)

        except Exception as e:
            logger.error(
                "Failed to embed assessment, skipping",
                extra={"assessment_name": assessment_name, "error": str(e)},
            )
            failed.append(assessment_name)
            continue

    if not records:
        raise RuntimeError(
            "No records to index — all embeddings failed. "
            "Check your GEMINI_API_KEY and network connection."
        )

    schema = get_lancedb_schema()
    table = db.create_table(
        table_name,
        data=records,
        schema=schema,
        mode="overwrite",
    )

    table.create_index(
        metric="cosine",
        num_partitions=2,
        num_sub_vectors=16,
    )

    logger.info(
        "Index built successfully",
        extra={
            "indexed": len(records),
            "failed": len(failed),
            "table": table_name,
            "lancedb_path": lancedb_path,
        },
    )

    if failed:
        logger.warning(
            "Some assessments were not indexed",
            extra={"failed_count": len(failed)},
        )
        print(f"\nWARNING: {len(failed)} assessments failed to embed")

    return table


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Loading catalog")
    try:
        catalog = load_catalog(Path(settings.CATALOG_PATH))
        print(f"Loaded {len(catalog)} assessments from {settings.CATALOG_PATH}")

        if not catalog:
            print("ERROR: Catalog is empty. Run: make scrape")
            raise SystemExit(1)

        table = build_index(catalog)

        print(f"\n✓ Index built: {len(table)} assessments indexed")
        print(f"  LanceDB path : {settings.LANCEDB_PATH}")
        print(f"  Table        : {settings.LANCEDB_TABLE}")
        print(f"  Embedding dim: {EMBEDDING_DIM}")
        print(f"  Model        : {EMBEDDING_MODEL}")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        raise SystemExit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        raise SystemExit(1)