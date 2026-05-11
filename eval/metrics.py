"""
eval/metrics.py
───────────────
Evaluation metrics for the SHL Assessment Recommender.

Metrics implemented:
  - Recall@K          : fraction of relevant assessments in top-K recommendations
  - Mean Recall@K     : averaged over all traces
  - MRR               : Mean Reciprocal Rank — how high is the first relevant result
  - Precision@K       : fraction of recommendations that are relevant
  - Hallucination Rate: fraction of recommended URLs not in catalog
  - Turn Efficiency   : average turns taken before first recommendation

These mirror exactly what SHL's automated evaluator measures.
Run locally before submitting to catch issues early.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class TraceResult:
    """Result of evaluating one conversation trace."""
    trace_id: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    hallucination_count: int
    total_recommendations: int
    turns_to_first_recommendation: int
    recommended_names: list[str] = field(default_factory=list)
    expected_names: list[str] = field(default_factory=list)
    hallucinated_urls: list[str] = field(default_factory=list)

@dataclass
class EvalSummary:
    """Aggregated metrics across all traces."""
    total_traces: int
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_reciprocal_rank: float
    hallucination_rate: float
    mean_turns_to_recommendation: float
    trace_results: list[TraceResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_traces": self.total_traces,
            "mean_recall_at_k": round(self.mean_recall_at_k, 4),
            "mean_precision_at_k": round(self.mean_precision_at_k, 4),
            "mean_reciprocal_rank": round(self.mean_reciprocal_rank, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "mean_turns_to_recommendation": round(self.mean_turns_to_recommendation, 2),
            "per_trace": [
                {
                    "trace_id": r.trace_id,
                    "recall_at_k": round(r.recall_at_k, 4),
                    "precision_at_k": round(r.precision_at_k, 4),
                    "reciprocal_rank": round(r.reciprocal_rank, 4),
                    "hallucination_count": r.hallucination_count,
                    "turns_to_first_rec": r.turns_to_first_recommendation,
                    "recommended": r.recommended_names,
                    "expected": r.expected_names,
                    "hallucinated_urls": r.hallucinated_urls,
                }
                for r in self.trace_results
            ],
        }

    def print_report(self) -> None:
        """Print a human-readable evaluation report."""
        print("\n" + "=" * 60)
        print("SHL ASSESSMENT RECOMMENDER — EVAL REPORT")
        print("=" * 60)
        print(f"  Traces evaluated       : {self.total_traces}")
        print(f"  Mean Recall@10         : {self.mean_recall_at_k:.4f}")
        print(f"  Mean Precision@10      : {self.mean_precision_at_k:.4f}")
        print(f"  Mean Reciprocal Rank   : {self.mean_reciprocal_rank:.4f}")
        print(f"  Hallucination Rate     : {self.hallucination_rate:.4f}")
        print(f"  Mean Turns to Rec      : {self.mean_turns_to_recommendation:.2f}")
        print("-" * 60)
        print("Per-trace breakdown:")
        for r in self.trace_results:
            status = "✓" if r.recall_at_k >= 0.5 else "✗"
            print(
                f"  {status} {r.trace_id:<20} "
                f"Recall={r.recall_at_k:.2f}  "
                f"MRR={r.reciprocal_rank:.2f}  "
                f"Turns={r.turns_to_first_recommendation}  "
                f"Halluc={r.hallucination_count}"
            )
        print("=" * 60 + "\n")

# ── Core metric functions ─────────────────────────────────────────────────────
def recall_at_k(
    recommended: list[str],
    relevant: list[str],
    k: int = 10,
) -> float:
    """
    Recall@K = |relevant ∩ top-K recommended| / |relevant|
    """
    if not relevant:
        return 1.0
    top_k = set(_normalize(name) for name in recommended[:k])
    relevant_set = set(_normalize(name) for name in relevant)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)

def precision_at_k(
    recommended: list[str],
    relevant: list[str],
    k: int = 10,
) -> float:
    """
    Precision@K = |relevant ∩ top-K recommended| / K
    """
    if not recommended:
        return 0.0
    top_k = [_normalize(name) for name in recommended[:k]]
    relevant_set = set(_normalize(name) for name in relevant)
    hits = sum(1 for name in top_k if name in relevant_set)
    return hits / min(k, len(recommended))

def reciprocal_rank(
    recommended: list[str],
    relevant: list[str],
) -> float:
    """
    Reciprocal Rank = 1 / rank of first relevant item.
    0 if no relevant item found.
    """
    relevant_set = set(_normalize(name) for name in relevant)
    for i, name in enumerate(recommended):
        if _normalize(name) in relevant_set:
            return 1.0 / (i + 1)
    return 0.0

def hallucination_rate(
    recommended_urls: list[str],
    catalog_urls: set[str],
) -> tuple[float, list[str]]:
    """
    Hallucination Rate = fraction of recommended URLs not in catalog.
    """
    if not recommended_urls:
        return 0.0, []
    hallucinated = [url for url in recommended_urls if url not in catalog_urls]
    rate = len(hallucinated) / len(recommended_urls)
    return rate, hallucinated

def turns_to_first_recommendation(conversation: list[dict[str, Any]]) -> int:
    """
    Count how many turns elapsed before the first non-empty recommendations list.
    """
    for i, turn in enumerate(conversation):
        if turn.get("recommendations"):
            return i + 1
    return 8  # MAX_TURNS — never recommended

# ── Aggregation ───────────────────────────────────────────────────────────────
def compute_trace_result(
    trace_id: str,
    recommended_items: list[dict[str, str]],
    expected_names: list[str],
    catalog_urls: set[str],
    conversation: list[dict[str, Any]],
    k: int = 10,
) -> TraceResult:
    """
    Compute all metrics for a single trace.
    """
    recommended_names = [item["name"] for item in recommended_items]
    recommended_urls = [item["url"] for item in recommended_items]
    r_at_k = recall_at_k(recommended_names, expected_names, k)
    p_at_k = precision_at_k(recommended_names, expected_names, k)
    rr = reciprocal_rank(recommended_names, expected_names)
    h_rate, h_urls = hallucination_rate(recommended_urls, catalog_urls)
    turns = turns_to_first_recommendation(conversation)
    return TraceResult(
        trace_id=trace_id,
        recall_at_k=r_at_k,
        precision_at_k=p_at_k,
        reciprocal_rank=rr,
        hallucination_count=len(h_urls),
        total_recommendations=len(recommended_items),
        turns_to_first_recommendation=turns,
        recommended_names=recommended_names,
        expected_names=expected_names,
        hallucinated_urls=h_urls,
    )

def aggregate_results(trace_results: list[TraceResult]) -> EvalSummary:
    """
    Aggregate per-trace results into overall eval summary.
    """
    if not trace_results:
        return EvalSummary(
            total_traces=0,
            mean_recall_at_k=0.0,
            mean_precision_at_k=0.0,
            mean_reciprocal_rank=0.0,
            hallucination_rate=0.0,
            mean_turns_to_recommendation=0.0,
            trace_results=[],
        )
    n = len(trace_results)
    total_recs = sum(r.total_recommendations for r in trace_results)
    total_halluc = sum(r.hallucination_count for r in trace_results)
    return EvalSummary(
        total_traces=n,
        mean_recall_at_k=sum(r.recall_at_k for r in trace_results) / n,
        mean_precision_at_k=sum(r.precision_at_k for r in trace_results) / n,
        mean_reciprocal_rank=sum(r.reciprocal_rank for r in trace_results) / n,
        hallucination_rate=total_halluc / total_recs if total_recs > 0 else 0.0,
        mean_turns_to_recommendation=sum(
            r.turns_to_first_recommendation for r in trace_results
        ) / n,
        trace_results=trace_results,
    )

# ── Helpers ───────────────────────────────────────────────────────────────────
def _normalize(name: str) -> str:
    """Normalize assessment name for fuzzy matching."""
    return name.lower().strip().replace("-", " ").replace("_", " ")

def load_catalog_urls(catalog_path: str = "catalog/catalog.json") -> set[str]:
    """Load all valid URLs from catalog.json for hallucination checking."""
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"catalog.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    return {item["url"] for item in catalog if item.get("url")}

def save_eval_results(summary: EvalSummary, output_path: str = "eval/results.json") -> None:
    """Save evaluation results to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)
    print(f"Eval results saved → {path}")
