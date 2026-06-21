"""HippoRAG-compatible retrieval metrics and summaries."""

from __future__ import annotations

import math
from typing import Any, Iterable

from .types import RetrievalResult

METHOD_NAME = "hyde"
RANKING_VIEWS = ("gold", "silver_loose", "union_loose")
HIT_VIEWS = ("gold_hit", "silver_strict_hit", "strict_union_hit")


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 12)


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return _round(sum(materialized) / len(materialized)) if materialized else None


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return next((1.0 / rank for rank, chunk_id in enumerate(retrieved[:k], 1) if chunk_id in relevant), 0.0)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1.0) for rank, chunk_id in enumerate(retrieved[:k], 1) if chunk_id in relevant)
    ideal = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def _strict_group_hit(groups: list[list[str]], retrieved: list[str], k: int) -> bool:
    top = set(retrieved[:k])
    return any(set(group).issubset(top) for group in groups if group)


def result_to_query_metrics(result: RetrievalResult, *, k_values: list[int]) -> dict[str, Any]:
    retrieved = _dedupe(result.retrieved_chunk_ids)
    gold = set(result.gold_chunk_ids)
    silver = set(result.silver_chunk_ids)
    groups = [list(group) for group in result.silver_chunk_groups if group]
    relevance = {"gold": gold, "silver_loose": silver, "union_loose": gold | silver}
    row: dict[str, Any] = {"query_id": result.query_id, "doc_id": result.doc_id, "question": result.question}
    for k in k_values:
        for view, relevant in relevance.items():
            for metric, function in (
                ("recall", recall_at_k),
                ("mrr", reciprocal_rank_at_k),
                ("ndcg", ndcg_at_k),
            ):
                row[f"{view}_{metric}@{k}"] = _round(function(retrieved, relevant, k)) if relevant else None
        row[f"gold_hit@{k}"] = float(bool(set(retrieved[:k]) & gold)) if gold else None
        row[f"silver_strict_hit@{k}"] = float(_strict_group_hit(groups, retrieved, k)) if groups else None
        row[f"strict_union_hit@{k}"] = (
            float(bool(set(retrieved[:k]) & gold) or _strict_group_hit(groups, retrieved, k))
            if gold or groups
            else None
        )
    row.update(
        {
            "retrieved_ids_top10": retrieved[:10],
            "retrieved_scores_top10": [_round(value) for value in result.scores[:10]],
            "gold_relevant_ids": sorted(gold),
            "silver_loose_relevant_ids": sorted(silver),
            "union_loose_relevant_ids": sorted(gold | silver),
            "silver_strict_groups": groups,
        }
    )
    return row


def summarize_metrics(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    split: str,
    run_name: str,
    k_values: list[int],
    average_chunk_tokens: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ranking: dict[str, dict[str, float | None]] = {view: {} for view in RANKING_VIEWS}
    hits: dict[str, dict[str, float | None]] = {view: {} for view in HIT_VIEWS}
    eligible: dict[str, int] = {}
    first_k = k_values[0]
    for view in RANKING_VIEWS:
        eligible[view] = sum(row.get(f"{view}_recall@{first_k}") is not None for row in rows)
        for k in k_values:
            for metric in ("recall", "mrr", "ndcg"):
                key = f"{view}_{metric}@{k}"
                ranking[view][f"{metric}@{k}"] = _mean(float(row[key]) for row in rows if row.get(key) is not None)
    for view in HIT_VIEWS:
        eligible[view] = sum(row.get(f"{view}@{first_k}") is not None for row in rows)
        for k in k_values:
            key = f"{view}@{k}"
            hits[view][f"hit_rate@{k}"] = _mean(float(row[key]) for row in rows if row.get(key) is not None)
    common = {
        "method_name": METHOD_NAME,
        "dataset_name": dataset_name,
        "split": split,
        "run_name": run_name,
        "average_chunk_tokens": _round(average_chunk_tokens),
        "n_queries": len(rows),
    }
    summary = {
        **common,
        "k_values": k_values,
        "ranking_metrics_by_view": ranking,
        "hit_rate_metrics_by_view": hits,
        "eligible_queries_by_view": eligible,
    }
    leaderboard = dict(common)
    for k in k_values:
        for view in RANKING_VIEWS:
            for metric in ("recall", "mrr", "ndcg"):
                leaderboard[f"{view}_{metric}@{k}"] = ranking[view][f"{metric}@{k}"]
        leaderboard[f"gold_hit@{k}"] = hits["gold_hit"][f"hit_rate@{k}"]
        leaderboard[f"silver_strict_hit@{k}"] = hits["silver_strict_hit"][f"hit_rate@{k}"]
        leaderboard[f"strict_union_hit@{k}"] = hits["strict_union_hit"][f"hit_rate@{k}"]
    return summary, leaderboard
