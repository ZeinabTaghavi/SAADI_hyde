from __future__ import annotations

import json

import numpy as np
import pytest

from hyde.generator import TransformersGenerator
from hyde.loogle.cache import HypothesisCache
from hyde.loogle.contriever import combine_hyde_embeddings, rank_embeddings
from hyde.loogle.evaluation import result_to_query_metrics, summarize_metrics
from hyde.loogle.types import RetrievalResult


def test_hyde_vector_averages_query_and_all_hypotheses():
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    assert np.allclose(combine_hyde_embeddings(embeddings), [2.0 / 3.0, 2.0 / 3.0])


def test_embedding_ranking_is_stable_and_returns_scores():
    indices, scores = rank_embeddings(
        np.asarray([1.0, 0.0]),
        np.asarray([[0.5, 0.0], [0.9, 0.0], [0.9, 0.0]], dtype=np.float32),
        k=3,
    )
    assert indices == [1, 2, 0]
    assert np.allclose(scores, [0.9, 0.9, 0.5])


def test_metrics_cover_gold_silver_and_strict_union():
    result = RetrievalResult(
        query_id="q",
        doc_id="d",
        question="?",
        retrieved_chunk_ids=["c2", "c1"],
        scores=[0.9, 0.8],
        gold_chunk_ids=["c1"],
        silver_chunk_ids=["c2", "c3"],
        silver_chunk_groups=[["c2", "c3"]],
    )
    row = result_to_query_metrics(result, k_values=[1, 2])
    assert row["gold_hit@1"] == 0.0
    assert row["gold_hit@2"] == 1.0
    assert row["silver_loose_recall@1"] == 0.5
    assert row["silver_strict_hit@2"] == 0.0
    assert row["strict_union_hit@2"] == 1.0
    summary, leaderboard = summarize_metrics(
        [row], dataset_name="loogle", split="test", run_name="fixture", k_values=[1, 2], average_chunk_tokens=5
    )
    assert summary["eligible_queries_by_view"]["gold"] == 1
    assert leaderboard["gold_mrr@2"] == 0.5


def test_cache_recovers_truncated_final_line_and_remains_appendable(tmp_path):
    path = tmp_path / "hypotheses.jsonl"
    valid = {"query_id": "q1", "hypothetical_documents": ["one"]}
    path.write_text(json.dumps(valid) + "\n" + '{"query_id":', encoding="utf-8")
    cache = HypothesisCache(path)
    assert cache.get("q1", expected_count=1) == valid
    cache.append({"query_id": "q2", "hypothetical_documents": ["two"]})
    reloaded = HypothesisCache(path)
    assert reloaded.get("q2", expected_count=1)["hypothetical_documents"] == ["two"]


def test_transformers_generator_caps_empty_output_attempts():
    generator = object.__new__(TransformersGenerator)
    generator.n = 2
    generator.max_attempts = 3
    generator.model_name = "fake/qwen"
    generator._generate_once = lambda _prompt: ""
    with pytest.raises(RuntimeError, match="0/2"):
        generator.generate("prompt")
