from __future__ import annotations

import numpy as np

from hyde.loogle.retrievers import (
    BM25Index,
    RETRIEVER_SPECS,
    normalize_retriever_name,
    rank_score_vector,
    retriever_spec,
)


def test_registry_contains_requested_retrievers_without_e5():
    assert list(RETRIEVER_SPECS) == ["bm25", "contriever", "bge_m3", "qwen", "jina"]
    assert "e5" not in RETRIEVER_SPECS
    assert normalize_retriever_name("bge-m3") == "bge_m3"
    assert normalize_retriever_name("bm25s") == "bm25"


def test_retriever_config_overrides_are_standalone():
    config = {
        "retrieval": {
            "retriever_configs": {
                "qwen": {
                    "batch_size": 3,
                    "max_length": 1024,
                }
            }
        }
    }
    spec = retriever_spec("qwen", config)
    assert spec.model_name == "Qwen/Qwen3-Embedding-8B"
    assert spec.batch_size == 3
    assert spec.max_length == 1024


def test_bm25_hyde_fuses_static_query_variants():
    index = BM25Index(
        [
            "alpha beta appears in this passage",
            "gamma delta appears elsewhere",
            "unrelated material only",
        ]
    )
    scores = index.hyde_scores(
        ["Where is alpha?", "A hypothetical alpha beta passage."],
        fusion="minmax_mean",
    )
    indices, ranked_scores = rank_score_vector(scores, k=3)
    assert indices[0] == 0
    assert ranked_scores[0] > ranked_scores[1]
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
