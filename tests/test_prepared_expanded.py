from __future__ import annotations

from pathlib import Path

from hyde.loogle.prepared_expanded import BUNDLE_SPECS, load_prepared_expanded_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(dataset_name: str):
    spec = BUNDLE_SPECS[dataset_name]
    return load_prepared_expanded_bundle(
        dataset_variant=dataset_name,
        split=str(spec["split"]),
        config_path=PROJECT_ROOT / "configs" / f"{dataset_name}_hyde.yaml",
        data_dir=f"../data/{dataset_name}",
    )


def test_qasper_64k_matches_main_saadi_population() -> None:
    documents, qa_entries, metadata = _load("qasper_64k")
    assert len(documents) == 23
    assert len(qa_entries) == 1372
    assert sum(bool(row["retrieval_spans"]) for row in qa_entries) == 1353
    assert metadata["target_context_tokens"] == 64_000
    assert all(row["document_id"] in documents for row in qa_entries)


def test_musique_32k_matches_main_saadi_population() -> None:
    documents, qa_entries, metadata = _load("musique_32k")
    assert len(documents) == 45
    assert len(qa_entries) == 900
    assert all(row["retrieval_spans"] for row in qa_entries)
    assert metadata["target_context_tokens"] == 32_000
    assert all(row["document_id"] in documents for row in qa_entries)
