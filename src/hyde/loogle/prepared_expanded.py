"""Checksum-validated loaders for the frozen main-SAADI expansions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_FILES = (
    "context_expansion_metadata.json",
    "documents.json",
    "group_membership.json",
    "qa.json",
)

BUNDLE_SPECS: dict[str, dict[str, Any]] = {
    "qasper_64k": {
        "dataset": "qasper",
        "split": "test",
        "target_context_tokens": 64_000,
        "documents": 23,
        "queries": 1_372,
        "retrieval_examples": 1_333,
        "queries_with_retrieval_spans": 1_353,
        "manifest_sha256": "a90998ad59ccace0e7369a42a49ec72a26035b1bb00fac252343f8ec2129cfd7",
    },
    "musique_32k": {
        "dataset": "musique",
        "split": "validation",
        "target_context_tokens": 32_000,
        "documents": 45,
        "queries": 900,
        "retrieval_examples": 900,
        "queries_with_retrieval_spans": 900,
        "manifest_sha256": "8f205e9f3a6ff06e46eae46871cce7108e03904161256a9bef28d85b3a9dc9fe",
    },
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Prepared dataset file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Prepared dataset file is invalid JSON: {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _resolve_data_dir(config_path: Path, data_dir: str | Path | None, dataset_variant: str) -> Path:
    raw = Path(data_dir) if data_dir else PROJECT_ROOT / "data" / dataset_variant
    raw = raw.expanduser()
    if not raw.is_absolute():
        raw = config_path.parent / raw
    return raw.resolve()


def _validate_bundle(
    dataset_variant: str,
    data_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    spec = BUNDLE_SPECS[dataset_variant]
    manifest_path = data_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or int(manifest.get("format_version", -1)) != 1:
        raise RuntimeError(f"Unsupported prepared dataset manifest: {manifest_path}")
    actual_manifest_sha256 = _sha256(manifest_path)
    if actual_manifest_sha256 != spec["manifest_sha256"]:
        raise RuntimeError(
            f"Prepared {dataset_variant} manifest checksum mismatch: "
            f"expected {spec['manifest_sha256']}, got {actual_manifest_sha256}"
        )

    file_specs = manifest.get("files")
    if not isinstance(file_specs, dict):
        raise RuntimeError(f"Prepared dataset manifest has no file checksums: {manifest_path}")
    for filename in REQUIRED_FILES:
        expected = (file_specs.get(filename) or {}).get("sha256")
        if not isinstance(expected, str) or not expected:
            raise RuntimeError(f"Manifest is missing a checksum for {filename}")
        actual = _sha256(data_dir / filename)
        if actual != expected:
            raise RuntimeError(
                f"Prepared dataset checksum mismatch for {filename}: expected {expected}, got {actual}"
            )

    metadata = _read_json(data_dir / "context_expansion_metadata.json")
    if not isinstance(metadata, dict):
        raise RuntimeError("context_expansion_metadata.json must contain an object")
    expected_values = {
        "dataset": spec["dataset"],
        "split": spec["split"],
        "target_context_tokens": spec["target_context_tokens"],
        "prepared_documents": spec["documents"],
        "qa_entries": spec["queries"],
    }
    for field, expected in expected_values.items():
        if metadata.get(field) != expected:
            raise RuntimeError(
                f"Prepared {dataset_variant} {field} mismatch: "
                f"expected {expected!r}, got {metadata.get(field)!r}"
            )
    if int(metadata.get("source_matchable_spans_lost", -1)) != 0:
        raise RuntimeError(f"Prepared {dataset_variant} reports lost source-matchable evidence spans")
    if manifest.get("dataset_variant") != dataset_variant:
        raise RuntimeError(
            f"Prepared dataset variant mismatch: expected {dataset_variant!r}, "
            f"got {manifest.get('dataset_variant')!r}"
        )
    return manifest, metadata, manifest_path


def load_prepared_expanded_bundle(
    *,
    dataset_variant: str,
    split: str,
    config_path: Path,
    data_dir: str | Path | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """Load one exact SAADI bundle into the standalone HyDE schema."""

    if dataset_variant not in BUNDLE_SPECS:
        raise ValueError(f"Unsupported prepared dataset variant: {dataset_variant!r}")
    spec = BUNDLE_SPECS[dataset_variant]
    if str(split).strip().lower() != spec["split"]:
        raise ValueError(
            f"The {dataset_variant} bundle contains split={spec['split']!r}, got {split!r}"
        )

    root = _resolve_data_dir(config_path, data_dir, dataset_variant)
    manifest, expansion_metadata, manifest_path = _validate_bundle(dataset_variant, root)
    raw_documents = _read_json(root / "documents.json")
    raw_membership = _read_json(root / "group_membership.json")
    raw_qa_entries = _read_json(root / "qa.json")
    if not isinstance(raw_documents, dict) or len(raw_documents) != spec["documents"]:
        actual = len(raw_documents) if isinstance(raw_documents, dict) else "invalid"
        raise RuntimeError(f"{dataset_variant} has {actual} prepared documents; expected {spec['documents']}")
    if not isinstance(raw_membership, dict) or set(raw_membership) != set(raw_documents):
        raise RuntimeError(f"{dataset_variant} group membership does not match its document IDs")
    if not isinstance(raw_qa_entries, list) or len(raw_qa_entries) != spec["queries"]:
        actual = len(raw_qa_entries) if isinstance(raw_qa_entries, list) else "invalid"
        raise RuntimeError(f"{dataset_variant} has {actual} prepared queries; expected {spec['queries']}")

    membership = {
        str(group_id): {str(source_id) for source_id in source_ids}
        for group_id, source_ids in raw_membership.items()
    }
    qa_entries: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    for source_index, row in enumerate(raw_qa_entries):
        if not isinstance(row, dict):
            raise RuntimeError(f"{dataset_variant} QA row {source_index} is not an object")
        doc_id = str(row.get("document_id", "")).strip()
        source_doc_id = str(row.get("source_document_id", "")).strip()
        if doc_id not in raw_documents:
            raise RuntimeError(f"{dataset_variant} QA row {source_index} references missing {doc_id!r}")
        if source_doc_id not in membership.get(doc_id, set()):
            raise RuntimeError(
                f"{dataset_variant} QA row {source_index} source {source_doc_id!r} "
                f"is not a member of {doc_id!r}"
            )
        query_id = f"{dataset_variant}::{row.get('id', row.get('query_id', source_index))}"
        if query_id in seen_query_ids:
            raise RuntimeError(f"Duplicate prepared query ID: {query_id}")
        seen_query_ids.add(query_id)
        qa_entries.append(
            {
                "id": query_id,
                "question": str(row.get("question", "")).strip(),
                "document_id": doc_id,
                "answers": _text_list(row.get("answers")),
                "retrieval_spans": _text_list(row.get("retrieval_spans")),
                "source_document_id": source_doc_id,
                "dataset_variant": dataset_variant,
            }
        )

    queries_with_spans = sum(bool(row["retrieval_spans"]) for row in qa_entries)
    if queries_with_spans != spec["queries_with_retrieval_spans"]:
        raise RuntimeError(
            f"{dataset_variant} has {queries_with_spans} queries with retrieval spans; "
            f"expected {spec['queries_with_retrieval_spans']}"
        )
    documents = {str(doc_id): str(text) for doc_id, text in raw_documents.items()}
    return documents, qa_entries, {
        "dataset_source": "prepared_main_saadi_expansion",
        "dataset_name": dataset_variant,
        "source_dataset": spec["dataset"],
        "split": spec["split"],
        "target_context_tokens": spec["target_context_tokens"],
        "documents": len(documents),
        "qa_entries": len(qa_entries),
        "queries_with_retrieval_spans": queries_with_spans,
        "prepared_root": str(root),
        "prepared_manifest": str(manifest_path),
        "prepared_manifest_sha256": spec["manifest_sha256"],
        "bundle_manifest": manifest,
        "context_expansion_metadata": expansion_metadata,
    }


def select_prepared_subset(
    documents: dict[str, str],
    qa_entries: list[dict[str, Any]],
    *,
    max_documents: int | None,
    max_qa_entries: int | None,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """Apply deterministic smoke-test limits without changing full-run ordering."""

    doc_ids = list(documents)
    if max_documents is not None:
        if max_documents <= 0:
            raise ValueError("max_documents must be positive")
        doc_ids = doc_ids[:max_documents]
    selected_ids = set(doc_ids)
    selected_qa = [row for row in qa_entries if str(row.get("document_id")) in selected_ids]
    if max_qa_entries is not None:
        if max_qa_entries <= 0:
            raise ValueError("max_qa_entries must be positive")
        selected_qa = selected_qa[:max_qa_entries]
    return (
        {doc_id: documents[doc_id] for doc_id in doc_ids},
        selected_qa,
        {
            "selected_document_ids": doc_ids,
            "documents_selected": len(doc_ids),
            "qa_entries_selected": len(selected_qa),
            "limited": max_documents is not None or max_qa_entries is not None,
        },
    )
