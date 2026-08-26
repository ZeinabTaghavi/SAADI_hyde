#!/usr/bin/env python3
"""Validate that a completed HyDE artifact matches the requested matrix cell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hyde.loogle.chunking import DEFAULT_CHUNK_SIZE  # noqa: E402
from hyde.loogle.runner import _generation_settings, load_config  # noqa: E402
from hyde.loogle.retrievers import normalize_retriever_name, retriever_spec  # noqa: E402


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def validate_artifact(
    *,
    manifest_path: Path,
    config_path: Path,
    dataset: str,
    retriever: str,
    top_k: int,
) -> list[str]:
    errors: list[str] = []
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]
    leaderboard_path = manifest_path.parent / "leaderboard_row.json"
    if not leaderboard_path.is_file():
        return [f"missing leaderboard: {leaderboard_path}"]

    manifest = _read_object(manifest_path)
    leaderboard = _read_object(leaderboard_path)
    context = dict(manifest.get("run_metadata", {}) or {})
    preparation = dict(context.get("preparation", {}) or {})
    config = load_config(config_path)
    chunking = dict(config.get("chunking", {}) or {})
    expected_chunk_size = int(chunking.get("chunk_size", DEFAULT_CHUNK_SIZE))
    expected_chunk_overlap = int(chunking.get("chunk_overlap", 0))
    requested_retriever = normalize_retriever_name(retriever)
    spec = retriever_spec(requested_retriever, config)

    actual_dataset = str(context.get("dataset_name") or leaderboard.get("dataset_name") or "").lower()
    if actual_dataset != dataset.lower():
        errors.append(f"dataset={actual_dataset!r}, expected {dataset!r}")

    actual_retriever = str(
        context.get("retriever_name") or leaderboard.get("retriever_name") or "contriever"
    ).lower()
    if normalize_retriever_name(actual_retriever) != requested_retriever:
        errors.append(f"retriever={actual_retriever!r}, expected {requested_retriever!r}")

    if int(preparation.get("chunk_size", -1)) != expected_chunk_size:
        errors.append(
            f"chunk_size={preparation.get('chunk_size')!r}, expected {expected_chunk_size}"
        )
    if int(preparation.get("chunk_overlap", -1)) != expected_chunk_overlap:
        errors.append(
            f"chunk_overlap={preparation.get('chunk_overlap')!r}, "
            f"expected {expected_chunk_overlap}"
        )
    if int(context.get("top_k", -1)) != int(top_k):
        errors.append(f"top_k={context.get('top_k')!r}, expected {top_k}")
    if str(context.get("retrieval_scope") or "") != "per_document":
        errors.append(f"retrieval_scope={context.get('retrieval_scope')!r}, expected 'per_document'")

    expected_generation = _generation_settings(config)
    if context.get("generation") != expected_generation:
        errors.append("HyDE generation settings do not match the current config")

    if spec.kind == "dense":
        if str(context.get("embedding_model_name") or "") != str(spec.model_name):
            errors.append(
                f"embedding_model={context.get('embedding_model_name')!r}, expected {spec.model_name!r}"
            )
        retriever_metadata = dict(context.get("retriever", {}) or {})
        if retriever_metadata:
            for field in ("pooling", "normalize_embeddings", "max_length"):
                if retriever_metadata.get(field) != getattr(spec, field):
                    errors.append(
                        f"{field}={retriever_metadata.get(field)!r}, expected {getattr(spec, field)!r}"
                    )
    else:
        retriever_metadata = dict(context.get("retriever", {}) or {})
        if retriever_metadata.get("score_fusion") != spec.score_fusion:
            errors.append(
                f"score_fusion={retriever_metadata.get('score_fusion')!r}, expected {spec.score_fusion!r}"
            )

    expected_population = dict(preparation.get("expected_population", {}) or {})
    actual_population = dict(preparation.get("actual_population", {}) or {})
    for field in ("documents", "chunks", "retrieval_examples"):
        if field in expected_population and actual_population.get(field) != expected_population.get(field):
            errors.append(
                f"population {field}={actual_population.get(field)!r}, "
                f"expected {expected_population.get(field)!r}"
            )
    expected_queries = actual_population.get("retrieval_examples")
    if expected_queries is not None and leaderboard.get("n_queries") != expected_queries:
        errors.append(
            f"n_queries={leaderboard.get('n_queries')!r}, expected {expected_queries!r}"
        )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_artifact(
        manifest_path=args.manifest.expanduser().resolve(),
        config_path=args.config.expanduser().resolve(),
        dataset=args.dataset,
        retriever=args.retriever,
        top_k=args.top_k,
    )
    if errors:
        print(json.dumps({"matches": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"matches": True, "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
