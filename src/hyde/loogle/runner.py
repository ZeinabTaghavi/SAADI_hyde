"""End-to-end standalone HyDE retrieval runner for the frozen LooGLE subset."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np

from hyde.generator import TransformersGenerator
from hyde.promptor import Promptor

from .cache import HypothesisCache
from .chunking import chunk_documents_grouped_records
from .contriever import (
    ContrieverEncoder,
    combine_hyde_embeddings,
    embedding_cache_path,
    load_or_encode_document,
    rank_embeddings,
)
from .dataset import load_loogle_bundle, load_subset_manifest, select_frozen_subset
from .evaluation import HIT_VIEWS, METHOD_NAME, RANKING_VIEWS, result_to_query_metrics, summarize_metrics
from .io import write_json, write_jsonl
from .labeling import build_retrieval_examples
from .types import ChunkRecord, RetrievalExample, RetrievalResult

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Experiment configuration requires PyYAML.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    return payload


def parse_top_ks(values: list[str] | tuple[str, ...] | list[int]) -> list[int]:
    parsed: set[int] = set()
    for raw in values:
        for part in re.split(r"[,\s]+", str(raw)):
            if part:
                value = int(part)
                if value <= 0:
                    raise ValueError("top-k values must be positive")
                parsed.add(value)
    if not parsed:
        raise ValueError("At least one top-k value is required")
    return sorted(parsed)


def _package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except Exception:
        return None


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def _average_chunk_tokens(chunks: list[ChunkRecord]) -> float | None:
    return sum(len(chunk.raw_text.split()) for chunk in chunks) / len(chunks) if chunks else None


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def prepare_data(
    config: dict[str, Any],
    *,
    config_path: Path,
    max_documents: int | None,
    max_qa_entries: int | None,
) -> tuple[list[ChunkRecord], dict[str, list[ChunkRecord]], list[RetrievalExample], dict[str, Any]]:
    dataset_cfg = dict(config.get("dataset", {}) or {})
    documents, qa_entries, dataset_metadata = load_loogle_bundle(
        split=str(dataset_cfg.get("split", "test")),
        config_name=str(dataset_cfg.get("config_name", "shortdep_qa")),
    )
    manifest_value = dataset_cfg.get("subset_manifest", "loogle_hipporag_subset.json")
    manifest_path = Path(str(manifest_value)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (config_path.parent / manifest_path).resolve()
    manifest = load_subset_manifest(manifest_path)
    documents, qa_entries, selection_metadata = select_frozen_subset(
        documents,
        qa_entries,
        manifest,
        max_documents=max_documents,
        max_qa_entries=max_qa_entries,
    )

    chunk_cfg = dict(config.get("chunking", {}) or {})
    chunk_size = int(chunk_cfg.get("chunk_size", 500))
    chunk_overlap = int(chunk_cfg.get("chunk_overlap", 0))
    doc_ids = list(documents)
    grouped = chunk_documents_grouped_records(
        [documents[doc_id] for doc_id in doc_ids],
        doc_ids=doc_ids,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = [chunk for doc_chunks in grouped for chunk in doc_chunks]
    chunks_by_doc = {doc_id: doc_chunks for doc_id, doc_chunks in zip(doc_ids, grouped)}
    examples = build_retrieval_examples(qa_entries, chunks_by_doc)
    if not examples:
        raise RuntimeError("No labeled LooGLE retrieval examples were built")

    limited = bool(selection_metadata["limited"])
    expected = dict(manifest.get("expected", {}) or {})
    actual = {
        "documents": len(documents),
        "chunks": len(chunks),
        "retrieval_examples": len(examples),
        "average_chunk_tokens": _average_chunk_tokens(chunks),
    }
    if not limited:
        for field in ("documents", "chunks", "retrieval_examples"):
            if field in expected and int(expected[field]) != int(actual[field]):
                raise RuntimeError(
                    f"Frozen LooGLE population mismatch for {field}: expected {expected[field]}, got {actual[field]}. "
                    "Check the dataset revision and standalone chunk/label implementation."
                )
        if "average_chunk_tokens" in expected:
            difference = abs(float(expected["average_chunk_tokens"]) - float(actual["average_chunk_tokens"] or 0.0))
            if difference > 1e-9:
                raise RuntimeError(
                    "Frozen LooGLE average chunk size mismatch: "
                    f"expected {expected['average_chunk_tokens']}, got {actual['average_chunk_tokens']}"
                )
    return chunks, chunks_by_doc, examples, {
        "dataset": dataset_metadata,
        "subset_manifest": str(manifest_path),
        "selection": selection_metadata,
        "expected_population": expected,
        "actual_population": actual,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def _generation_settings(config: dict[str, Any]) -> dict[str, Any]:
    hyde_cfg = dict(config.get("hyde", {}) or {})
    generator_cfg = dict(hyde_cfg.get("generator", {}) or {})
    return {
        "prompt_task": str(hyde_cfg.get("prompt_task", "web search")),
        "model_name": str(generator_cfg.get("model_name", "Qwen/Qwen3-30B-A3B-Instruct-2507")),
        "n": int(generator_cfg.get("n", 8)),
        "max_new_tokens": int(generator_cfg.get("max_new_tokens", 512)),
        "temperature": float(generator_cfg.get("temperature", 0.7)),
        "top_p": float(generator_cfg.get("top_p", 0.8)),
        "stop": list(generator_cfg.get("stop", ["\n\n\n"])),
        "device_map": generator_cfg.get("device_map", "auto"),
        "torch_dtype": generator_cfg.get("torch_dtype", "auto"),
        "trust_remote_code": bool(generator_cfg.get("trust_remote_code", True)),
        "attempts_per_hypothesis": int(generator_cfg.get("attempts_per_hypothesis", 3)),
    }


def _make_generator(settings: dict[str, Any]) -> TransformersGenerator:
    return TransformersGenerator(
        settings["model_name"],
        api_key=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        n=settings["n"],
        max_new_tokens=settings["max_new_tokens"],
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        stop=settings["stop"],
        device_map=settings["device_map"],
        torch_dtype=settings["torch_dtype"],
        cache_dir=os.getenv("HF_HUB_CACHE"),
        trust_remote_code=settings["trust_remote_code"],
        local_files_only=None,
        max_attempts=settings["n"] * settings["attempts_per_hypothesis"],
    )


def _generate_hypotheses(generator: Any, prompt: str, *, retries: int, expected_count: int) -> list[str]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            output = [str(value).strip() for value in generator.generate(prompt)]
            if len(output) == expected_count and all(output):
                return output
            last_error = RuntimeError(f"Generator returned {len(output)} non-validated hypotheses; expected {expected_count}")
        except Exception as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(attempt, 5))
    raise RuntimeError(f"HyDE generation failed after {retries} attempts; the completed cache remains resumable") from last_error


def _truncate_result(result: RetrievalResult, top_k: int) -> RetrievalResult:
    return RetrievalResult(
        query_id=result.query_id,
        doc_id=result.doc_id,
        question=result.question,
        retrieved_chunk_ids=result.retrieved_chunk_ids[:top_k],
        retrieved_indices=result.retrieved_indices[:top_k],
        scores=result.scores[:top_k],
        gold_chunk_ids=list(result.gold_chunk_ids),
        silver_chunk_ids=list(result.silver_chunk_ids),
        silver_chunk_groups=[list(group) for group in result.silver_chunk_groups],
    )


def _write_run_artifacts(
    run_dir: Path,
    *,
    chunks: list[ChunkRecord],
    examples: list[RetrievalExample],
    results: list[RetrievalResult],
    payload_rows: list[dict[str, Any]],
    dataset_name: str,
    split: str,
    run_name: str,
    k_values: list[int],
    context: dict[str, Any],
    config_path: Path,
    command_used: str,
    repository_root: Path,
) -> None:
    index_dir = run_dir / "index"
    retrieval_dir = run_dir / "retrieval"
    chunk_index_path = write_jsonl(
        index_dir / "chunk_index.jsonl",
        (
            {
                **chunk.to_dict(),
                "token_count": len(chunk.raw_text.split()),
            }
            for chunk in chunks
        ),
    )
    average_tokens = _average_chunk_tokens(chunks)
    index_stats_path = write_json(
        index_dir / "index_stats.json",
        {"method_name": METHOD_NAME, "n_chunks": len(chunks), "average_chunk_tokens": average_tokens},
    )
    results_path = write_json(retrieval_dir / "retrieval_results.json", [result.to_dict() for result in results])
    examples_path = write_jsonl(retrieval_dir / "retrieval_examples.jsonl", (example.to_dict() for example in examples))
    payloads_path = write_jsonl(retrieval_dir / "retrieval_payloads.jsonl", payload_rows)

    per_query = [result_to_query_metrics(result, k_values=k_values) for result in results]
    summary, leaderboard = summarize_metrics(
        per_query,
        dataset_name=dataset_name,
        split=split,
        run_name=run_name,
        k_values=k_values,
        average_chunk_tokens=average_tokens,
    )
    summary_path = write_json(run_dir / "metrics_summary.json", summary)
    per_query_path = write_jsonl(run_dir / "metrics_per_query.jsonl", per_query)
    leaderboard_path = write_json(run_dir / "leaderboard_row.json", leaderboard)
    manifest_path = run_dir / "evaluation_manifest.json"
    write_json(
        manifest_path,
        {
            "input_files_used": {
                "config": str(config_path),
                "subset_manifest": context["preparation"]["subset_manifest"],
                "chunk_index_jsonl": str(chunk_index_path),
                "index_stats_json": str(index_stats_path),
                "retrieval_results_json": str(results_path),
                "retrieval_examples_jsonl": str(examples_path),
                "retrieval_payloads_jsonl": str(payloads_path),
                "hypothesis_cache_jsonl": context["hypothesis_cache"],
            },
            "output_files_written": {
                "metrics_summary_json": str(summary_path),
                "metrics_per_query_jsonl": str(per_query_path),
                "leaderboard_row_json": str(leaderboard_path),
                "evaluation_manifest_json": str(manifest_path),
            },
            "relevance_source_used": {
                "labels_file": str(examples_path),
                "fields": ["gold_chunk_ids", "silver_chunk_ids", "silver_chunk_groups"],
                "label_rows_loaded": len(results),
            },
            "metric_schema": {
                "ranking_views": list(RANKING_VIEWS),
                "ranking_metrics": ["recall", "mrr", "ndcg"],
                "hit_rate_views": list(HIT_VIEWS),
                "hit_rate_metrics": ["hit_rate"],
            },
            "assumptions": [
                "Retrieval is restricted to chunks from the query's source LooGLE document.",
                "The HyDE vector is the arithmetic mean of the normalized Contriever embeddings for the question and eight hypothetical documents.",
                "Retrieved chunk IDs are deduplicated while preserving rank before metric calculation.",
                "Silver-S Hit@K requires a complete silver_chunk_group; Union-S is Gold Hit@K OR Silver-S Hit@K.",
            ],
            "missing_metrics": [],
            "command_used": command_used,
            "package_versions": {
                "python": sys.version,
                "platform": platform.platform(),
                "git_commit": _git_commit(repository_root),
                "datasets": _package_version("datasets"),
                "numpy": _package_version("numpy"),
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
                "huggingface_hub": _package_version("huggingface_hub"),
            },
            "run_metadata": context,
            "counts": {"queries_evaluated": len(results), "eligible_queries_by_view": summary["eligible_queries_by_view"]},
        },
    )


def run_experiment(
    *,
    config_path: str | Path,
    output_root: str | Path,
    work_root: str | Path,
    top_ks: list[int],
    run_name: str | None = None,
    max_documents: int | None = None,
    max_qa_entries: int | None = None,
    embedding_device: str | None = None,
    resume: bool = True,
    force: bool = False,
    force_embeddings: bool = False,
    validate_only: bool = False,
    generator_override: Any | None = None,
    encoder_override: Any | None = None,
) -> list[Path]:
    started = time.perf_counter()
    config_path = Path(config_path).expanduser().resolve()
    repository_root = config_path.parent.parent
    config = load_config(config_path)
    top_ks = sorted(set(int(value) for value in top_ks))
    if not top_ks or min(top_ks) <= 0:
        raise ValueError("top_ks must contain positive integers")
    limited = max_documents is not None or max_qa_entries is not None
    if run_name is None:
        run_name = "loogle_retrieval_ablation_hyde"
        if limited:
            run_name += f"_smoke_d{max_documents or 'all'}_q{max_qa_entries or 'all'}"
    output_root = Path(output_root).expanduser().resolve()
    work_root = Path(work_root).expanduser().resolve()
    output_dirs = [output_root / "loogle" / METHOD_NAME / f"top_{top_k}" / run_name for top_k in top_ks]
    if not force and any((path / "leaderboard_row.json").exists() for path in output_dirs):
        raise FileExistsError(f"Completed output already exists for run={run_name}; pass --force to overwrite artifacts")

    chunks, chunks_by_doc, examples, preparation = prepare_data(
        config,
        config_path=config_path,
        max_documents=max_documents,
        max_qa_entries=max_qa_entries,
    )
    logger.info(
        "Prepared frozen LooGLE population documents=%d chunks=%d labeled_queries=%d",
        preparation["actual_population"]["documents"],
        len(chunks),
        len(examples),
    )
    if validate_only:
        print(json.dumps({"validated": True, **preparation["actual_population"]}, indent=2))
        return []

    generation = _generation_settings(config)
    seed = config.get("seed")
    seed = int(seed) if seed is not None else None
    _set_seed(seed)
    retrieval_cfg = dict(config.get("retrieval", {}) or {})
    encoder_cfg = dict(retrieval_cfg.get("encoder", {}) or {})
    encoder = encoder_override or ContrieverEncoder(
        str(encoder_cfg.get("model_name", "facebook/contriever")),
        device=embedding_device or encoder_cfg.get("device"),
        batch_size=int(encoder_cfg.get("batch_size", 128)),
        max_length=int(encoder_cfg.get("max_length", 512)),
        cache_dir=os.getenv("HF_HUB_CACHE"),
        local_files_only=None,
    )

    work_dir = work_root / "loogle" / METHOD_NAME / run_name
    hypothesis_cache = HypothesisCache(work_dir / "hypotheses.jsonl", resume=resume)
    embedding_root = work_dir / "document_embeddings"
    max_top_k = max(top_ks)
    document_embeddings: dict[str, np.ndarray] = {}
    embedding_cache_hits = 0
    for doc_id, doc_chunks in chunks_by_doc.items():
        embeddings, hit = load_or_encode_document(
            encoder,
            doc_chunks,
            embedding_cache_path(embedding_root, doc_id),
            force=force_embeddings,
        )
        document_embeddings[doc_id] = embeddings
        embedding_cache_hits += int(hit)

    promptor = Promptor(generation["prompt_task"])
    generator = generator_override
    model_snapshot: str | None = None
    results: list[RetrievalResult] = []
    payload_rows: list[dict[str, Any]] = []
    cached_queries = 0
    total_generation_seconds = 0.0
    total_retrieval_seconds = 0.0
    for query_number, example in enumerate(examples, 1):
        prompt = promptor.build_prompt(example.question)
        cached = hypothesis_cache.get(example.query_id, expected_count=generation["n"])
        cache_hit = bool(
            cached is not None
            and cached.get("question") == example.question
            and cached.get("model_name") == generation["model_name"]
            and cached.get("generation_settings") == generation
        )
        if cache_hit:
            hypotheses = list(cached["hypothetical_documents"])
            generation_seconds = 0.0
            cached_queries += 1
        else:
            if generator is None:
                generator = _make_generator(generation)
                if hasattr(generator, "prepare_snapshot"):
                    model_snapshot = str(generator.prepare_snapshot())
            generation_started = time.perf_counter()
            hypotheses = _generate_hypotheses(
                generator,
                prompt,
                retries=generation["attempts_per_hypothesis"],
                expected_count=generation["n"],
            )
            generation_seconds = time.perf_counter() - generation_started
            total_generation_seconds += generation_seconds
            hypothesis_cache.append(
                {
                    "query_id": example.query_id,
                    "doc_id": example.doc_id,
                    "question": example.question,
                    "prompt": prompt,
                    "model_name": generation["model_name"],
                    "generation_settings": generation,
                    "hypothetical_documents": hypotheses,
                    "generation_seconds": round(generation_seconds, 6),
                }
            )

        retrieval_started = time.perf_counter()
        component_embeddings = encoder.encode([example.question, *hypotheses])
        hyde_vector = combine_hyde_embeddings(component_embeddings)
        doc_chunks = chunks_by_doc[example.doc_id]
        indices, scores = rank_embeddings(hyde_vector, document_embeddings[example.doc_id], k=max_top_k)
        retrieved_ids = [doc_chunks[index].chunk_id for index in indices]
        retrieval_seconds = time.perf_counter() - retrieval_started
        total_retrieval_seconds += retrieval_seconds
        results.append(
            RetrievalResult(
                query_id=example.query_id,
                doc_id=example.doc_id,
                question=example.question,
                retrieved_chunk_ids=retrieved_ids,
                retrieved_indices=indices,
                scores=scores,
                gold_chunk_ids=list(example.gold_chunk_ids),
                silver_chunk_ids=list(example.silver_chunk_ids),
                silver_chunk_groups=[list(group) for group in example.silver_chunk_groups],
            )
        )
        payload_rows.append(
            {
                "query_id": example.query_id,
                "doc_id": example.doc_id,
                "question": example.question,
                "prompt": prompt,
                "hypothetical_documents": hypotheses,
                "hypothesis_count": len(hypotheses),
                "hypothesis_cache_hit": cache_hit,
                "generation_seconds": round(generation_seconds, 6),
                "retrieval_seconds": round(retrieval_seconds, 6),
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved_indices": indices,
                "retrieved_scores": scores,
                "retrieved_texts_top10": [doc_chunks[index].raw_text for index in indices[:10]],
            }
        )
        if query_number == 1 or query_number % 10 == 0 or query_number == len(examples):
            logger.info("HyDE progress queries=%d/%d cache_hits=%d", query_number, len(examples), cached_queries)

    command_used = " ".join([Path(sys.argv[0]).name, *sys.argv[1:]])
    common_context = {
        "method_name": METHOD_NAME,
        "dataset_name": "loogle",
        "split": str(config.get("dataset", {}).get("split", "test")),
        "run_name": run_name,
        "retrieval_scope": "per_document",
        "config": config,
        "preparation": preparation,
        "generation": generation,
        "seed": seed,
        "model_snapshot": model_snapshot,
        "embedding_model_name": encoder.model_name,
        "embedding_model_revision": getattr(getattr(getattr(encoder, "model", None), "config", None), "_commit_hash", None),
        "embedding_device": getattr(encoder, "device", embedding_device),
        "hypothesis_cache": str(hypothesis_cache.path),
        "document_embedding_cache_root": str(embedding_root),
        "hypothesis_cache_hits": cached_queries,
        "document_embedding_cache_hits": embedding_cache_hits,
        "generation_seconds": round(total_generation_seconds, 3),
        "embedding_and_retrieval_seconds": round(total_retrieval_seconds, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    written: list[Path] = []
    for top_k, run_dir in zip(top_ks, output_dirs):
        truncated_results = [_truncate_result(result, top_k) for result in results]
        truncated_payloads = []
        for row in payload_rows:
            truncated = dict(row)
            for key in ("retrieved_chunk_ids", "retrieved_indices", "retrieved_scores", "retrieved_texts_top10"):
                truncated[key] = list(truncated[key])[:top_k]
            truncated_payloads.append(truncated)
        k_values = [value for value in top_ks if value <= top_k]
        context = {**common_context, "top_k": top_k}
        _write_run_artifacts(
            run_dir,
            chunks=chunks,
            examples=examples,
            results=truncated_results,
            payload_rows=truncated_payloads,
            dataset_name="loogle",
            split=common_context["split"],
            run_name=run_name,
            k_values=k_values,
            context=context,
            config_path=config_path,
            command_used=command_used,
            repository_root=repository_root,
        )
        logger.info("Wrote HyDE evaluation artifacts: %s", run_dir)
        written.append(run_dir)
    return written
