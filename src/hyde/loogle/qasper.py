"""Standalone QASPER loader matching the SAADI retrieval schema."""

from __future__ import annotations

import logging
from importlib.metadata import version as package_version
from typing import Any, Iterable

from .dataset import coerce_to_text

logger = logging.getLogger(__name__)
DATASET_ID = "allenai/qasper"


def _datasets_major() -> int | None:
    try:
        return int(package_version("datasets").split(".", 1)[0])
    except Exception:
        return None


def _load_dataset(config_name: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("QASPER loading requires the 'datasets' package.") from exc
    major = _datasets_major()
    kwargs: dict[str, Any] = {}
    if major is not None and major >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    elif major is None or major < 4:
        kwargs["trust_remote_code"] = True
    resolved = config_name
    try:
        return load_dataset(DATASET_ID, name=resolved, **kwargs) if resolved else load_dataset(DATASET_ID, **kwargs)
    except TypeError:
        return load_dataset(DATASET_ID, name=resolved) if resolved else load_dataset(DATASET_ID)


def _document_text(row: dict[str, Any]) -> str:
    paragraphs = (row.get("full_text") or {}).get("paragraphs", "")
    if isinstance(paragraphs, list):
        if paragraphs and isinstance(paragraphs[0], list):
            return "\n".join(
                paragraph
                for section in paragraphs
                for paragraph in section
                if isinstance(paragraph, str)
            )
        return "\n".join(paragraph for paragraph in paragraphs if isinstance(paragraph, str))
    return coerce_to_text(paragraphs)


def parse_qasper_rows(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    documents: dict[str, str] = {}
    qa_entries: list[dict[str, Any]] = []
    query_index = 0
    for row in rows:
        doc_id = str(row.get("id", "")).strip()
        if not doc_id:
            continue
        documents[doc_id] = _document_text(row)
        qas = row.get("qas") or {}
        questions = qas.get("question") or []
        answers_all = qas.get("answers") or []
        for question, answer_bundle in zip(questions, answers_all):
            answer_texts: list[str] = []
            retrieval_spans: list[str] = []
            for answer in (answer_bundle or {}).get("answer", []) or []:
                if not isinstance(answer, dict) or answer.get("unanswerable"):
                    continue
                if answer.get("extractive_spans"):
                    text = " ".join(value for value in answer.get("extractive_spans", []) if isinstance(value, str))
                elif isinstance(answer.get("free_form_answer"), str) and answer["free_form_answer"].strip():
                    text = answer["free_form_answer"].strip()
                elif answer.get("yes_no") is not None:
                    text = "Yes" if bool(answer["yes_no"]) else "No"
                else:
                    text = ""
                evidence = answer.get("evidence") or answer.get("highlighted_evidence") or []
                if text:
                    answer_texts.append(text)
                if isinstance(evidence, str) and evidence.strip():
                    retrieval_spans.append(evidence.strip())
                elif isinstance(evidence, list):
                    retrieval_spans.extend(
                        value.strip() for value in evidence if isinstance(value, str) and value.strip()
                    )
            question_text = str(question or "").strip()
            if question_text and (answer_texts or retrieval_spans):
                qa_entries.append(
                    {
                        "id": query_index,
                        "question": question_text,
                        "document_id": doc_id,
                        "answers": answer_texts,
                        "retrieval_spans": retrieval_spans,
                    }
                )
                query_index += 1
    return documents, qa_entries


def load_qasper_bundle(
    *, split: str = "test", config_name: str | None = "default"
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    dataset = _load_dataset(config_name)
    if split not in dataset:
        raise KeyError(f"QASPER split {split!r} is unavailable; found {list(dataset.keys())}")
    documents, qa_entries = parse_qasper_rows(row for row in dataset[split] if isinstance(row, dict))
    metadata = {
        "dataset_source": "huggingface",
        "dataset_id": DATASET_ID,
        "requested_revision": "refs/convert/parquet" if (_datasets_major() or 0) >= 4 else None,
        "dataset_fingerprint": getattr(dataset[split], "_fingerprint", None),
        "dataset_name": "qasper",
        "config_name": config_name,
        "split": split,
        "documents": len(documents),
        "qa_entries": len(qa_entries),
    }
    logger.info("Loaded QASPER documents=%d qa_entries=%d", len(documents), len(qa_entries))
    return documents, qa_entries, metadata
