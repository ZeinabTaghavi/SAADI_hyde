"""Load LooGLE without importing the parent SAADI repository."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DATASET_IDS = ("bigai-nlco/LooGLE", "bigainlco/LooGLE")


def _html_to_text(text: str) -> str:
    if "<html>" in text.lower():
        try:
            from bs4 import BeautifulSoup

            text = BeautifulSoup(text, "html.parser").get_text()
        except Exception:
            pass
    return re.sub(r"\n+", "\n", text)


def _datasets():
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise RuntimeError("LooGLE loading requires the 'datasets' package.") from exc
    return load_dataset, get_dataset_config_names


def _datasets_major_version() -> int | None:
    try:
        return int(package_version("datasets").split(".", 1)[0])
    except Exception:
        return None


def coerce_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        return _html_to_text(text) if text else ""
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return " ".join(part for part in (coerce_to_text(item) for item in value) if part)
    if isinstance(value, dict):
        for key in ("text", "document", "story", "content", "passage", "article", "context"):
            text = coerce_to_text(value.get(key))
            if text:
                return text
        return " ".join(part for part in (coerce_to_text(item) for item in value.values()) if part)
    return str(value).strip()


def _get_config_names(dataset_id: str) -> list[str]:
    _, get_dataset_config_names = _datasets()
    kwargs: dict[str, Any] = {}
    if (_datasets_major_version() or 0) >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    try:
        try:
            return list(get_dataset_config_names(dataset_id, **kwargs) or [])
        except TypeError:
            return list(get_dataset_config_names(dataset_id) or [])
    except Exception:
        return []


def _resolve_dataset(config_name: str) -> tuple[str, str]:
    for dataset_id in DATASET_IDS:
        configs = _get_config_names(dataset_id)
        if config_name in configs:
            return dataset_id, config_name
        if configs and "shortdep_qa" in configs:
            return dataset_id, "shortdep_qa"
    return DATASET_IDS[0], config_name


def _load_dataset(config_name: str):
    load_dataset, _ = _datasets()
    preferred_id, preferred_config = _resolve_dataset(config_name)
    ids = [preferred_id, *[item for item in DATASET_IDS if item != preferred_id]]
    kwargs_options: list[dict[str, Any]] = []
    if (_datasets_major_version() or 0) >= 4:
        kwargs_options.append({"revision": "refs/convert/parquet"})
    kwargs_options.append({})

    def attempt(*, offline: bool):
        previous_hub = os.environ.get("HF_HUB_OFFLINE")
        previous_datasets = os.environ.get("HF_DATASETS_OFFLINE")
        download_config = None
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            try:
                from datasets import DownloadConfig

                download_config = DownloadConfig(local_files_only=True)
            except Exception:
                pass
        last_error: Exception | None = None
        try:
            for dataset_id in ids:
                for candidate_config in (preferred_config, config_name):
                    for kwargs in kwargs_options:
                        call_kwargs = dict(kwargs)
                        if download_config is not None:
                            call_kwargs["download_config"] = download_config
                        try:
                            return load_dataset(dataset_id, name=candidate_config, **call_kwargs)
                        except TypeError:
                            try:
                                return load_dataset(dataset_id, name=candidate_config)
                            except Exception as exc:
                                last_error = exc
                        except Exception as exc:
                            last_error = exc
            return last_error
        finally:
            if offline:
                if previous_hub is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = previous_hub
                if previous_datasets is None:
                    os.environ.pop("HF_DATASETS_OFFLINE", None)
                else:
                    os.environ["HF_DATASETS_OFFLINE"] = previous_datasets

    result = attempt(offline=False)
    if not isinstance(result, Exception):
        return result
    offline_result = attempt(offline=True)
    if not isinstance(offline_result, Exception):
        return offline_result
    raise RuntimeError(f"Could not load LooGLE config={config_name!r} from {DATASET_IDS}.") from offline_result


def _iter_rows(dataset: Any, split: str) -> Iterable[dict[str, Any]]:
    if split not in dataset:
        raise KeyError(f"LooGLE split {split!r} is unavailable; found {list(dataset.keys())}")
    for row in dataset[split]:
        if isinstance(row, dict):
            yield row


def _document_id(row: dict[str, Any], index: int) -> str:
    for key in ("doc_id", "document_id", "docid", "title"):
        value = row.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    return f"doc_{index}"


def _document_text(row: dict[str, Any]) -> str:
    for key in ("context", "input", "document", "text", "article", "passage"):
        text = coerce_to_text(row.get(key))
        if text:
            return text
    return ""


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_text_list(item))
        return result
    text = coerce_to_text(value)
    return [text] if text else []


def _qa_pairs(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value.strip())
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except Exception:
            pass
    return []


def parse_loogle_rows(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Convert raw LooGLE rows to stable document and QA records."""

    documents: dict[str, str] = {}
    qa_entries: list[dict[str, Any]] = []
    query_index = 0
    for row_index, row in enumerate(rows):
        doc_id = _document_id(row, row_index)
        text = _document_text(row)
        if text and doc_id not in documents:
            documents[doc_id] = text

        pairs = _qa_pairs(row.get("qa_pairs"))
        if pairs:
            candidates = [
                (
                    coerce_to_text(pair.get("Q") or pair.get("question")),
                    _text_list(pair.get("A") or pair.get("answer")),
                    _text_list(pair.get("S") or pair.get("evidence")),
                )
                for pair in pairs
            ]
        else:
            candidates = [
                (
                    coerce_to_text(row.get("question") or row.get("Q") or row.get("query")),
                    _text_list(row.get("answer") or row.get("A") or row.get("answers")),
                    _text_list(row.get("evidence") or row.get("S") or row.get("span")),
                )
            ]

        for question, answers, spans in candidates:
            if not question or not (answers or spans):
                continue
            qa_entries.append(
                {
                    "id": query_index,
                    "question": question,
                    "document_id": doc_id,
                    "answers": answers,
                    "retrieval_spans": spans,
                }
            )
            query_index += 1
    return documents, qa_entries


def load_loogle_bundle(*, split: str = "test", config_name: str = "shortdep_qa") -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    dataset = _load_dataset(config_name)
    documents, qa_entries = parse_loogle_rows(_iter_rows(dataset, split))
    split_dataset = dataset[split]
    metadata = {
        "dataset_source": "huggingface",
        "dataset_ids_tried": list(DATASET_IDS),
        "requested_revision": "refs/convert/parquet" if (_datasets_major_version() or 0) >= 4 else None,
        "dataset_fingerprint": getattr(split_dataset, "_fingerprint", None),
        "dataset_name": "loogle",
        "config_name": config_name,
        "split": split,
        "documents": len(documents),
        "qa_entries": len(qa_entries),
    }
    logger.info("Loaded LooGLE documents=%d qa_entries=%d", len(documents), len(qa_entries))
    return documents, qa_entries, metadata


def load_subset_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("document_ids"), list):
        raise ValueError(f"Invalid subset manifest: {path}")
    return payload


def select_frozen_subset(
    documents: dict[str, str],
    qa_entries: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    max_documents: int | None = None,
    max_qa_entries: int | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    frozen_ids = [str(value) for value in manifest["document_ids"]]
    if max_documents is not None:
        if max_documents <= 0:
            raise ValueError("max_documents must be positive")
        frozen_ids = frozen_ids[:max_documents]
    missing_ids = [doc_id for doc_id in frozen_ids if doc_id not in documents]
    if missing_ids:
        raise RuntimeError(f"Frozen LooGLE document IDs are missing from the loaded dataset: {missing_ids}")
    selected_documents = {doc_id: documents[doc_id] for doc_id in frozen_ids}
    selected_set = set(frozen_ids)
    selected_qa = [row for row in qa_entries if str(row.get("document_id")) in selected_set]
    if max_qa_entries is not None:
        if max_qa_entries <= 0:
            raise ValueError("max_qa_entries must be positive")
        selected_qa = selected_qa[:max_qa_entries]
    return selected_documents, selected_qa, {
        "frozen_manifest_name": manifest.get("name"),
        "selected_document_ids": frozen_ids,
        "documents_selected": len(selected_documents),
        "qa_entries_selected": len(selected_qa),
        "limited": max_documents is not None or max_qa_entries is not None,
    }
