"""Standalone NovelHopQA loader over the external whole-book corpus."""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from .dataset import coerce_to_text

logger = logging.getLogger(__name__)
DATASET_ID = "abhaygupta1266/novelhopqa"
HOP_SPLITS = ("hop_1", "hop_2", "hop_3", "hop_4")


def _datasets_major() -> int | None:
    try:
        return int(package_version("datasets").split(".", 1)[0])
    except Exception:
        return None


def _normalize_book_key(value: str | None) -> str:
    text = str(value or "").strip()
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\xa0", " ")
        .replace("\ufeff", " ")
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", text)
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9]+", " ", text)).strip().casefold()


def _safe_component(value: str, default: str) -> str:
    output = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return output or default


def _title_variants(value: str | None) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    variants = [raw]
    for separator in (":", ";", ",", " - ", " — ", " – "):
        if separator in raw:
            head = raw.split(separator, 1)[0].strip()
            if head:
                variants.append(head)
    for candidate in list(variants):
        lowered = candidate.casefold()
        for prefix in ("the ", "a ", "an "):
            if lowered.startswith(prefix):
                variants.append(candidate[len(prefix) :].strip())
    return list(dict.fromkeys(value for value in variants if value))


def _title_like_lines(text: str, limit: int = 5) -> list[str]:
    candidates: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.replace("\ufeff", "").replace("\xa0", " ").strip()
        if not line:
            continue
        lowered = line.casefold()
        if "project gutenberg ebook of" in lowered:
            match = re.search(r"project gutenberg ebook of\s+(.+?)(?:,\s+by\b|$)", line, re.IGNORECASE)
            if match:
                candidates.append(match.group(1).strip(" .,:;!-"))
            continue
        if (
            "project gutenberg" in lowered
            or "www.gutenberg.org" in lowered
            or lowered.startswith(("***", "by ", "translated by ", "produced by ", "release date", "language:"))
            or len(line) > 160
            or sum(character.isalpha() for character in line) < 3
        ):
            continue
        candidates.append(line.strip(" .,:;!-"))
        if len(candidates) >= limit:
            break
    return list(dict.fromkeys(value for value in candidates if value))


def _resolve_books_root(configured: str | os.PathLike[str] | None) -> Path:
    raw = os.getenv("NOVELHOPQA_BOOKS_ROOT") or str(configured or "").strip()
    if not raw:
        raise RuntimeError(
            "NovelHopQA requires the whole-book corpus. Set NOVELHOPQA_BOOKS_ROOT to the directory "
            "containing bookmeta.json and Books/."
        )
    root = Path(raw).expanduser().resolve()
    if not (root / "bookmeta.json").is_file():
        raise RuntimeError(f"NovelHopQA bookmeta.json was not found under {root}")
    return root


def _find_book_text(root: Path, doc_id: str, copyright_group: str | None) -> Path | None:
    candidates = []
    if copyright_group:
        candidates.append(root / "Books" / str(copyright_group) / f"{doc_id}.txt")
    candidates.extend(
        [
            root / "Books" / "PublicDomain" / f"{doc_id}.txt",
            root / "Books" / "CopyrightProtected" / f"{doc_id}.txt",
            root / "Demonstration" / f"{doc_id}.txt",
            root / f"{doc_id}.txt",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return next((path for path in root.rglob(f"{doc_id}.txt") if path.is_file()), None)


def load_book_subset(
    books_root: str | os.PathLike[str] | None,
    allowed_doc_ids: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    root = _resolve_books_root(books_root)
    with (root / "bookmeta.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected object-shaped NovelHopQA bookmeta.json under {root}")
    documents: dict[str, str] = {}
    title_to_doc: dict[str, str] = {}
    for doc_id in allowed_doc_ids:
        row = metadata.get(doc_id)
        if not isinstance(row, dict):
            raise RuntimeError(f"NovelHopQA book {doc_id} is absent from bookmeta.json")
        title = coerce_to_text(row.get("title") or row.get("book") or row.get("name"))
        path = _find_book_text(root, doc_id, row.get("copyright"))
        if path is None:
            raise RuntimeError(f"NovelHopQA text file for {doc_id} ({title}) was not found under {root}")
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            raise RuntimeError(f"NovelHopQA text file is empty: {path}")
        documents[doc_id] = text
        for alias in (title, doc_id, path.stem, *_title_like_lines(text)):
            for variant in _title_variants(alias):
                key = _normalize_book_key(variant)
                if key:
                    title_to_doc.setdefault(key, doc_id)
    return documents, title_to_doc


def _load_hop(split_name: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("NovelHopQA loading requires the 'datasets' package.") from exc
    kwargs: dict[str, Any] = {}
    if (_datasets_major() or 0) >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    try:
        return load_dataset(DATASET_ID, "default", split=split_name, **kwargs)
    except TypeError:
        return load_dataset(DATASET_ID, split=split_name, **kwargs)


def parse_novelhop_rows(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    title_to_doc: dict[str, str],
) -> tuple[list[dict[str, Any]], set[str]]:
    qa_entries: list[dict[str, Any]] = []
    missing_titles: set[str] = set()
    for split_name in HOP_SPLITS:
        for index, row in enumerate(rows_by_split.get(split_name, [])):
            title = coerce_to_text(row.get("book") or row.get("title") or row.get("book_title"))
            doc_id = title_to_doc.get(_normalize_book_key(title))
            if doc_id is None:
                if title:
                    missing_titles.add(title)
                continue
            context = coerce_to_text(row.get("context") or row.get("passage") or row.get("document"))
            question = coerce_to_text(row.get("question") or row.get("query"))
            answer = coerce_to_text(row.get("answer") or row.get("gold_answer"))
            if not context or not question or not answer:
                continue
            base_id = row.get("qid") or row.get("question_id") or row.get("id") or index
            qa_entries.append(
                {
                    "id": f"{split_name}:{_safe_component(str(base_id), str(index))}",
                    "question": question,
                    "document_id": doc_id,
                    "book_title": title,
                    "gold_context_window": context,
                    "retrieval_span_mode": "window",
                    "answers": [answer],
                    "retrieval_spans": [context],
                }
            )
    return qa_entries, missing_titles


def load_novelhopqa_bundle(
    *,
    split: str = "test",
    config_name: str | None = "all",
    books_root: str | os.PathLike[str] | None = None,
    allowed_doc_ids: set[str],
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    _ = split
    mode = str(config_name or "all").strip().lower()
    selected_splits = list(HOP_SPLITS) if mode in {"all", "default", "all_hops"} else [mode]
    if any(value not in HOP_SPLITS for value in selected_splits):
        raise ValueError(f"Unsupported NovelHopQA config_name={config_name!r}")
    documents, title_to_doc = load_book_subset(books_root, allowed_doc_ids)
    rows_by_split = {
        split_name: [row for row in _load_hop(split_name) if isinstance(row, dict)]
        for split_name in selected_splits
    }
    qa_entries, missing_titles = parse_novelhop_rows(rows_by_split, title_to_doc=title_to_doc)
    if missing_titles:
        logger.info("Ignored %d NovelHopQA title(s) outside the frozen book subset", len(missing_titles))
    metadata = {
        "dataset_source": "huggingface_plus_local_books",
        "dataset_id": DATASET_ID,
        "requested_revision": "refs/convert/parquet" if (_datasets_major() or 0) >= 4 else None,
        "dataset_name": "novelhopqa",
        "config_name": config_name,
        "hop_splits": selected_splits,
        "split": split,
        "books_root": str(_resolve_books_root(books_root)),
        "documents": len(documents),
        "qa_entries": len(qa_entries),
    }
    logger.info("Loaded NovelHopQA documents=%d qa_entries=%d", len(documents), len(qa_entries))
    return documents, qa_entries, metadata
