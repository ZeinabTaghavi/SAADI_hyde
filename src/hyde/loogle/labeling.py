"""Map LooGLE evidence spans to SAADI-compatible gold and silver labels."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any, Iterable

from .types import ChunkRecord, RetrievalExample


def _tokens(text: str) -> list[str]:
    normalized = (
        str(text or "")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\xa0", " ")
        .replace("\ufeff", " ")
    )
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return [token for token in re.sub(r"[^A-Za-z0-9]+", " ", normalized.casefold()).split() if token]


def _contains(text: list[str], pattern: list[str]) -> bool:
    if not pattern:
        return True
    if len(pattern) > len(text):
        return False
    return any(text[start : start + len(pattern)] == pattern for start in range(len(text) - len(pattern) + 1))


def _ordered_subsequence(text: list[str], pattern: list[str]) -> bool:
    if not pattern:
        return True
    position = 0
    for token in text:
        if token == pattern[position]:
            position += 1
            if position == len(pattern):
                return True
    return False


def _boundary_overlap(a: list[str], b: list[str]) -> int:
    maximum = min(len(a), len(b) - 1)
    suffix = next((size for size in range(maximum, 0, -1) if a[-size:] == b[:size]), 0)
    prefix = next((size for size in range(maximum, 0, -1) if a[:size] == b[-size:]), 0)
    return max(suffix, prefix)


def _classify(chunk_tokens: list[str], span_tokens: list[str]) -> str:
    if _contains(chunk_tokens, span_tokens) or _ordered_subsequence(chunk_tokens, span_tokens):
        return "full"
    return "partial" if _boundary_overlap(chunk_tokens, span_tokens) else "none"


def _dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _spans(entry: dict[str, Any]) -> list[str]:
    raw = entry.get("retrieval_spans")
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        return [str(value).strip() for value in raw if str(value).strip()]
    answers = entry.get("answers")
    if isinstance(answers, str):
        return [answers] if answers.strip() else []
    return [str(value).strip() for value in answers or [] if str(value).strip()]


def _targets(entry: dict[str, Any], chunks: list[ChunkRecord]) -> tuple[list[str], list[str], list[list[str]]]:
    tokenized = [(chunk, _tokens(chunk.raw_text)) for chunk in chunks]
    gold: list[str] = []
    silver: list[str] = []
    groups: list[list[str]] = []
    seen_groups: set[tuple[str, ...]] = set()
    for span in _spans(entry):
        span_tokens = _tokens(span)
        full = [chunk.chunk_id for chunk, chunk_tokens in tokenized if _classify(chunk_tokens, span_tokens) == "full"]
        if full:
            gold.extend(full)
            continue
        partial = [chunk.chunk_id for chunk, chunk_tokens in tokenized if _classify(chunk_tokens, span_tokens) == "partial"]
        partial = _dedupe(partial)
        if len(partial) <= 1:
            continue
        group_key = tuple(partial)
        if group_key not in seen_groups:
            groups.append(partial)
            seen_groups.add(group_key)
        silver.extend(partial)
    return _dedupe(gold), _dedupe(silver), groups


def build_retrieval_examples(
    qa_entries: list[dict[str, Any]], chunks_by_doc: dict[str, list[ChunkRecord]]
) -> list[RetrievalExample]:
    entries_by_doc: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, entry in enumerate(qa_entries):
        entries_by_doc[str(entry.get("document_id", ""))].append((index, entry))
    examples: list[RetrievalExample] = []
    for doc_id, entries in entries_by_doc.items():
        chunks = chunks_by_doc.get(doc_id, [])
        if not chunks:
            continue
        for fallback_index, entry in entries:
            question = str(entry.get("question", "")).strip()
            if not question:
                continue
            gold, silver, groups = _targets(entry, chunks)
            if not (gold or silver):
                continue
            examples.append(
                RetrievalExample(
                    query_id=str(entry.get("id", f"q{fallback_index}")),
                    doc_id=doc_id,
                    question=question,
                    gold_chunk_ids=gold,
                    silver_chunk_ids=silver,
                    silver_chunk_groups=groups,
                )
            )
    return examples
