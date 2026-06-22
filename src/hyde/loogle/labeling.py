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
    prefix = [0] * len(pattern)
    matched = 0
    for index in range(1, len(pattern)):
        while matched and pattern[index] != pattern[matched]:
            matched = prefix[matched - 1]
        if pattern[index] == pattern[matched]:
            matched += 1
            prefix[index] = matched
    matched = 0
    for token in text:
        while matched and token != pattern[matched]:
            matched = prefix[matched - 1]
        if token == pattern[matched]:
            matched += 1
            if matched == len(pattern):
                return True
    return False


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


def _window_overlaps(chunk_tokens: list[str], span_tokens: list[str]) -> bool:
    if not chunk_tokens or not span_tokens:
        return False
    return (
        _contains(chunk_tokens, span_tokens)
        or _contains(span_tokens, chunk_tokens)
        or _suffix_prefix_overlap(chunk_tokens, span_tokens) > 0
        or _suffix_prefix_overlap(span_tokens, chunk_tokens) > 0
    )


def _suffix_prefix_overlap(text: list[str], prefix_source: list[str]) -> int:
    """Length of the longest suffix of text equal to a prefix of prefix_source."""

    if not text or not prefix_source:
        return 0
    prefix = [0] * len(prefix_source)
    matched = 0
    for index in range(1, len(prefix_source)):
        while matched and prefix_source[index] != prefix_source[matched]:
            matched = prefix[matched - 1]
        if prefix_source[index] == prefix_source[matched]:
            matched += 1
            prefix[index] = matched
    matched = 0
    for index, token in enumerate(text):
        while matched and token != prefix_source[matched]:
            matched = prefix[matched - 1]
        if token == prefix_source[matched]:
            matched += 1
            if matched == len(prefix_source) and index != len(text) - 1:
                matched = prefix[matched - 1]
    return matched


def _build_window_index(
    tokenized: list[tuple[ChunkRecord, list[str]]],
) -> tuple[str, list[tuple[ChunkRecord, int, int]]]:
    parts: list[str] = []
    ranges: list[tuple[ChunkRecord, int, int]] = []
    offset = 0
    for chunk, tokens in tokenized:
        text = " ".join(tokens)
        start = offset
        end = start + len(text)
        parts.append(text)
        ranges.append((chunk, start, end))
        offset = end + 1
    return " ".join(parts), ranges


def _indexed_window_matches(
    span_tokens: list[str],
    window_index: tuple[str, list[tuple[ChunkRecord, int, int]]],
) -> list[str]:
    span_text = " ".join(span_tokens)
    if not span_text:
        return []
    document_text, ranges = window_index
    occurrence_ranges: list[tuple[int, int]] = []
    start = document_text.find(span_text)
    while start >= 0:
        occurrence_ranges.append((start, start + len(span_text)))
        start = document_text.find(span_text, start + 1)
    if not occurrence_ranges:
        return []
    return _dedupe(
        chunk.chunk_id
        for chunk, chunk_start, chunk_end in ranges
        if any(chunk_start < span_end and span_start < chunk_end for span_start, span_end in occurrence_ranges)
    )


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


def _targets(
    entry: dict[str, Any],
    chunks: list[ChunkRecord],
    *,
    tokenized: list[tuple[ChunkRecord, list[str]]] | None = None,
    window_index: tuple[str, list[tuple[ChunkRecord, int, int]]] | None = None,
) -> tuple[list[str], list[str], list[list[str]]]:
    if tokenized is None:
        tokenized = [(chunk, _tokens(chunk.raw_text)) for chunk in chunks]
    span_mode = str(entry.get("retrieval_span_mode") or "text").strip().lower()
    gold: list[str] = []
    silver: list[str] = []
    groups: list[list[str]] = []
    seen_groups: set[tuple[str, ...]] = set()
    for span in _spans(entry):
        span_tokens = _tokens(span)
        if span_mode == "window":
            matches = _indexed_window_matches(span_tokens, window_index) if window_index is not None else []
            if not matches:
                matches = _dedupe(
                    chunk.chunk_id for chunk, chunk_tokens in tokenized if _window_overlaps(chunk_tokens, span_tokens)
                )
            if len(matches) == 1:
                gold.extend(matches)
            elif len(matches) > 1:
                group_key = tuple(matches)
                if group_key not in seen_groups:
                    groups.append(matches)
                    seen_groups.add(group_key)
                silver.extend(matches)
            continue
        if span_mode != "text":
            raise ValueError(f"Unsupported retrieval_span_mode={span_mode!r}")
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
        tokenized = [(chunk, _tokens(chunk.raw_text)) for chunk in chunks]
        window_index = _build_window_index(tokenized)
        for fallback_index, entry in entries:
            question = str(entry.get("question", "")).strip()
            if not question:
                continue
            gold, silver, groups = _targets(entry, chunks, tokenized=tokenized, window_index=window_index)
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
