"""SAADI-compatible sentence-aware word chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .dataset import coerce_to_text
from .types import ChunkRecord

_CLAUSE_BREAK_RE = re.compile(r"(?<=[;:])\s+")
_SENTENCE_EXTRACT_RE = re.compile(r".+?(?:[.!?;:](?:[\"')\]]+)?|$)", flags=re.DOTALL)


@dataclass(slots=True)
class _Unit:
    text: str
    words: list[str]


@dataclass(slots=True)
class _SpannedUnit:
    text: str
    start: int
    end: int


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_EXTRACT_RE.finditer(str(text or "").strip()) if match.group(0).strip()]


def _hard_split(text: str, chunk_size: int) -> list[_Unit]:
    words = text.split()
    return [_Unit(" ".join(words[start : start + chunk_size]), words[start : start + chunk_size]) for start in range(0, len(words), chunk_size)]


def _units(text: str, chunk_size: int) -> list[_Unit]:
    sentence_texts = _sentences(text) or [str(text or "").strip()]
    output: list[_Unit] = []
    for sentence in sentence_texts:
        words = sentence.split()
        if len(words) <= chunk_size:
            if words:
                output.append(_Unit(sentence, words))
            continue
        clauses = [part.strip() for part in _CLAUSE_BREAK_RE.split(sentence) if part.strip()]
        if len(clauses) <= 1:
            output.extend(_hard_split(sentence, chunk_size))
            continue
        for clause in clauses:
            clause_words = clause.split()
            if len(clause_words) <= chunk_size:
                output.append(_Unit(clause, clause_words))
            else:
                output.extend(_hard_split(clause, chunk_size))
    return output


def chunk_text(text: str, *, chunk_size: int = 500, chunk_overlap: int = 0) -> list[tuple[str, int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    sentence_texts = _sentences(text)
    if len(sentence_texts) <= 1:
        words = str(text or "").split()
        output: list[tuple[str, int, int]] = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            output.append((" ".join(words[start:end]), start, end))
            if end == len(words):
                break
            start = end - chunk_overlap
        return output

    spanned: list[_SpannedUnit] = []
    offset = 0
    for unit in _units(text, chunk_size):
        spanned.append(_SpannedUnit(unit.text, offset, offset + len(unit.words)))
        offset += len(unit.words)

    output: list[tuple[str, int, int]] = []
    index = 0
    while index < len(spanned):
        end_index = index
        word_count = 0
        while end_index < len(spanned):
            unit_count = spanned[end_index].end - spanned[end_index].start
            if word_count and word_count + unit_count > chunk_size:
                break
            word_count += unit_count
            end_index += 1
        if end_index == index:
            end_index += 1
        packed = spanned[index:end_index]
        output.append((" ".join(unit.text.strip() for unit in packed), packed[0].start, packed[-1].end))
        if end_index >= len(spanned):
            break
        if chunk_overlap == 0:
            index = end_index
            continue
        overlap_words = 0
        overlap_units = 0
        for unit_index in range(end_index - 1, index - 1, -1):
            count = spanned[unit_index].end - spanned[unit_index].start
            if overlap_words + count > chunk_overlap:
                break
            overlap_words += count
            overlap_units += 1
        index = max(index + 1, end_index - overlap_units)
    return output


def chunk_documents_grouped_records(
    documents: list[Any],
    *,
    doc_ids: list[str],
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> list[list[ChunkRecord]]:
    if len(documents) != len(doc_ids):
        raise ValueError("documents and doc_ids must have the same length")
    grouped: list[list[ChunkRecord]] = []
    for doc_id, document in zip(doc_ids, documents):
        text = coerce_to_text(document)
        grouped.append(
            [
                ChunkRecord(
                    doc_id=str(doc_id),
                    chunk_id=f"{doc_id}:{index}",
                    chunk_index=index,
                    raw_text=chunk,
                    token_start=start,
                    token_end=end,
                )
                for index, (chunk, start, end) in enumerate(
                    chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                )
            ]
        )
    return grouped
