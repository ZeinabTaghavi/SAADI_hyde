"""Append-only, resumable cache for expensive hypothetical documents."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HypothesisCache:
    def __init__(self, path: str | Path, *, resume: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: dict[str, dict[str, Any]] = {}
        if resume and self.path.exists():
            self._load()
        elif not resume:
            self.path.write_text("", encoding="utf-8")

    def _load(self) -> None:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        valid_lines: list[str] = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if index == len(lines) - 1:
                    logger.warning("Ignoring an incomplete final cache line in %s", self.path)
                    self.path.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""), encoding="utf-8")
                    break
                raise ValueError(f"Corrupt hypothesis cache line {index + 1}: {self.path}") from exc
            if not isinstance(row, dict) or "query_id" not in row:
                raise ValueError(f"Invalid hypothesis cache row {index + 1}: {self.path}")
            self.rows[str(row["query_id"])] = row
            valid_lines.append(line)

    def get(self, query_id: str, *, expected_count: int) -> dict[str, Any] | None:
        row = self.rows.get(str(query_id))
        if row is None:
            return None
        hypotheses = row.get("hypothetical_documents")
        if not isinstance(hypotheses, list) or len(hypotheses) != expected_count:
            logger.warning("Ignoring incomplete hypothesis cache entry query_id=%s", query_id)
            return None
        if any(not isinstance(item, str) or not item.strip() for item in hypotheses):
            logger.warning("Ignoring empty hypothesis cache entry query_id=%s", query_id)
            return None
        return row

    def append(self, row: dict[str, Any]) -> None:
        query_id = str(row.get("query_id", ""))
        if not query_id:
            raise ValueError("A cached hypothesis row requires query_id")
        serialized = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows[query_id] = dict(row)
