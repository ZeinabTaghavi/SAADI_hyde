#!/usr/bin/env python3
"""Validate the bundled QASPER-64K and MuSiQue-32K main-table data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hyde.loogle.prepared_expanded import BUNDLE_SPECS, load_prepared_expanded_bundle  # noqa: E402


def main() -> int:
    report = {}
    for dataset_name, spec in BUNDLE_SPECS.items():
        config_path = SCRIPT_DIR / "configs" / f"{dataset_name}_hyde.yaml"
        documents, qa_entries, metadata = load_prepared_expanded_bundle(
            dataset_variant=dataset_name,
            split=str(spec["split"]),
            config_path=config_path,
            data_dir=f"../data/{dataset_name}",
        )
        report[dataset_name] = {
            "checksum_validation": "passed",
            "documents": len(documents),
            "queries": len(qa_entries),
            "queries_with_retrieval_spans": sum(bool(row["retrieval_spans"]) for row in qa_entries),
            "target_context_tokens": metadata["target_context_tokens"],
            "prepared_root": metadata["prepared_root"],
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
