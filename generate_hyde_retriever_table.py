#!/usr/bin/env python3
"""Generate a main-retrieval-style table from HyDE leaderboard rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "hyde_evaluations"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "hyde_evaluations_Tables"

DATASET_ORDER = {"loogle": 0, "qasper_64k": 1, "musique_32k": 2, "novelhopqa": 3}
DATASET_LABELS = {
    "loogle": "LooGLE",
    "qasper_64k": "QASPER-64K",
    "musique_32k": "MuSiQue-32K",
    "novelhopqa": "NovelHopQA",
}
RETRIEVER_LABELS = {
    "bm25": "BM25",
    "contriever": "Contriever",
    "bge_m3": "BGE-M3",
    "qwen": "Qwen",
    "jina": "Jina",
}
RETRIEVER_ORDER = {name: index for index, name in enumerate(RETRIEVER_LABELS)}
METHOD_LABEL = "HyDE"
EXPECTED_QUERY_COUNTS = {
    "loogle": 1_832,
    "qasper_64k": 1_333,
    "musique_32k": 900,
    "novelhopqa": 985,
}
LABELING_VERSION = "saadi_support_targets_v2"
EXPECTED_DATASET_CONTRACTS = {
    "loogle": {
        "chunk_size": 500,
        "chunk_overlap": 0,
        "n_chunks": 3_346,
        "average_chunk_tokens": 480.1628810520024,
        "eligible_queries_by_view": {
            "gold": 1_816,
            "silver_loose": 16,
            "union_loose": 1_832,
            "gold_hit": 1_816,
            "silver_strict_hit": 16,
            "strict_union_hit": 1_832,
        },
    },
    "qasper_64k": {
        "chunk_size": 100,
        "chunk_overlap": 0,
        "n_chunks": 17_159,
        "average_chunk_tokens": 88.8344891893467,
        "eligible_queries_by_view": {
            "gold": 645,
            "silver_loose": 1_111,
            "union_loose": 1_333,
            "gold_hit": 645,
            "silver_strict_hit": 1_111,
            "strict_union_hit": 1_333,
        },
    },
    "musique_32k": {
        "chunk_size": 100,
        "chunk_overlap": 0,
        "n_chunks": 17_102,
        "average_chunk_tokens": 87.48526488130044,
        "eligible_queries_by_view": {
            "gold": 688,
            "silver_loose": 823,
            "union_loose": 900,
            "gold_hit": 688,
            "silver_strict_hit": 823,
            "strict_union_hit": 900,
        },
    },
    "novelhopqa": {
        "chunk_size": 500,
        "chunk_overlap": 0,
        "n_chunks": 7_736,
        "average_chunk_tokens": 488.2821871768356,
        "eligible_queries_by_view": {
            "gold": 489,
            "silver_loose": 496,
            "union_loose": 985,
            "gold_hit": 489,
            "silver_strict_hit": 496,
            "strict_union_hit": 985,
        },
    },
}

# Exact column order used by outputs/Tables/main/retrieval/table_main_retrieval.
METRIC_FIELDS = (
    "gold_ndcg@5",
    "gold_recall@5",
    "gold_ndcg@10",
    "gold_recall@10",
    "silver_loose_ndcg@5",
    "silver_loose_recall@5",
    "silver_loose_ndcg@10",
    "silver_loose_recall@10",
    "union_loose_ndcg@5",
    "union_loose_recall@5",
    "union_loose_ndcg@10",
    "union_loose_recall@10",
    "gold_hit@5",
    "gold_hit@10",
    "silver_strict_hit@5",
    "silver_strict_hit@10",
    "strict_union_hit@5",
    "strict_union_hit@10",
)

CSV_FIELDS = (
    "dataset",
    "dataset_label",
    "retriever",
    "retriever_label",
    "method",
    "split",
    "run_name",
    "n_queries",
    "average_chunk_tokens",
    "labeling_version",
    "chunk_size",
    "chunk_overlap",
    "n_chunks",
) + METRIC_FIELDS + ("source_path",)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _top_k(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"top_(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer top-k from {path}")


def _metric_value(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return None if value is None else float(value)


def _is_retriever_scoped(path: Path, retriever: str) -> bool:
    parts = path.parts
    return any(
        parts[index] == "hyde" and parts[index + 1] == retriever and parts[index + 2].startswith("top_")
        for index in range(max(0, len(parts) - 2))
    )


def collect_records(input_root: Path) -> list[dict[str, Any]]:
    """Merge HyDE top-5/top-10 files into one main-table record per run."""

    grouped: dict[tuple[str, str, str], list[tuple[int, Path, dict[str, Any]]]] = {}
    for path in sorted(input_root.rglob("leaderboard_row.json")):
        payload = _read_json(path)
        if str(payload.get("method_name")) != "hyde":
            continue
        dataset = str(payload.get("dataset_name") or "").strip().lower()
        # The old 25-document QASPER run is intentionally excluded. The clean
        # main table contains only the four finalized dataset variants.
        if dataset not in DATASET_ORDER:
            continue
        retriever = str(payload.get("retriever_name") or "").strip().lower()
        if not retriever:
            # Backward compatibility for the completed legacy Contriever run.
            retriever = "contriever"
        run_name = str(payload.get("run_name") or path.parent.name)
        grouped.setdefault((dataset, retriever, run_name), []).append((_top_k(path), path, payload))

    records: list[dict[str, Any]] = []
    for (dataset, retriever, run_name), candidates in grouped.items():
        scoped = [candidate for candidate in candidates if _is_retriever_scoped(candidate[1], retriever)]
        if scoped:
            # Prefer the new retriever-scoped layout over a legacy Contriever
            # directory when both are present after an intentional rerun.
            candidates = scoped
        merged: dict[str, Any] = {}
        source_path: Path | None = None
        for _, path, payload in sorted(candidates, key=lambda item: item[0]):
            for field in (
                "dataset_name",
                "method_name",
                "split",
                "run_name",
                "n_queries",
                "average_chunk_tokens",
                "labeling_version",
                "chunk_size",
                "chunk_overlap",
                "n_chunks",
                "eligible_queries_by_view",
            ):
                previous = merged.get(field)
                current = payload.get(field)
                if previous is not None and current is not None and previous != current:
                    raise ValueError(
                        f"Mismatched HyDE top-k artifacts for {dataset}/{run_name}: "
                        f"field {field!r} is {previous!r} versus {current!r} in {path}"
                    )
            for key, current in payload.items():
                previous = merged.get(key)
                if (
                    "@" in key
                    and previous is not None
                    and current is not None
                    and previous != current
                ):
                    raise ValueError(
                        f"Mismatched HyDE top-k artifacts for {dataset}/{run_name}: "
                        f"metric {key!r} is {previous!r} versus {current!r} in {path}"
                    )
            merged.update(payload)
            source_path = path
        if source_path is None:
            continue
        record: dict[str, Any] = {
            "dataset": dataset,
            "dataset_label": DATASET_LABELS.get(dataset, dataset),
            "retriever": retriever,
            "retriever_label": RETRIEVER_LABELS.get(retriever, retriever),
            "method": METHOD_LABEL,
            "split": str(merged.get("split") or ""),
            "run_name": run_name,
            "n_queries": merged.get("n_queries"),
            "average_chunk_tokens": merged.get("average_chunk_tokens"),
            "labeling_version": merged.get("labeling_version"),
            "chunk_size": merged.get("chunk_size"),
            "chunk_overlap": merged.get("chunk_overlap"),
            "n_chunks": merged.get("n_chunks"),
            "eligible_queries_by_view": merged.get("eligible_queries_by_view"),
            "source_path": str(source_path.relative_to(input_root)),
        }
        for key in METRIC_FIELDS:
            record[key] = _metric_value(merged, key)
        records.append(record)

    records.sort(
        key=lambda record: (
            DATASET_ORDER.get(str(record["dataset"]), 99),
            str(record["dataset"]),
            RETRIEVER_ORDER.get(str(record["retriever"]), 99),
            str(record["retriever"]),
            str(record["run_name"]),
        )
    )
    _validate_unique_records(records)
    return records


def _validate_unique_records(records: list[dict[str, Any]]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for record in records:
        key = (str(record["dataset"]), str(record["retriever"]))
        source = str(record["source_path"])
        if key in seen:
            raise ValueError(
                f"Multiple HyDE runs found for dataset={key[0]!r} retriever={key[1]!r}: "
                f"{seen[key]} and {source}. "
                "Use --input-root to select one completed run population."
            )
        seen[key] = source


def validate_complete_main_grid(records: list[dict[str, Any]]) -> None:
    expected = {
        (dataset, retriever)
        for dataset in DATASET_ORDER
        for retriever in RETRIEVER_LABELS
    }
    actual = {(str(record["dataset"]), str(record["retriever"])) for record in records}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    count_errors = [
        (
            str(record["dataset"]),
            str(record["retriever"]),
            record.get("n_queries"),
            EXPECTED_QUERY_COUNTS[str(record["dataset"])],
        )
        for record in records
        if record.get("n_queries") != EXPECTED_QUERY_COUNTS[str(record["dataset"])]
    ]
    contract_errors: list[str] = []
    for record in records:
        dataset = str(record["dataset"])
        contract = EXPECTED_DATASET_CONTRACTS[dataset]
        if record.get("labeling_version") != LABELING_VERSION:
            contract_errors.append(
                f"{dataset}/{record['retriever']}:labeling_version="
                f"{record.get('labeling_version')!r}!=expected_{LABELING_VERSION!r}"
            )
        for field in ("chunk_size", "chunk_overlap", "n_chunks", "eligible_queries_by_view"):
            if record.get(field) != contract[field]:
                contract_errors.append(
                    f"{dataset}/{record['retriever']}:{field}="
                    f"{record.get(field)!r}!=expected_{contract[field]!r}"
                )
        average = record.get("average_chunk_tokens")
        if average is None or abs(float(average) - float(contract["average_chunk_tokens"])) > 1e-9:
            contract_errors.append(
                f"{dataset}/{record['retriever']}:average_chunk_tokens="
                f"{average!r}!=expected_{contract['average_chunk_tokens']!r}"
            )
    if missing or unexpected or count_errors or contract_errors:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(f"{dataset}/{retriever}" for dataset, retriever in missing))
        if unexpected:
            details.append(
                "unexpected=" + ", ".join(f"{dataset}/{retriever}" for dataset, retriever in unexpected)
            )
        if count_errors:
            details.append(
                "query_count_mismatches="
                + ", ".join(
                    f"{dataset}/{retriever}:{actual_count}!=expected_{expected_count}"
                    for dataset, retriever, actual_count, expected_count in count_errors
                )
            )
        if contract_errors:
            details.append("contract_mismatches=" + ", ".join(contract_errors))
        raise ValueError("Incomplete HyDE main-table grid: " + "; ".join(details))


def _format_pct(value: Any) -> str:
    return "--" if value is None else f"{float(value) * 100.0:.1f}"


def _metric_cells(record: dict[str, Any]) -> list[str]:
    return [_format_pct(record.get(key)) for key in METRIC_FIELDS]


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(CSV_FIELDS),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def _write_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    headers = [
        "Dataset",
        "Retriever",
        "Method",
        "Gold nDCG@5",
        "Gold Recall@5",
        "Gold nDCG@10",
        "Gold Recall@10",
        "Silver-L nDCG@5",
        "Silver-L Recall@5",
        "Silver-L nDCG@10",
        "Silver-L Recall@10",
        "Union-L nDCG@5",
        "Union-L Recall@5",
        "Union-L nDCG@10",
        "Union-L Recall@10",
        "Gold HR@5",
        "Gold HR@10",
        "Silver-S HR@5",
        "Silver-S HR@10",
        "Union-S HR@5",
        "Union-S HR@10",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for record in records:
        cells = [
            str(record["dataset_label"]),
            str(record["retriever_label"]),
            str(record["method"]),
            *_metric_cells(record),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def build_main_retrieval_latex(records: list[dict[str, Any]]) -> str:
    lines = [
        "% HyDE retrieval summary table.",
        "% Ranking metrics: NDCG@5/@10 and Recall@5/@10 over Gold/Silver-L/Union-L.",
        "% Binary metrics: HR@5 and HR@10 over Gold/Silver-S/Union-S.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Main retrieval summary for HyDE. Values are percentages.}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrrrrrrrrrrrrrr}",
        r"\toprule",
        r"Dataset & Retriever & Method & \multicolumn{12}{c}{Ranking Metrics} & \multicolumn{6}{c}{Binary Metrics} \\",
        r"\cmidrule(lr){4-15}\cmidrule(lr){16-21}",
        r"& & & \multicolumn{4}{c}{Gold} & \multicolumn{4}{c}{Silver-L} & \multicolumn{4}{c}{Union-L} & \multicolumn{2}{c}{Gold} & \multicolumn{2}{c}{Silver-S} & \multicolumn{2}{c}{Union-S} \\",
        r"& & & NDCG@5 & Recall@5 & NDCG@10 & Recall@10 & NDCG@5 & Recall@5 & NDCG@10 & Recall@10 & NDCG@5 & Recall@5 & NDCG@10 & Recall@10 & HR@5 & HR@10 & HR@5 & HR@10 & HR@5 & HR@10 \\",
        r"\midrule",
    ]
    for index, record in enumerate(records):
        if index:
            lines.append(r"\midrule")
        cells = [
            _latex_escape(str(record["dataset_label"])),
            _latex_escape(str(record["retriever_label"])),
            _latex_escape(str(record["method"])),
            *_metric_cells(record),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="table_main_retrieval_hyde")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require all 20 finalized dataset/retriever rows with exact query counts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = args.input_root.expanduser().resolve()
    records = collect_records(input_root)
    if not records:
        raise FileNotFoundError(f"No HyDE leaderboard rows found under {input_root}")
    if args.strict:
        validate_complete_main_grid(records)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix)
    outputs = {
        "jsonl": output_dir / f"{prefix}.jsonl",
        "csv": output_dir / f"{prefix}.csv",
        "markdown": output_dir / f"{prefix}.md",
        "latex": output_dir / f"{prefix}.tex",
        "text": output_dir / f"{prefix}.txt",
    }
    _write_jsonl(outputs["jsonl"], records)
    _write_csv(outputs["csv"], records)
    _write_markdown(outputs["markdown"], records)
    latex = build_main_retrieval_latex(records)
    outputs["latex"].write_text(latex, encoding="utf-8")
    outputs["text"].write_text(latex, encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(records),
                "outputs": {key: str(value) for key, value in outputs.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
