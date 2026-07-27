#!/usr/bin/env python3
"""Run a standalone HyDE retrieval baseline from a frozen dataset configuration."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hyde.loogle.runner import parse_top_ks, run_experiment  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "configs" / "loogle_hyde.yaml"))
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "hyde_evaluations"))
    parser.add_argument("--work-root", default=str(SCRIPT_DIR / "hyde_runs"))
    parser.add_argument("--top-ks", nargs="+", default=["5", "10"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--retriever",
        default="contriever",
        choices=["bm25", "contriever", "bge_m3", "qwen", "jina"],
        help="Standalone retriever evaluated with the shared HyDE hypotheses.",
    )
    parser.add_argument("--max-documents", type=int, default=None, help="Smoke-test limit over the frozen document list.")
    parser.add_argument("--max-qa-entries", type=int, default=None, help="Smoke-test limit applied before evidence labeling.")
    parser.add_argument("--embedding-device", default=None, help="Dense retriever device, for example cuda:0 or cpu.")
    parser.add_argument(
        "--embedding-device-map",
        default=None,
        help="Optional Transformers device_map for a dense retriever, for example auto.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume completed hypothetical documents from the JSONL cache (default: true).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite completed evaluation artifacts while retaining caches.")
    parser.add_argument("--force-embeddings", action="store_true", help="Recompute cached dense document embeddings.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load, select, chunk, and label the dataset, then verify frozen counts without loading either model.",
    )
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument(
        "--generate-only",
        action="store_true",
        help="Prepare and cache all hypothetical documents, then exit before loading a retriever.",
    )
    phase.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Require a complete hypothesis cache and run only the selected retriever.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        outputs = run_experiment(
            config_path=args.config,
            output_root=args.output_root,
            work_root=args.work_root,
            top_ks=parse_top_ks(args.top_ks),
            run_name=args.run_name,
            retriever_name=args.retriever,
            max_documents=args.max_documents,
            max_qa_entries=args.max_qa_entries,
            embedding_device=args.embedding_device,
            embedding_device_map=args.embedding_device_map,
            resume=args.resume,
            force=args.force,
            force_embeddings=args.force_embeddings,
            validate_only=args.validate_only,
            generate_only=args.generate_only,
            retrieval_only=args.retrieval_only,
        )
    except Exception as exc:
        logging.getLogger("run_hyde").exception("HyDE dataset run failed: %s", exc)
        return 1
    if outputs:
        print(json.dumps({"output_dirs": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
