from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import yaml

from hyde.loogle import runner

TABLE_SCRIPT = Path(__file__).resolve().parents[1] / "generate_hyde_retriever_table.py"
TABLE_SPEC = importlib.util.spec_from_file_location("generate_hyde_retriever_table", TABLE_SCRIPT)
assert TABLE_SPEC is not None and TABLE_SPEC.loader is not None
table_generator = importlib.util.module_from_spec(TABLE_SPEC)
TABLE_SPEC.loader.exec_module(table_generator)


class FakeGenerator:
    def generate(self, prompt: str) -> list[str]:
        return [f"Alpha hypothetical passage {index}" for index in range(8)]


class FakeEncoder:
    model_name = "fake/contriever"
    device = "cpu"

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            rows.append([1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def test_mocked_end_to_end_run_writes_top5_top10_and_tables(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    manifest_path = config_dir / "subset.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "fixture",
                "document_ids": ["d1"],
                "expected": {"documents": 1, "chunks": 2, "retrieval_examples": 2, "average_chunk_tokens": 5.0},
            }
        ),
        encoding="utf-8",
    )
    config = {
        "dataset": {"config_name": "shortdep_qa", "split": "test", "subset_manifest": "subset.json"},
        "chunking": {"chunk_size": 5, "chunk_overlap": 0},
        "retrieval": {"encoder": {"model_name": "fake/contriever"}},
        "hyde": {
            "prompt_task": "web search",
            "generator": {
                "model_name": "fake/qwen",
                "n": 8,
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.8,
                "attempts_per_hypothesis": 2,
            },
        },
        "seed": None,
    }
    config_path = config_dir / "loogle_hyde.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    documents = {"d1": "Alpha beta gamma delta epsilon. Zeta eta theta iota kappa."}
    qa_entries = [
        {"id": 0, "document_id": "d1", "question": "Where is alpha?", "retrieval_spans": ["Alpha beta"]},
        {"id": 1, "document_id": "d1", "question": "What crosses?", "retrieval_spans": ["epsilon Zeta"]},
    ]
    monkeypatch.setattr(runner, "load_loogle_bundle", lambda **_: (documents, qa_entries, {"dataset_name": "loogle"}))

    outputs = runner.run_experiment(
        config_path=config_path,
        output_root=tmp_path / "evaluations",
        work_root=tmp_path / "runs",
        top_ks=[5, 10],
        generator_override=FakeGenerator(),
        encoder_override=FakeEncoder(),
    )
    assert len(outputs) == 2
    for output in outputs:
        for relative in (
            "index/chunk_index.jsonl",
            "index/index_stats.json",
            "retrieval/retrieval_examples.jsonl",
            "retrieval/retrieval_payloads.jsonl",
            "retrieval/retrieval_results.json",
            "metrics_per_query.jsonl",
            "metrics_summary.json",
            "leaderboard_row.json",
            "evaluation_manifest.json",
        ):
            assert (output / relative).is_file(), relative
    top10 = json.loads((outputs[1] / "leaderboard_row.json").read_text())
    assert top10["method_name"] == "hyde"
    assert top10["n_queries"] == 2
    assert "gold_recall@5" in top10 and "gold_recall@10" in top10

    table_dir = tmp_path / "tables"
    assert table_generator.main(["--input-root", str(tmp_path / "evaluations"), "--output-dir", str(table_dir)]) == 0
    assert (table_dir / "hyde_retriever_rows.csv").is_file()
    assert (table_dir / "hyde_retriever_rows.md").is_file()
    assert (table_dir / "hyde_retriever_rows.tex").is_file()
