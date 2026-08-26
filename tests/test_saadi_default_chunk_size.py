from pathlib import Path

import yaml

from hyde.loogle.chunking import DEFAULT_CHUNK_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAADI_DATASET_CONFIGS = (
    "qasper_hyde.yaml",
    "qasper_64k_hyde.yaml",
    "musique_32k_hyde.yaml",
)


def test_qasper_and_musique_use_saadi_default_chunk_size() -> None:
    assert DEFAULT_CHUNK_SIZE == 100
    for filename in SAADI_DATASET_CONFIGS:
        payload = yaml.safe_load((PROJECT_ROOT / "configs" / filename).read_text(encoding="utf-8"))
        assert payload["chunking"]["chunk_size"] == DEFAULT_CHUNK_SIZE
        assert payload["chunking"]["chunk_overlap"] == 0
