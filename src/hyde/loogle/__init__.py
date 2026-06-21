"""Standalone LooGLE support for the HyDE retrieval baseline."""

from .chunking import chunk_documents_grouped_records
from .dataset import load_loogle_bundle, select_frozen_subset
from .labeling import build_retrieval_examples
from .types import ChunkRecord, RetrievalExample, RetrievalResult

__all__ = [
    "ChunkRecord",
    "RetrievalExample",
    "RetrievalResult",
    "build_retrieval_examples",
    "chunk_documents_grouped_records",
    "load_loogle_bundle",
    "select_frozen_subset",
]
