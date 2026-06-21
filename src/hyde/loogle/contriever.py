"""Local Contriever encoder and reusable per-document embedding cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .types import ChunkRecord


def mean_pool(last_hidden_state: Any, attention_mask: Any, torch_module: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return torch_module.nn.functional.normalize(pooled, p=2, dim=1)


def combine_hyde_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Average the question and hypothetical-document embeddings, as in HyDE."""

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("HyDE requires a 2-D matrix containing a question and at least one hypothesis")
    return matrix.mean(axis=0, dtype=np.float32)


class ContrieverEncoder:
    def __init__(
        self,
        model_name: str = "facebook/contriever",
        *,
        device: str | None = None,
        batch_size: int = 128,
        max_length: int = 512,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Contriever requires torch and transformers.") from exc
        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Contriever device {self.device!r} requested but CUDA is unavailable")
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.cache_dir = cache_dir
        if local_files_only is None:
            local_files_only = os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"}
        common = {"cache_dir": cache_dir, "local_files_only": local_files_only, "token": False}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **common)
        # Contriever publishes pytorch_model.bin on main; forcing it avoids a lookup of community safetensors refs.
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=False, **common)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            hidden = int(getattr(self.model.config, "hidden_size", 0))
            return np.empty((0, hidden), dtype=np.float32)
        batches: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                encoded = self.tokenizer(
                    texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"], self.torch)
                batches.append(pooled.detach().cpu().float().numpy())
        return np.concatenate(batches, axis=0)


def _chunk_signature(chunks: list[ChunkRecord]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.raw_text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def embedding_cache_path(root: str | Path, doc_id: str) -> Path:
    safe_prefix = "".join(char if char.isalnum() or char in "._-" else "_" for char in doc_id)[:80] or "doc"
    suffix = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:10]
    return Path(root) / f"{safe_prefix}-{suffix}.npz"


def load_or_encode_document(
    encoder: ContrieverEncoder,
    chunks: list[ChunkRecord],
    cache_path: str | Path,
    *,
    force: bool = False,
) -> tuple[np.ndarray, bool]:
    path = Path(cache_path)
    signature = _chunk_signature(chunks)
    if path.exists() and not force:
        try:
            with np.load(path, allow_pickle=False) as cached:
                metadata = json.loads(str(cached["metadata"].item()))
                embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
            if (
                metadata.get("chunk_signature") == signature
                and metadata.get("model_name") == encoder.model_name
                and metadata.get("max_length") == getattr(encoder, "max_length", None)
                and embeddings.shape[0] == len(chunks)
            ):
                return embeddings, True
        except Exception:
            pass
    embeddings = encoder.encode([chunk.raw_text for chunk in chunks])
    metadata = {
        "doc_id": chunks[0].doc_id if chunks else "",
        "model_name": encoder.model_name,
        "max_length": getattr(encoder, "max_length", None),
        "chunk_signature": signature,
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "shape": list(embeddings.shape),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, embeddings=embeddings, metadata=json.dumps(metadata, sort_keys=True))
    os.replace(temporary, path)
    return embeddings, False


def rank_embeddings(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    *,
    k: int,
) -> tuple[list[int], list[float]]:
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    documents = np.asarray(document_embeddings, dtype=np.float32)
    if documents.ndim != 2 or documents.shape[1] != query.shape[0]:
        raise ValueError("Query and document embedding dimensions do not match")
    top_k = min(max(int(k), 0), documents.shape[0])
    scores = documents @ query
    indices = np.argsort(-scores, kind="stable")[:top_k]
    return [int(index) for index in indices], [float(scores[index]) for index in indices]
