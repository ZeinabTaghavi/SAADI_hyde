"""Standalone retriever registry used by the HyDE benchmark runner.

The implementations in this module deliberately do not import the parent SAADI
package.  They mirror the retriever families used by the main project, except
for E5, which is intentionally excluded from this standalone experiment.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RetrieverSpec:
    name: str
    label: str
    kind: str
    model_name: str | None = None
    pooling: str = "mean"
    normalize_embeddings: bool = True
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = 64
    max_length: int = 512
    trust_remote_code: bool = False
    use_safetensors: bool | None = None
    score_fusion: str | None = None


RETRIEVER_SPECS: dict[str, RetrieverSpec] = {
    "bm25": RetrieverSpec(
        name="bm25",
        label="BM25",
        kind="lexical",
        score_fusion="minmax_mean",
    ),
    "contriever": RetrieverSpec(
        name="contriever",
        label="Contriever",
        kind="dense",
        model_name="facebook/contriever",
        pooling="mean",
        batch_size=128,
        max_length=512,
        use_safetensors=False,
    ),
    "bge_m3": RetrieverSpec(
        name="bge_m3",
        label="BGE-M3",
        kind="dense",
        model_name="BAAI/bge-m3",
        pooling="cls",
        batch_size=16,
        max_length=8192,
    ),
    "qwen": RetrieverSpec(
        name="qwen",
        label="Qwen",
        kind="dense",
        model_name="Qwen/Qwen3-Embedding-8B",
        pooling="last_token",
        batch_size=8,
        max_length=512,
        trust_remote_code=True,
    ),
    "jina": RetrieverSpec(
        name="jina",
        label="Jina",
        kind="dense",
        model_name="jinaai/jina-embeddings-v3",
        pooling="mean",
        batch_size=32,
        max_length=512,
        trust_remote_code=True,
    ),
}

RETRIEVER_ALIASES = {
    "bge": "bge_m3",
    "bge-m3": "bge_m3",
    "bge_m3_embedding": "bge_m3",
    "qwen_embedding": "qwen",
    "qwen-embedding": "qwen",
    "jina_embedding": "jina",
    "jina-embedding": "jina",
    "bm25s": "bm25",
}


def normalize_retriever_name(value: str | None) -> str:
    name = str(value or "contriever").strip().lower()
    name = RETRIEVER_ALIASES.get(name, name)
    if name not in RETRIEVER_SPECS:
        options = ", ".join(RETRIEVER_SPECS)
        raise ValueError(f"Unsupported standalone HyDE retriever={value!r}. Expected one of: {options}")
    return name


def retriever_spec(name: str, config: dict[str, Any] | None = None) -> RetrieverSpec:
    normalized = normalize_retriever_name(name)
    spec = RETRIEVER_SPECS[normalized]
    retrieval_cfg = dict((config or {}).get("retrieval", {}) or {})
    overrides = dict((retrieval_cfg.get("retriever_configs", {}) or {}).get(normalized, {}) or {})
    if normalized == "contriever":
        # Preserve the original standalone config format.
        overrides = {**dict(retrieval_cfg.get("encoder", {}) or {}), **overrides}
    allowed = {
        "model_name",
        "pooling",
        "normalize_embeddings",
        "query_prefix",
        "document_prefix",
        "batch_size",
        "max_length",
        "trust_remote_code",
        "use_safetensors",
        "score_fusion",
    }
    values = {key: value for key, value in overrides.items() if key in allowed and value is not None}
    return replace(spec, **values)


def retriever_metadata(spec: RetrieverSpec) -> dict[str, Any]:
    return asdict(spec)


def _offline_mode() -> bool:
    return os.getenv("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


class DenseRetrieverEncoder:
    """Generic normalized Hugging Face encoder with model-specific pooling."""

    def __init__(
        self,
        spec: RetrieverSpec,
        *,
        device: str | None = None,
        device_map: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if spec.kind != "dense" or not spec.model_name:
            raise ValueError(f"DenseRetrieverEncoder requires a dense model spec, got {spec}")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Dense HyDE retrievers require torch and transformers.") from exc

        self.torch = torch
        self.spec = spec
        self.retriever_name = spec.name
        self.model_name = str(spec.model_name)
        self.pooling = str(spec.pooling)
        self.normalize_embeddings = bool(spec.normalize_embeddings)
        self.query_prefix = str(spec.query_prefix)
        self.document_prefix = str(spec.document_prefix)
        self.batch_size = int(spec.batch_size)
        self.max_length = int(spec.max_length)
        self.cache_dir = cache_dir

        env_prefix = f"HYDE_{spec.name.upper()}"
        requested_device = (
            device
            or os.getenv(f"{env_prefix}_DEVICE")
            or os.getenv("HYDE_RETRIEVER_DEVICE")
            or None
        )
        requested_device_map = (
            device_map
            or os.getenv(f"{env_prefix}_DEVICE_MAP")
            or os.getenv("HYDE_RETRIEVER_DEVICE_MAP")
            or ""
        ).strip()
        if requested_device_map:
            self.device = "cpu"
            self.device_map = requested_device_map
        else:
            resolved = str(requested_device or "").strip()
            if not resolved:
                resolved = "cuda" if torch.cuda.is_available() else "cpu"
            if resolved.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"Retriever device {resolved!r} requested but CUDA is unavailable")
            self.device = resolved
            self.device_map = ""

        common: dict[str, Any] = {
            "cache_dir": cache_dir,
            "local_files_only": _offline_mode(),
            "token": False,
            "trust_remote_code": bool(spec.trust_remote_code),
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **common)
        model_kwargs = dict(common)
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        if spec.use_safetensors is not None:
            model_kwargs["use_safetensors"] = bool(spec.use_safetensors)
        self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        if not self.device_map:
            self.model.to(self.device)
        self.model.eval()

    def _input_device(self) -> Any:
        try:
            embeddings = self.model.get_input_embeddings()
            weight = getattr(embeddings, "weight", None)
            device = getattr(weight, "device", None)
            if device is not None and str(device) != "meta":
                return device
        except (AttributeError, NotImplementedError):
            pass
        for parameter in self.model.parameters():
            if str(parameter.device) != "meta":
                return parameter.device
        return self.device

    def _prepare(self, texts: list[str], *, text_kind: str) -> list[str]:
        prefix = self.query_prefix if text_kind == "query" else self.document_prefix
        return [f"{prefix}{str(text or '')}" for text in texts]

    def _pool(self, output: Any, attention_mask: Any) -> Any:
        if getattr(output, "sentence_embeddings", None) is not None:
            pooled = output.sentence_embeddings
        elif getattr(output, "embeddings", None) is not None:
            pooled = output.embeddings
        else:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, tuple) and output:
                hidden = output[0]
            if hidden is None:
                raise RuntimeError(f"Could not extract embeddings from model={self.model_name}")
            if self.pooling == "cls":
                pooled = hidden[:, 0]
            elif self.pooling == "last_token":
                lengths = attention_mask.sum(dim=1).clamp(min=1).long() - 1
                indices = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
                pooled = hidden.gather(dim=1, index=indices).squeeze(1)
            else:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if self.normalize_embeddings:
            pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled

    def _encode(self, texts: list[str], *, text_kind: str) -> np.ndarray:
        if not texts:
            hidden = int(getattr(self.model.config, "hidden_size", 0) or 0)
            return np.empty((0, hidden), dtype=np.float32)
        prepared = self._prepare(texts, text_kind=text_kind)
        batches: list[np.ndarray] = []
        input_device = self._input_device()
        with self.torch.inference_mode():
            for start in range(0, len(prepared), self.batch_size):
                encoded = self.tokenizer(
                    prepared[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(input_device) for key, value in encoded.items()}
                output = self.model(**encoded)
                pooled = self._pool(output, encoded["attention_mask"])
                batches.append(pooled.detach().cpu().float().numpy())
        return np.concatenate(batches, axis=0)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, text_kind="query")

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, text_kind="document")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Backward-compatible query encoder used by older tests/callers."""

        return self.encode_queries(texts)


def build_dense_encoder(
    name: str,
    config: dict[str, Any],
    *,
    device: str | None = None,
    device_map: str | None = None,
    cache_dir: str | None = None,
) -> DenseRetrieverEncoder:
    return DenseRetrieverEncoder(
        retriever_spec(name, config),
        device=device,
        device_map=device_map,
        cache_dir=cache_dir,
    )


def _normalize_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    return [_normalize_token(token) for token in re.findall(r"[A-Za-z0-9]+", str(text or "").lower())]


class BM25Index:
    """Native deterministic BM25 scorer with HyDE score-vector fusion."""

    def __init__(self, corpus: list[str]) -> None:
        self.corpus = list(corpus)
        self.document_tokens = [_tokenize(text) for text in self.corpus]
        self.document_lengths = [len(tokens) for tokens in self.document_tokens]
        self.average_length = sum(self.document_lengths) / max(1, len(self.document_lengths))
        self.term_frequencies = [Counter(tokens) for tokens in self.document_tokens]
        self.document_frequency: defaultdict[str, int] = defaultdict(int)
        for frequencies in self.term_frequencies:
            for term in frequencies:
                self.document_frequency[term] += 1

    def scores(self, query: str) -> np.ndarray:
        terms = _tokenize(query)
        scores = np.zeros(len(self.corpus), dtype=np.float32)
        if not terms:
            return scores
        total = len(self.corpus)
        k1 = 1.5
        b = 0.75
        for index, frequencies in enumerate(self.term_frequencies):
            document_length = self.document_lengths[index] or 1
            value = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency.get(term, 0)
                inverse_frequency = math.log(
                    1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + k1 * (
                    1.0 - b + b * document_length / max(1e-9, self.average_length)
                )
                value += inverse_frequency * (frequency * (k1 + 1.0)) / denominator
            scores[index] = value
        return scores

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        minimum = float(values.min()) if values.size else 0.0
        maximum = float(values.max()) if values.size else 0.0
        if maximum <= minimum:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - minimum) / (maximum - minimum)).astype(np.float32)

    def hyde_scores(self, query_variants: list[str], *, fusion: str = "minmax_mean") -> np.ndarray:
        if fusion != "minmax_mean":
            raise ValueError(f"Unsupported BM25 HyDE score fusion={fusion!r}")
        if not query_variants:
            return np.zeros(len(self.corpus), dtype=np.float32)
        normalized = [self._minmax(self.scores(query)) for query in query_variants]
        return np.mean(np.stack(normalized, axis=0), axis=0, dtype=np.float32)


def rank_score_vector(scores: np.ndarray, *, k: int) -> tuple[list[int], list[float]]:
    vector = np.asarray(scores, dtype=np.float32).reshape(-1)
    top_k = min(max(int(k), 0), len(vector))
    indices = np.argsort(-vector, kind="stable")[:top_k]
    return [int(index) for index in indices], [float(vector[index]) for index in indices]


__all__ = [
    "BM25Index",
    "DenseRetrieverEncoder",
    "RETRIEVER_SPECS",
    "RetrieverSpec",
    "build_dense_encoder",
    "normalize_retriever_name",
    "rank_score_vector",
    "retriever_metadata",
    "retriever_spec",
]
