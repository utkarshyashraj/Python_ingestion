"""Pluggable semantic representation layer.

Semantic similarity is treated as *one signal among several* -- never the sole
grouping mechanism. The default backend is dependency-free and deterministic:
it hashes word unigrams/bigrams and character n-grams into a fixed-width vector
with sub-linear term weighting and L2 normalisation. This captures a useful
amount of lexical/sub-word relatedness (e.g. "authentication" ~ "auth",
"login") without requiring heavyweight ML dependencies.

For higher-quality synonym awareness (e.g. "MFA" ~ "multi-factor login") a
transformer backend can be plugged in by setting
``EngineConfig.embedding_backend = "sentence-transformers"`` -- it is used only
when the optional package is available.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Optional, Sequence

import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _hash(token: str, dim: int) -> int:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % dim


def _sign(token: str) -> float:
    h = hashlib.blake2b((token + "#sign").encode("utf-8"), digest_size=1).digest()
    return 1.0 if (h[0] & 1) else -1.0


class EmbeddingBackend:
    """Interface for semantic backends."""

    dim: int

    def fit(self, texts: Sequence[str]) -> None:
        """Optional corpus fit (e.g. to learn IDF weights). Default: no-op."""
        return None

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class HashingEmbeddingBackend(EmbeddingBackend):
    """Deterministic hashing vectorizer over words + char n-grams.

    An optional corpus fit learns per-index inverse document frequency (IDF)
    weights so that ubiquitous boilerplate tokens are down-weighted and
    distinctive, topical terms dominate the similarity -- a standard, generic
    technique that markedly improves grouping without any business knowledge.
    """

    def __init__(self, dim: int = 512, char_ngrams: Sequence[int] = (3, 4)) -> None:
        self.dim = dim
        self.char_ngrams = tuple(char_ngrams)
        self.idf: Optional[np.ndarray] = None

    def _features(self, text: str) -> List[str]:
        tokens = tokenize(text)
        feats: List[str] = []
        # Word unigrams and bigrams.
        feats.extend(f"w:{t}" for t in tokens)
        feats.extend(f"b:{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
        # Character n-grams over the collapsed alphanumeric string (sub-word).
        collapsed = "".join(tokens)
        for n in self.char_ngrams:
            if len(collapsed) >= n:
                feats.extend(f"c{n}:{collapsed[i:i + n]}" for i in range(len(collapsed) - n + 1))
        return feats

    def fit(self, texts: Sequence[str]) -> None:
        n_docs = len(texts)
        if n_docs == 0:
            return
        df = np.zeros(self.dim, dtype=np.float32)
        for text in texts:
            seen = {_hash(f, self.dim) for f in self._features(text or "")}
            for idx in seen:
                df[idx] += 1.0
        self.idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            feats = self._features(text or "")
            if not feats:
                continue
            counts: dict = {}
            for f in feats:
                counts[f] = counts.get(f, 0) + 1
            for f, c in counts.items():
                idx = _hash(f, self.dim)
                weight = _sign(f) * (1.0 + math.log(c))
                if self.idf is not None:
                    weight *= self.idf[idx]
                out[row, idx] += weight
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class SentenceTransformerBackend(EmbeddingBackend):
    """Optional transformer backend, used only if the package is installed."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # noqa: F401

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def build_backend(name: str, dim: int, model_name: str) -> EmbeddingBackend:
    if name == "sentence-transformers":
        try:
            return SentenceTransformerBackend(model_name)
        except Exception:
            # Gracefully fall back so the engine always runs.
            return HashingEmbeddingBackend(dim=dim)
    return HashingEmbeddingBackend(dim=dim)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors (assumes possibly non-normalised)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))