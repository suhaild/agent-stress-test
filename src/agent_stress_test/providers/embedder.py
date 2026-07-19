"""Deterministic, offline text embedder for failure clustering.

A bag-of-hashed-tokens embedding: each token is hashed into one of a fixed
number of buckets, counts are accumulated, and the vector is L2-normalized.
Lexically similar texts land on overlapping buckets and so have high cosine
similarity. It needs no model download and no network, so it backs both the
tests and the default clustering path; a semantic embedder (provider or local
model) can replace it behind the ``Embedder`` port without touching callers.
"""

import hashlib
import math
import re

from agent_stress_test.ports import Embedder

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder(Embedder):
    """Hashing bag-of-words embedder (stdlib only, deterministic)."""

    def __init__(self, dim: int = 256) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self._dim = dim

    def _bucket(self, token: str) -> int:
        # Bucket assignment only, not a security use of the hash — silences
        # bandit's B324 without changing the digest or the buckets it maps to.
        digest = hashlib.sha1(token.encode("utf-8"), usedforsecurity=False).digest()
        return int.from_bytes(digest[:4], "big") % self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in _TOKEN.findall(text.lower()):
            vector[self._bucket(token)] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
