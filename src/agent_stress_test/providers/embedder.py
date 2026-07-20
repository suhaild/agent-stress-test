"""Offline bag-of-hashed-tokens text embedder for failure clustering.

Tokens hash into fixed buckets and the resulting vector is L2-normalized, so
lexically similar texts land on overlapping buckets and get high cosine
similarity — no model download or network needed.
"""

import hashlib
import math
import re

from agent_stress_test.ports import Embedder

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 256) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self._dim = dim

    def _bucket(self, token: str) -> int:
        # usedforsecurity=False: bucket assignment only, silences bandit B324.
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
