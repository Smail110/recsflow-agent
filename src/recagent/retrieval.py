"""Deterministic local lexical retrieval primitives.

The index is intentionally small and immutable after construction.  It is an
adapter boundary: callers decide how catalog records are projected to text.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

_TOKEN = re.compile(r"[\W_]+", re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Casefold Unicode words/numbers, with Russian ё normalized to е."""
    return tuple(part for part in _TOKEN.sub(" ", text.casefold().replace("ё", "е")).split() if part)


@dataclass(frozen=True)
class SearchHit:
    item_id: str
    score: float
    rank: int = 0


class BM25Index:
    def __init__(self, documents: Mapping[str, str] | Iterable[tuple[str, str]], *, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        pairs = documents.items() if isinstance(documents, Mapping) else documents
        self.documents = {str(i): str(t) for i, t in pairs}
        self._tokens = {i: tokenize(t) for i, t in self.documents.items()}
        self._lengths = {i: len(ts) for i, ts in self._tokens.items()}
        self._avgdl = sum(self._lengths.values()) / len(self._lengths) if self._lengths else 0.0
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        for item_id, tokens in self._tokens.items():
            for term, count in Counter(tokens).items():
                self._postings[term][item_id] = count

    @property
    def average_document_length(self) -> float:
        return self._avgdl

    def search(self, query: str, limit: int = 100) -> list[SearchHit]:
        if limit <= 0 or not self.documents:
            return []
        scores: dict[str, float] = defaultdict(float)
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1.0 + (len(self.documents) - df + 0.5) / (df + 0.5))
            for item_id, tf in posting.items():
                dl = self._lengths[item_id]
                norm = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl) if self._avgdl else tf + self.k1
                scores[item_id] += idf * tf * (self.k1 + 1) / norm
        ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
        return [SearchHit(item_id=i, score=s, rank=n) for n, (i, s) in enumerate(ordered, 1)]

    def rebuild(self, documents: Mapping[str, str] | Iterable[tuple[str, str]]) -> BM25Index:
        return type(self)(documents, k1=self.k1, b=self.b)
