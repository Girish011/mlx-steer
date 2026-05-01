from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_repetition_score(text: str, *, ngram: int = 3) -> float:
    """Heuristic repetition score in [0, 1].

    Measures how concentrated the n-gram distribution is. 0 ~= diverse text,
    1 ~= highly repetitive loops.
    """

    tokens = text.strip().split()
    grams = _ngrams(tokens, ngram)
    if not grams:
        return 0.0

    counts = Counter(grams)
    total = sum(counts.values())
    if total == 0:
        return 0.0

    # Fraction of mass in the most common n-gram.
    return max(counts.values()) / total

