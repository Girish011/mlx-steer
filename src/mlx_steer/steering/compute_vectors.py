from __future__ import annotations

from typing import Any, Iterable

import mlx.core as mx


def compute_steering_vectors(
    activations: Any,
    *,
    normalize: bool = True,
    eps: float = 1e-8,
) -> mx.array:
    """Compute a steering vector from extracted contrastive activations.

    Supported inputs:
    - A single activation dict from `extract_contrastive_activations()` (uses `delta`)
    - An iterable/list of such dicts (averages deltas)
    """

    if isinstance(activations, dict):
        deltas = [activations["delta"]]
    else:
        deltas = [a["delta"] for a in activations]

    deltas = [d if isinstance(d, mx.array) else mx.array(d) for d in deltas]
    v = mx.mean(mx.stack(deltas, axis=0), axis=0)

    if normalize:
        denom = mx.sqrt(mx.sum(v * v)) + mx.array(eps)
        v = v / denom
    return v

