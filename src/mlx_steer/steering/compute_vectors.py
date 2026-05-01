from __future__ import annotations

from typing import Any


def compute_steering_vectors(activations: Any) -> Any:
    """Compute steering vectors from extracted contrastive activations.

    Expected shape/structure depends on the extraction implementation; callers
    should treat this as an evolving API until the Week 9 vector pipeline lands.
    """

    raise NotImplementedError(
        "Steering vector computation is not yet implemented in this repo snapshot."
    )

