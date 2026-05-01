from __future__ import annotations

from typing import Any


def extract_contrastive_activations(
    model: Any,
    tokenizer: Any,
    *,
    positive_prompt: str,
    negative_prompt: str,
    layer_idx: int,
    max_tokens: int = 1,
) -> dict:
    """Extract activations for a contrastive prompt pair at a given layer.

    This module is a thin API placeholder so `mlx_steer` is importable and
    installable while the full steering pipeline is being finalized.
    """

    raise NotImplementedError(
        "Activation extraction is not yet implemented in this repo snapshot."
    )

