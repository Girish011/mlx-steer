from __future__ import annotations

from typing import Any

import mlx.core as mx

from mlx_steer.monitor.hidden_states import extract_hidden_states

def extract_contrastive_activations(
    model: Any,
    tokenizer: Any,
    *,
    positive_prompt: str,
    negative_prompt: str,
    layer_idx: int,
    max_tokens: int = 1,  # reserved for future token-by-token capture
) -> dict:
    """Extract activations for a contrastive prompt pair at a given layer.

    Current behavior (Week 9 implementation):
    - Runs a single forward pass for each prompt (no generation loop).
    - Captures the specified transformer layer output.
    - Returns pooled vectors suitable for simple steering vector computation.
    """

    _, pos_states = extract_hidden_states(model, tokenizer, positive_prompt, layer_idx=layer_idx)
    _, neg_states = extract_hidden_states(model, tokenizer, negative_prompt, layer_idx=layer_idx)

    # Normalize captured tensors to (seq_len, hidden_dim) for pooling.
    def _to_seq_hidden(x: Any) -> mx.array:
        if not isinstance(x, mx.array):
            x = mx.array(x)
        if len(x.shape) == 3:
            # (1, seq, hidden) -> (seq, hidden)
            return x[0]
        if len(x.shape) == 2:
            return x
        raise ValueError(f"Unexpected hidden state shape: {x.shape}")

    pos = _to_seq_hidden(pos_states)
    neg = _to_seq_hidden(neg_states)

    # Mean-pool over sequence dimension.
    pos_mean = mx.mean(pos, axis=0)
    neg_mean = mx.mean(neg, axis=0)

    return {
        "layer_idx": int(layer_idx),
        "positive_mean": pos_mean,
        "negative_mean": neg_mean,
        "positive_states": pos,
        "negative_states": neg,
        "delta": pos_mean - neg_mean,
    }

