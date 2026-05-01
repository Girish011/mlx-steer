from __future__ import annotations

from typing import Any

import mlx.core as mx

from mlx_steer.utils.capture_layer import CaptureLayer


def _get_transformer_layers(model: Any) -> Any:
    # mlx-lm style: model.model.layers
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is not None:
        return layers

    # mlx-vlm style (Gemma 4 conversions often wrap a language_model):
    # model.language_model.model.layers
    lm = getattr(model, "language_model", None)
    layers = getattr(getattr(lm, "model", None), "layers", None)
    if layers is not None:
        return layers

    raise AttributeError("Could not find transformer layers (expected model.model.layers or model.language_model.model.layers).")


def extract_hidden_states(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    layer_idx: int = 0,
) -> tuple[str, Any]:
    """Run a single forward pass and capture a layer's output.

    Notes:
    - This captures the specified layer output for the *prompt* forward pass,
      not token-by-token generation.
    - MLX-LM commonly exposes transformer layers as `model.model.layers`.
    """

    layers = _get_transformer_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"layer_idx out of range: {layer_idx} (num_layers={len(layers)})")

    original = layers[layer_idx]
    wrapper = CaptureLayer(original)
    layers[layer_idx] = wrapper
    try:
        token_ids = None
        if hasattr(tokenizer, "encode"):
            token_ids = tokenizer.encode(prompt)
        elif hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "encode"):
            token_ids = tokenizer.tokenizer.encode(prompt)
        elif callable(tokenizer):
            encoded = tokenizer(prompt)
            if isinstance(encoded, dict) and "input_ids" in encoded:
                token_ids = encoded["input_ids"]
            else:
                token_ids = getattr(encoded, "input_ids", None)

        if token_ids is None:
            raise AttributeError("Tokenizer does not support encode() and did not return input_ids.")

        # Normalize to a 1D list[int]
        if isinstance(token_ids, mx.array):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        x = mx.array([token_ids], dtype=mx.int32)
        _ = model(x)
        captured = wrapper.captured
    finally:
        layers[layer_idx] = original

    if captured is None:
        raise RuntimeError("Failed to capture layer output (captured=None).")

    # Some models return tuples (hidden, cache/extra). If so, prefer hidden.
    if isinstance(captured, tuple) and len(captured) >= 1:
        captured = captured[0]

    return prompt, captured

