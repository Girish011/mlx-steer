from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SteeringWrapper:
    """Wrap a transformer layer and add a steering vector to its output."""

    layer: Any
    vector: Any

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self.layer(*args, **kwargs)
        # We intentionally keep this permissive: depending on the model, a layer
        # may return a tensor or a tuple (hidden, cache). We only steer the first.
        if isinstance(out, tuple) and len(out) >= 1:
            hidden = out[0] + self.vector
            return (hidden, *out[1:])
        return out + self.vector

    def __getattr__(self, name: str) -> Any:
        # Delegate to wrapped layer for model code that inspects attributes
        # (e.g., Gemma4 checks layer_type to choose global/local attention).
        return getattr(self.layer, name)


@dataclass
class SteeringEngine:
    """Minimal steering engine API for package import stability.

    The full implementation (vector loading, multi-layer steering, restore
    semantics, MLX-LM integration) is planned for Week 9 tasks.
    """

    model: Any
    layer_idx: int = 0
    vector: Optional[Any] = None
    _original_layer: Any = field(default=None, init=False, repr=False)
    _enabled: bool = field(default=False, init=False, repr=False)

    def set_vector(self, vector: Any) -> None:
        self.vector = vector

    def enable(self) -> None:
        if self._enabled:
            return
        if self.vector is None:
            raise ValueError("No steering vector set. Call set_vector() first.")

        # Common MLX-LM layout: model.model.layers
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            # mlx-vlm layout: model.language_model.model.layers
            lm = getattr(self.model, "language_model", None)
            layers = getattr(getattr(lm, "model", None), "layers", None)
        if layers is None:
            raise AttributeError(
                "Expected model.model.layers or model.language_model.model.layers to exist for steering."
            )

        self._original_layer = layers[self.layer_idx]
        layers[self.layer_idx] = SteeringWrapper(self._original_layer, self.vector)
        self._enabled = True

    def disable(self) -> None:
        if not self._enabled:
            return

        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            lm = getattr(self.model, "language_model", None)
            layers = getattr(getattr(lm, "model", None), "layers", None)
        if layers is None:
            return

        if self._original_layer is not None:
            layers[self.layer_idx] = self._original_layer
        self._enabled = False

