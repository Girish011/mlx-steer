from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CaptureLayer:
    """Wrap a transformer layer to capture its output.

    MLX doesn't provide PyTorch-style forward hooks. A common pattern is to
    temporarily replace an entry in `model.model.layers[]` with this wrapper,
    capture the output during a forward pass, and restore the original layer
    in a `finally` block.
    """

    layer: Any
    captured: Optional[Any] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self.layer(*args, **kwargs)
        self.captured = out
        return out

