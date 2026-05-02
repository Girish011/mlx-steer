"""Per-model defaults for CHI rolling baseline and degradation thresholds."""

from __future__ import annotations

from typing import Any

MODEL_CALIBRATION: dict[str, dict[str, Any]] = {
    "gemma-4-e4b-it": {
        "cold_start_turns": 2,
        "cold_start_chi": 0.85,
        "baseline_window": 3,
        "degradation_threshold": 0.82,
    },
    "qwen2.5-coder-14b": {
        "cold_start_turns": 2,
        "cold_start_chi": 0.85,
        "baseline_window": 3,
        "degradation_threshold": 0.85,
    },
}


def get_calibration_for_model(model_key: str | None) -> dict[str, Any]:
    """Return a fresh dict of calibration parameters for ``model_key``.

    Unknown keys fall back to ``qwen2.5-coder-14b`` so callers always get a
    complete mapping. When ``model_key`` is None, the same fallback applies
    (for partial overrides merged in :class:`CHIMonitor`).
    """

    key = model_key if model_key in MODEL_CALIBRATION else "qwen2.5-coder-14b"
    return dict(MODEL_CALIBRATION[key])
