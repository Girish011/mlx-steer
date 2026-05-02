from __future__ import annotations

import numpy as np

from mlx_steer.monitor.calibration import MODEL_CALIBRATION, get_calibration_for_model
from mlx_steer.monitor.chi import CHIMonitor


def test_model_calibration_keys():
    assert "gemma-4-e4b-it" in MODEL_CALIBRATION
    assert "qwen2.5-coder-14b" in MODEL_CALIBRATION
    g = MODEL_CALIBRATION["gemma-4-e4b-it"]
    assert g["degradation_threshold"] == 0.82
    q = MODEL_CALIBRATION["qwen2.5-coder-14b"]
    assert q["degradation_threshold"] == 0.85


def test_get_calibration_fallback():
    cal = get_calibration_for_model("unknown-model-id")
    assert cal == get_calibration_for_model("qwen2.5-coder-14b")


def test_chimonitor_legacy_matches_simple_blend():
    m = CHIMonitor(smoothing=0.2)
    r0 = m.update("hello world task manager", {"goal": "task manager"})
    assert "chi" in r0 and "signals" in r0
    assert "entropy" not in r0["signals"] or r0["signals"].get("entropy") is None
    r1 = m.update("hello world task manager", {"goal": "task manager"})
    assert 0.0 <= r1["chi"] <= 1.0


def test_chimonitor_calibrated_degradation_threshold_alias():
    m = CHIMonitor(model_key="gemma-4-e4b-it")
    assert m.degradation_threshold == 0.82
    assert m.degraded_threshold == 0.82


def test_chimonitor_calibrated_entropy_and_cosine_channels():
    m = CHIMonitor(model_key="qwen2.5-coder-14b", smoothing=1.0)
    v0 = np.ones(8, dtype=np.float64)
    m.update("alpha beta gamma delta", {"goal": "alpha beta"}, mean_token_entropy=2.0, hidden_pooled=v0)
    m.update("alpha beta gamma delta epsilon", {"goal": "alpha beta"}, mean_token_entropy=2.1, hidden_pooled=v0 * 0.99)
    assert 0.0 <= m.score <= 1.0
