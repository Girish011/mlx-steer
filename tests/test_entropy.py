import math

import pytest


mx = pytest.importorskip("mlx.core")

from mlx_steer.monitor.entropy import compute_token_entropy  # noqa: E402


def test_entropy_uniform_binary_is_one_bit():
    # Two equal probs => 1 bit. Softmax([0,0]) = [0.5, 0.5]
    logits = mx.array([[0.0, 0.0]])
    h = compute_token_entropy(logits)
    assert h.shape == (1,)
    assert float(h[0]) == pytest.approx(1.0, abs=1e-5)


def test_entropy_peaked_is_near_zero():
    # Softmax([10, -10]) ~ [1, 0] => entropy ~ 0
    logits = mx.array([[10.0, -10.0]])
    h = compute_token_entropy(logits)
    assert float(h[0]) == pytest.approx(0.0, abs=1e-3)

