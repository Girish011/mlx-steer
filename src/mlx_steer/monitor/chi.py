from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlx_steer.monitor.calibration import get_calibration_for_model
from mlx_steer.monitor.goal_drift import compute_goal_drift
from mlx_steer.monitor.repetition import compute_repetition_score


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _rolling_health(value: float, history: deque[float]) -> float:
    """Higher when ``value`` is close to recent baseline (mean + MAD scale)."""

    if len(history) == 0:
        return 1.0
    mu = sum(history) / len(history)
    mad = sum(abs(x - mu) for x in history) / len(history) + 1e-6
    return 1.0 - _clamp01(abs(value - mu) / (3.0 * mad))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class CHIMonitor:
    """Cognitive Health Index monitor.

    The monitor reports health; policy decisions (what to do when degraded)
    are intentionally left to the caller.

    **Legacy mode** (default): ``model_key`` and ``calibration`` unset — same
    combination as before: repetition + goal drift with exponential smoothing.

    **Calibrated mode**: pass ``model_key`` (e.g. ``"qwen2.5-coder-14b"``) or a
    ``calibration`` dict to enable cold-start CHI, rolling-baseline drift
    normalization on optional entropy and hidden-state cosine channels, and
    model-specific ``degradation_threshold``.
    """

    degraded_threshold: float = 0.55
    smoothing: float = 0.2
    model_key: str | None = None
    calibration: Mapping[str, Any] | None = None

    _score: float = field(default=1.0, init=False)
    _turn: int = field(default=0, init=False)
    _prev_hidden: np.ndarray | None = field(default=None, init=False)
    _calibrated: bool = field(default=False, init=False)
    _cold_start_turns: int = field(default=0, init=False)
    _cold_start_chi: float = field(default=0.85, init=False)
    _baseline_window: int = field(default=3, init=False)
    _entropy_hist: deque[float] = field(init=False)
    _repetition_hist: deque[float] = field(init=False)
    _goal_hist: deque[float] = field(init=False)
    _cos_hist: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self._calibrated = self.model_key is not None or self.calibration is not None
        if self._calibrated:
            cal = get_calibration_for_model(self.model_key)
            if self.calibration:
                cal = {**cal, **dict(self.calibration)}
            self._cold_start_turns = int(cal["cold_start_turns"])
            self._cold_start_chi = float(cal["cold_start_chi"])
            self._baseline_window = max(1, int(cal["baseline_window"]))
            if "degradation_threshold" in cal:
                self.degraded_threshold = float(cal["degradation_threshold"])
            bw = self._baseline_window
            self._entropy_hist = deque(maxlen=bw)
            self._repetition_hist = deque(maxlen=bw)
            self._goal_hist = deque(maxlen=bw)
            self._cos_hist = deque(maxlen=bw)
        else:
            self._entropy_hist = deque(maxlen=1)
            self._repetition_hist = deque(maxlen=1)
            self._goal_hist = deque(maxlen=1)
            self._cos_hist = deque(maxlen=1)

    @property
    def degradation_threshold(self) -> float:
        """Alias for ``degraded_threshold`` (TASK-055 naming)."""

        return self.degraded_threshold

    def update(
        self,
        response: str,
        context: dict | None = None,
        *,
        mean_token_entropy: float | None = None,
        hidden_pooled: Sequence[float] | np.ndarray | None = None,
    ) -> dict:
        context = context or {}
        goal = context.get("goal")

        repetition = compute_repetition_score(response)
        goal_drift = compute_goal_drift(response, goal=goal)

        repetition_health = 1.0 - _clamp01(repetition)
        goal_health = 1.0 - _clamp01(goal_drift)

        cosine_similarity: float | None = None
        hidden_drift: float | None = None
        if hidden_pooled is not None:
            v = np.asarray(hidden_pooled, dtype=np.float64).ravel()
            if self._prev_hidden is not None and v.size == self._prev_hidden.size:
                cosine_similarity = _cosine_similarity(v, self._prev_hidden)
                hidden_drift = 1.0 - _clamp01(cosine_similarity)
            self._prev_hidden = v.copy()

        if not self._calibrated:
            raw_score = _clamp01(0.6 * repetition_health + 0.4 * goal_health)
            self._score = _clamp01((1.0 - self.smoothing) * self._score + self.smoothing * raw_score)
            self._turn += 1
            return {
                "chi": self._score,
                "signals": {
                    "repetition": repetition,
                    "goal_drift": goal_drift,
                },
                "raw": {
                    "raw_score": raw_score,
                },
            }

        # --- Calibrated path: cold start + rolling drift on all channels ---
        ent_health: float | None = None
        if mean_token_entropy is not None:
            e = float(mean_token_entropy)
            ent_intrinsic = 1.0 - _clamp01(e / 16.0)
            ent_roll = _rolling_health(e, self._entropy_hist)
            ent_health = _clamp01(0.5 * ent_intrinsic + 0.5 * ent_roll)

        rep_roll = _rolling_health(repetition, self._repetition_hist)
        rep_health_c = _clamp01(0.5 * repetition_health + 0.5 * rep_roll)

        goal_roll = _rolling_health(goal_drift, self._goal_hist)
        goal_health_c = _clamp01(0.5 * goal_health + 0.5 * goal_roll)

        cos_health: float | None = None
        if cosine_similarity is not None and hidden_drift is not None:
            cos_roll = _rolling_health(cosine_similarity, self._cos_hist)
            cos_intrinsic = 1.0 - _clamp01(hidden_drift)
            cos_health = _clamp01(0.5 * cos_intrinsic + 0.5 * cos_roll)

        parts: list[float] = []
        if ent_health is not None:
            parts.append(ent_health)
        parts.extend([rep_health_c, goal_health_c])
        if cos_health is not None:
            parts.append(cos_health)

        if self._turn < self._cold_start_turns:
            chi_raw = self._cold_start_chi
        else:
            chi_raw = _clamp01(sum(parts) / max(len(parts), 1))

        self._score = _clamp01((1.0 - self.smoothing) * self._score + self.smoothing * chi_raw)

        if mean_token_entropy is not None:
            self._entropy_hist.append(float(mean_token_entropy))
        self._repetition_hist.append(repetition)
        self._goal_hist.append(goal_drift)
        if cosine_similarity is not None:
            self._cos_hist.append(cosine_similarity)

        self._turn += 1

        signals: dict[str, Any] = {
            "entropy": mean_token_entropy,
            "repetition": repetition,
            "goal_drift": goal_drift,
            "cosine_similarity": cosine_similarity,
        }
        return {
            "chi": self._score,
            "signals": signals,
            "raw": {
                "chi_raw": chi_raw,
                "repetition_health": rep_health_c,
                "goal_health": goal_health_c,
                "entropy_health": ent_health,
                "cosine_health": cos_health,
                "hidden_drift": hidden_drift,
            },
        }

    @property
    def score(self) -> float:
        return self._score

    @property
    def is_degraded(self) -> bool:
        return self._score < self.degraded_threshold
