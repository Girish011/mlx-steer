from __future__ import annotations

from dataclasses import dataclass, field

from mlx_steer.monitor.goal_drift import compute_goal_drift
from mlx_steer.monitor.repetition import compute_repetition_score


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class CHIMonitor:
    """Cognitive Health Index monitor.

    The monitor reports health; policy decisions (what to do when degraded)
    are intentionally left to the caller.
    """

    degraded_threshold: float = 0.55
    smoothing: float = 0.2
    _score: float = field(default=1.0, init=False)

    def update(self, response: str, context: dict | None = None) -> dict:
        context = context or {}
        goal = context.get("goal")

        repetition = compute_repetition_score(response)
        goal_drift = compute_goal_drift(response, goal=goal)

        # Convert "bad" signals into "good" health components.
        repetition_health = 1.0 - _clamp01(repetition)
        goal_health = 1.0 - _clamp01(goal_drift)

        raw_score = _clamp01(0.6 * repetition_health + 0.4 * goal_health)
        self._score = _clamp01((1.0 - self.smoothing) * self._score + self.smoothing * raw_score)

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

    @property
    def score(self) -> float:
        return self._score

    @property
    def is_degraded(self) -> bool:
        return self._score < self.degraded_threshold

