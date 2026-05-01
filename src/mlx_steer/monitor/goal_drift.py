from __future__ import annotations


def compute_goal_drift(response: str, goal: str | None = None) -> float:
    """Heuristic goal-drift score in [0, 1] (higher = more drift).

    This is intentionally lightweight (no model required). If `goal` is
    provided, we approximate drift by overlap between goal keywords and the
    response.
    """

    if not response.strip():
        return 1.0
    if not goal or not goal.strip():
        return 0.0

    goal_tokens = {t.lower() for t in goal.split() if t.strip()}
    resp_tokens = {t.lower() for t in response.split() if t.strip()}
    if not goal_tokens:
        return 0.0

    overlap = len(goal_tokens & resp_tokens) / len(goal_tokens)
    # If overlap is low, drift is high.
    return float(max(0.0, min(1.0, 1.0 - overlap)))

