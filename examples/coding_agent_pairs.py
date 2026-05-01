"""Reference contrastive pairs for coding-agent style steering.

This file intentionally lives in `examples/` (not imported by the library).
It's a holding place for agent-specific pairs like tool formatting, task focus,
and code coherence, for users who want to adapt `mlx-steer` into an agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContrastivePair:
    name: str
    positive: str
    negative: str


# Placeholders: fill with your own agent protocol / tool-format pairs.
TASK_FOCUS_PAIRS: list[ContrastivePair] = []
TOOL_FORMAT_PAIRS: list[ContrastivePair] = []
CODE_COHERENCE_PAIRS: list[ContrastivePair] = []

