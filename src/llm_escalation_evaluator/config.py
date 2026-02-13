from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GraderConfig:
    model: str = "gpt-5-mini"
    max_turns: int = 16
    max_step: float = 0.35  # caps per-turn score movement for inertia realism
    timeout_s: float = 30.0