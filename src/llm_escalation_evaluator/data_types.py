from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, List, Optional
from datetime import datetime

Role = Literal["patient", "nurse"]

@dataclass
class Turn:
    role: Role
    text: str
    ts: Optional[datetime] = None

@dataclass
class ConversationState:
    escalation: float = 0.0  # running patient escalation level in [-1, +1]
    # optional extra signals you might incorporate later (HUME, BERT, etc.)
    metadata: dict = field(default_factory=dict)

@dataclass
class GradeResult:
    turn_label: Literal["escalatory", "deescalatory", "neutral"]
    nurse_impact: float                 # in [-1, +1] (continuous)
    patient_escalation_level: float     # in [-1, +1] (continuous)
    confidence: float                   # in [0, 1]
    # signals: List[str]
    # rationale: str
    # coaching: List[str]                 # suggested better nurse lines