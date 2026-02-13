from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import json

from .data_types import Turn, ConversationState, GradeResult
from .grader import EscalationGrader
from .history import TurnBuffer
from .exceptions import SchemaValidationError
from .logging_utils import get_logger

logger = get_logger()

@dataclass
class TrainingSession:
    grader: EscalationGrader
    state: ConversationState = field(default_factory=ConversationState)
    buffer: TurnBuffer = field(default_factory=lambda: TurnBuffer(maxlen=60))
    context: dict = field(default_factory=dict)

    def add_patient(self, text: str) -> None:
        self.buffer.add(Turn(role="patient", text=text))

    def add_nurse_and_grade(self, text: str) -> GradeResult:
        # Add nurse line as a Turn, but grade using history prior + nurse_line (cleaner)
        history: List[Turn] = self.buffer.turns[:]  # snapshot

        result = self.grader.evaluate_nurse_turn(
            history=history,
            nurse_line=text,
            state=self.state,
            context=self.context,
        )

        # Apply updates and add the nurse turn to history
        self.state = self.grader.apply(self.state, result)
        self.buffer.add(Turn(role="nurse", text=text))

        logger.info(
            "Graded nurse line | label=%s impact=%.3f new_state=%.3f conf=%.2f",
            result.turn_label,
            result.nurse_impact,
            result.patient_escalation_level,
            result.confidence,
            # ",".join(result.signals),
        )
        return result

    def export_json(self) -> str:
        payload = {
            "state": {"escalation": self.state.escalation, "metadata": self.state.metadata},
            "context": self.context,
            "turns": [{"role": t.role, "text": t.text} for t in self.buffer.turns],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)