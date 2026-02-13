# escalation_grader/grader.py
from __future__ import annotations
import json
from typing import List, Optional
from .data_types import Turn, ConversationState, GradeResult
from .prompt import SYSTEM_RUBRIC, format_history
from .schema import SCHEMA_BODY,SCHEMA_NAME
from .openai_client import OpenAIResponsesClient

def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

class EscalationGrader:
    """
    Grades nurse lines for impact on patient escalation.
    """

    def __init__(
        self,
        client: Optional[OpenAIResponsesClient] = None,
        model: str = "gpt-4o-mini",
        max_turns: int = 16,
        max_step: float = 0.35,  # inertia guardrail: cap per-turn movement
    ):
        self.client = client or OpenAIResponsesClient()
        self.model = model
        self.max_turns = max_turns
        self.max_step = max_step

    def evaluate_nurse_turn(
        self,
        *,
        history: List[Turn],
        nurse_line: str,
        state: ConversationState,
        context: Optional[dict] = None,  # e.g., ward type, scenario, constraints
    ) -> GradeResult:
        payload = {
            "previous_escalation_score": state.escalation,
            "conversation_history": format_history(history, max_turns=self.max_turns),
            "current_nurse_line": nurse_line,
            "context": context or {},
        }

        raw = self.client.grade(
                model=self.model,
                system=SYSTEM_RUBRIC,
                user_payload=payload,
                schema_name=SCHEMA_NAME,
                schema_body=SCHEMA_BODY,)
        

        data = json.loads(raw)

        # Optional extra guardrail: cap movement between previous and new.
        proposed = float(data["patient_escalation_level"])
        prev = float(state.escalation)
        delta = clamp(proposed - prev, -self.max_step, self.max_step)
        stabilized = clamp(prev + delta)

        # Keep nurse_impact consistent with stabilized result (optional)
        nurse_impact = float(data["nurse_impact"])
        # If you want nurse_impact to reflect the stabilized step:
        # nurse_impact = clamp(delta / max(self.max_step, 1e-6), -1.0, 1.0)

        return GradeResult(
            turn_label=data["turn_label"],
            nurse_impact=clamp(nurse_impact),
            patient_escalation_level=stabilized,
            confidence=float(data["confidence"]),
            # signals=list(data["signals"]),
            # rationale=data["rationale"],
            # coaching=list(data["coaching"]),
        )

    def apply(self, state: ConversationState, result: GradeResult) -> ConversationState:
        """
        Update and return a new ConversationState.
        """
        return ConversationState(
            escalation=result.patient_escalation_level
        )