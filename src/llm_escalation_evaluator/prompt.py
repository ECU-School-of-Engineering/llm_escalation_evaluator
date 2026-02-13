from __future__ import annotations
from typing import List
from .data_types import Turn

SYSTEM_RUBRIC = """\
You are grading a nurse-patient conversation for training purposes.

Goal:
Estimate how much the NURSE's current line affects the PATIENT's escalation level.

Escalation scale is continuous in [-1, +1]:
+1.0 = extremely escalated (shouting, threatening, panic, refusing care)
 0.0 = neutral/controlled
-1.0 = very calm, regulated, cooperative

Output fields:
- turn_label: escalatory | deescalatory | neutral
- nurse_impact: continuous impact in [-1,+1] (negative calms, positive escalates)
- patient_escalation_level: estimated patient state AFTER the nurse line in [-1,+1]
- confidence: [0,1]
- signals: short tags that justify (e.g., validation, threat, command, sarcasm, autonomy, plan)
- rationale: 1–2 sentences
- coaching: 0–2 rewritten nurse lines that would be more de-escalatory (keep clinically safe)

Important:
- Judge from patient perception, not whether the nurse is clinically correct.
- Emotional inertia: large jumps (>0.4) should be rare unless there is a strong trigger (threat, humiliation, strong validation + clear plan).
- If nurse must set a boundary or refuse a request, it can still be de-escalatory if respectful + explains next step + preserves dignity.
"""

def format_history(turns: List[Turn], max_turns: int = 16) -> str:
    # keep last N turns
    recent = turns[-max_turns:]
    lines = []
    for t in recent:
        speaker = "PATIENT" if t.role == "patient" else "NURSE"
        lines.append(f"{speaker}: {t.text}")
    return "\n".join(lines)