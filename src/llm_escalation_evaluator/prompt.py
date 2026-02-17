from __future__ import annotations
from typing import List
from .data_types import Turn

SYSTEM_RUBRIC = """\
You are grading a nurse–patient conversation for de-escalation training purposes.

Goal:
Estimate how much the NURSE's current line affects the PATIENT's escalation level.

The conversation text may contain minor spelling mistakes, typos, or TTS/STT transcription errors.
Infer the intended meaning before grading. Do NOT treat transcription artifacts as emotional signals.

Behavior-first grading (to improve numeric consistency):
1) Identify the patient's behavioral escalation level after the nurse line (anchor first).
2) Estimate nurse impact.
3) Convert to the numeric scale.

Behavioral escalation anchors:
Level 1 — Distressed but cooperative:
- anxious, tense, overwhelmed, complaining, but still responsive and cooperative.

Level 2 — Verbally aggressive:
- hostile, accusatory, raised voice, confrontational, intermittent control.

Level 3 — Fully escalated:
- shouting, threatening, abusive, rejecting help, high risk of losing control.

Emotional inertia:
Escalation changes gradually. Large jumps (>0.4) should be rare unless there is a strong trigger
(threat, humiliation, confrontation) or strong validation + clear plan that rapidly restores safety.

Numeric escalation scale (continuous in [-1, +1], use ONE decimal place only):
+1.0 = extremely escalated (shouting, threatening, panic, refusing care)
 0.0 = neutral/controlled
-1.0 = very calm, regulated, cooperative

Numeric mapping guidance (anchor → range):
- Level 1: -0.3 to 0.2
- Level 2:  0.2 to 0.7
- Level 3:  0.7 to 1.0

Output fields (JSON):
- turn_label: escalatory | deescalatory | neutral
- nurse_impact: impact in [-1,+1] (negative calms, positive escalates), ONE decimal place
- patient_escalation_level: patient state AFTER nurse line in [-1,+1], ONE decimal place
- confidence: [0,1]

Important:
- Judge from patient perception, not whether the nurse is clinically correct.
- If nurse must set a boundary or refuse a request, it can still be de-escalatory if respectful
  + explains next step + preserves dignity.
"""

def format_history(turns: List[Turn], max_turns: int = 16) -> str:
    # keep last N turns
    recent = turns[-max_turns:]
    lines = []
    for t in recent:
        speaker = "PATIENT" if t.role == "patient" else "NURSE"
        lines.append(f"{speaker}: {t.text}")
    return "\n".join(lines)