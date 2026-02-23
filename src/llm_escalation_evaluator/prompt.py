from __future__ import annotations
from typing import List
from .data_types import Turn


SYSTEM_RUBRIC = """
You are grading a nurse–patient conversation for training.

Your task is to determine how the PATIENT'S emotional state changed
IMMEDIATELY AFTER the nurse's last statement.

Focus ONLY on the patient's emotional reaction from their perspective.

Classify the change in emotional state as one of:

Escalation
→ the patient becomes more emotionally distressed,
   tense, agitated, angry, resistant, or upset

De-Escalation
→ the patient becomes calmer,
   reassured, cooperative, or less emotionally activated

Neutral
→ no meaningful emotional change

Important:
- Judge from the patient’s perception, not nurse intention
- Emotional inertia applies: emotional change is gradual
- Ignore spelling mistakes or transcription errors
- Compare emotional state BEFORE vs AFTER the nurse speaks
- Choose the closest label even if change is small

Output EXACTLY one word:

Escalation
De-Escalation
Neutral
"""
# SYSTEM_RUBRIC = """\
# You are grading a nurse-patient conversation for training purposes.

# Goal:
# Estimate the patient's emotional state AFTER the nurse speaks,
# based on the patient's behaviour and emotional trajectory,
# not the nurse's intention.

# The escalation scale is continuous in [-1, +1].

# Use the following scale as reference. The descriptions are guides.
# Values between them are allowed.

# ESCALATION SCALE:

# -1.0 → completely calm, emotionally regulated, cooperative
# -0.9 → very calm, comfortable, trusting
# -0.8 → calm and settled
# -0.7 → relaxed but slightly concerned
# -0.6 → mild worry, still stable
# -0.5 → uneasy but cooperative
# -0.4 → noticeable distress
# -0.3 → anxious or frustrated
# -0.2 → tense, emotionally activated
# -0.1 → slightly unsettled

#  0.0 → neutral emotional control

# +0.1 → irritated
# +0.2 → frustrated
# +0.3 → clearly agitated
# +0.4 → emotionally elevated
# +0.5 → verbally upset
# +0.6 → angry or confrontational
# +0.7 → strongly agitated
# +0.8 → shouting or losing control
# +0.9 → aggressive or threatening
# +1.0 → fully escalated, panic, threats, refusing care

# Values between these points are acceptable.
# Small numeric differences are not important.

# Output fields:
# - turn_label: escalatory | deescalatory | neutral



# Important:
# - Judge from patient perception.
# - Emotional inertia applies: escalation usually changes gradually.
# - Do NOT treat spelling mistakes or transcription errors as emotional signals.

# """

# - nurse_impact: continuous impact in [-1,+1]
# - patient_escalation_level: estimated patient state AFTER the nurse line in [-1,+1]
# - confidence: [0,1]
# - context_alignment: yes | partial | no
#   (Did the nurse response directly address the patient's last message?)
# - sarcasm_detected: yes | no
#   (Did the nurse use sarcasm, mockery, or ironic tone?)

def format_history(turns: List[Turn], max_turns: int = 16) -> str:
    # keep last N turns
    recent = turns[-max_turns:]
    lines = []
    for t in recent:
        speaker = "PATIENT" if t.role == "patient" else "NURSE"
        lines.append(f"{speaker}: {t.text}")
    return "\n".join(lines)