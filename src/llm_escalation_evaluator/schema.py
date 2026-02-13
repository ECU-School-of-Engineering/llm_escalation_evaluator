SCHEMA = {
    "name": "nurse_patient_escalation_grade",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "turn_label": {
                "type": "string",
                "enum": ["escalatory", "deescalatory", "neutral"]
            },
            "nurse_impact": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "How the nurse line shifts the patient (negative calms, positive escalates)."
            },
            "patient_escalation_level": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "Estimated patient escalation level AFTER this nurse line."
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "signals": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "rationale": {"type": "string"},
            "coaching": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 2,
                "description": "1–2 alternative nurse lines that would reduce escalation."
            },
        },
        "required": [
            "turn_label",
            "nurse_impact",
            "patient_escalation_level",
            "confidence",
            "signals",
            "rationale",
            "coaching",
        ],
    },
}