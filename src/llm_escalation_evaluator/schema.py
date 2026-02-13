SCHEMA_NAME = "nurse_patient_escalation_grade"

SCHEMA_BODY = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "turn_label": {"type": "string", "enum": ["escalatory", "deescalatory", "neutral"]},
        "nurse_impact": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "patient_escalation_level": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "signals": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        # "rationale": {"type": "string"},
        # "coaching": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 2},
    },
    "required": [
        "turn_label",
        "nurse_impact",
        "patient_escalation_level",
        "confidence",
        "signals",
        # "rationale",
        # "coaching",
    ],
}