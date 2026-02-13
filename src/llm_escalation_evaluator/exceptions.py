class EscalationGraderError(Exception):
    """Base exception for escalation_grader."""
    pass

class SchemaValidationError(EscalationGraderError):
    """Raised when model output does not match expected schema."""
    pass

class ModelResponseError(EscalationGraderError):
    """Raised when OpenAI response is missing/invalid."""
    pass