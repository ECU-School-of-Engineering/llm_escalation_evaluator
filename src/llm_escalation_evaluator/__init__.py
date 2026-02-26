# from .data_types import Turn, ConversationState, GradeResult
# from .grader import EscalationGrader

# __all__ = [
#     "Turn",
#     "ConversationState",
#     "GradeResult",
#     "EscalationGrader",
# ]

from .data_types import Turn, ConversationState, GradeResult
from .grader import EscalationGrader

# One grader per profile, created once and reused across all calls.
# Safe because EscalationGrader is stateless — history is passed in per call.
_graders: dict[str, EscalationGrader] = {}


def _get_grader(profile: str) -> EscalationGrader:
    if profile not in _graders:
        _graders[profile] = EscalationGrader(profile=profile)
    return _graders[profile]


def evaluate(
    profile: str,
    list_history: list,
    user_sentence: str,
) -> str:
    """
    Single-call entry point for the backend.

    profile:       'barry' or 'maddie'
    list_history:  conversation so far as OpenAI-style dicts
                   [{"role": "user", "content": "..."}, ...]
    user_sentence: the nurse's latest line (not yet in history)
    """
    return _get_grader(profile).evaluate_nurse_turn(
        nurse_line=user_sentence, history=list_history
    )


__all__ = ["Turn", "ConversationState", "GradeResult", "EscalationGrader", "evaluate"]