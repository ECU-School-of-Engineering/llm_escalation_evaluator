# from .data_types import Turn, ConversationState, GradeResult
# from .grader import EscalationGrader

# __all__ = [
#     "Turn",
#     "ConversationState",
#     "GradeResult",
#     "EscalationGrader",
# ]

import logging
from .data_types import Turn, ConversationState, GradeResult
from .grader import EscalationGrader

logger = logging.getLogger(__name__)

# Keyed by (profile, config_path) so the same profile can use different configs.
# Safe because EscalationGrader is stateless — history is passed in per call.
_graders: dict[tuple[str, str | None], EscalationGrader] = {}


def configure(profile: str, config_path: str | None = None, **kwargs) -> None:
    """
    Pre-initialise the grader for a profile with a custom config file.
    Call once at startup before evaluate().

    profile:     profile name that must exist in the config file
    config_path: path to a custom YAML config file; uses the built-in
                 grader_config.yaml if omitted
    kwargs:      forwarded to EscalationGrader (model, temperature, etc.)
    """
    grader = EscalationGrader(profile=profile, config_path=config_path, **kwargs)
    _graders[(profile, config_path)] = grader
    # Also register under (profile, None) so evaluate() finds it without
    # an explicit config_path argument.
    _graders[(profile, None)] = grader


def _get_grader(profile: str, config_path: str | None = None) -> EscalationGrader:
    key = (profile, config_path)
    if key not in _graders:
        _graders[key] = EscalationGrader(profile=profile, config_path=config_path)
    return _graders[key]


def evaluate(
    profile: str,
    list_history: list,
    user_sentence: str,
    config_path: str | None = None,
) -> str:
    """
    Single-call entry point for the backend.

    profile:       'barry' or 'maddie' (or any profile in your config)
    list_history:  conversation so far as OpenAI-style dicts
                   [{"role": "user", "content": "..."}, ...]
    user_sentence: the nurse's latest line (not yet in history)
    config_path:   optional path to a custom YAML config; if you called
                   configure() first you can omit this
    """
    grader = _get_grader(profile, config_path)
    logger.debug(
        "evaluate() profile=%r  config=%s",
        profile,
        grader.config_path if grader.config_path is not None else "<default>",
    )
    return grader.evaluate_nurse_turn(
        nurse_line=user_sentence, history=list_history
    )


__all__ = ["Turn", "ConversationState", "GradeResult", "EscalationGrader", "configure", "evaluate"]