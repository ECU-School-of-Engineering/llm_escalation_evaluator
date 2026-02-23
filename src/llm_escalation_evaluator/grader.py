# escalation_grader/grader.py
from __future__ import annotations
import json
import logging
from typing import List, Literal, Optional
from .data_types import Turn
from .prompt import SYSTEM_RUBRIC, format_history
from .schema import SCHEMA_BODY, SCHEMA_NAME
from .openai_client import OpenAIResponsesClient

logger = logging.getLogger(__name__)


class EscalationGrader:
    """
    Grades nurse lines for impact on patient escalation.
    """

    def __init__(
        self,
        client: Optional[OpenAIResponsesClient] = None,
        model: str = "gpt-4o-mini",
        max_turns: int = 16,
        temperature: float = 0.0,
        seed: int | None = None,
        debug: bool = False,
    ):
        self.client = client or OpenAIResponsesClient()
        self.model = model
        self.max_turns = max_turns
        self.temperature = temperature
        self.seed = seed
        if debug:
            logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
            logger.setLevel(logging.DEBUG)

    def evaluate_nurse_turn(
        self,
        *,
        history: List[Turn],
        nurse_line: str,
        context: Optional[dict] = None,
    ) -> Literal["escalatory", "deescalatory", "neutral"]:
        payload = {
            "conversation_history": format_history(history, max_turns=self.max_turns),
            "current_nurse_line": nurse_line,
            "context": context or {},
        }

        logger.debug("=== LLM INPUT ===\n[SYSTEM]\n%s\n[USER]\n%s", SYSTEM_RUBRIC, json.dumps(payload, indent=2, ensure_ascii=False))

        raw = self.client.grade(
            model=self.model,
            system=SYSTEM_RUBRIC,
            user_payload=payload,
            schema_name=SCHEMA_NAME,
            schema_body=SCHEMA_BODY,
            temperature=self.temperature,
            seed=self.seed,
        )

        logger.debug("=== LLM OUTPUT ===\n%s", raw)

        data = json.loads(raw)
        return data["turn_label"]
