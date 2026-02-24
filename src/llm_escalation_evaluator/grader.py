from __future__ import annotations
import json
import logging
from typing import List, Literal, Optional
from .data_types import Turn
from .prompt import format_history
from .openai_client import OpenAIResponsesClient
from .yaml_config import load_config, build_schema_body

logger = logging.getLogger(__name__)


class EscalationGrader:
    """
    Grades nurse lines for impact on patient escalation.
    Settings are loaded from a YAML config file (default: grader_config.yaml
    packaged with the library). Any kwarg passed explicitly overrides the YAML.
    """

    def __init__(
        self,
        config_path: str | None = None,
        client: Optional[OpenAIResponsesClient] = None,
        model: str | None = None,
        max_turns: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        use_schema: bool | None = None,
        conversation_context_aware: bool | None = None,
        scene_aware: bool | None = None,
        debug: bool = False,
    ):
        cfg = load_config(config_path)

        self.model       = model       if model       is not None else cfg["model"]
        self.max_turns   = max_turns   if max_turns   is not None else cfg["max_turns"]
        self.temperature = temperature if temperature is not None else cfg["temperature"]
        self.seed        = seed        if seed        is not None else cfg.get("seed")
        self.use_schema  = use_schema  if use_schema  is not None else cfg["use_schema"]
        self.conversation_context_aware = conversation_context_aware if conversation_context_aware is not None else cfg.get("conversation_context_aware", True)
        self.scene_aware                = scene_aware                if scene_aware                is not None else cfg.get("scene_aware", True)

        self._system      = cfg["system_prompt"]
        self._schema_name = cfg["schema"]["name"]
        self._schema_body = build_schema_body(cfg["schema"]["fields"])

        self.client = client or OpenAIResponsesClient()

        if debug:
            log_level = logging.DEBUG
        else:
            log_level = getattr(logging, cfg.get("log_level", "WARNING").upper(), logging.WARNING)
        logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
        logger.setLevel(log_level)

    def evaluate_nurse_turn(
        self,
        *,
        nurse_line: str,
        history: List[Turn] = [],
        context: Optional[dict] = None,
    ) -> Literal["escalatory", "deescalatory", "neutral"]:
        payload: dict = {"current_nurse_line": nurse_line}
        if self.conversation_context_aware:
            payload["conversation_history"] = format_history(history, max_turns=self.max_turns)
        if self.scene_aware:
            payload["context"] = context or {}

        logger.debug("=== LLM INPUT (use_schema=%s) ===\n[SYSTEM]\n%s\n[USER]\n%s",
                     self.use_schema, self._system, json.dumps(payload, indent=2, ensure_ascii=False))

        raw = self.client.grade(
            model=self.model,
            system=self._system,
            user_payload=payload,
            schema_name=self._schema_name,
            schema_body=self._schema_body,
            temperature=self.temperature,
            seed=self.seed,
            use_schema=self.use_schema,
        )

        logger.debug("=== LLM OUTPUT (use_schema=%s) ===\n%s", self.use_schema, raw)

        if self.use_schema:
            data = json.loads(raw)
            return data["turn_label"]
        else:
            return raw.strip()
