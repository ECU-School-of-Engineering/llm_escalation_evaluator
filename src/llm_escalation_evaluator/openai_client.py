# src/llm_escalation_evaluator/openai_client.py
from __future__ import annotations
import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIResponsesClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found. Set env var or pass api_key.")
        self.client = OpenAI(api_key=api_key)

    def grade(self, *, model: str, system: str, user_payload: dict, schema_name: str, schema_body: dict) -> str:
        resp = self.client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,    
                    "schema": schema_body,    
                    "strict": True,
                }
            },
        )

        out = getattr(resp, "output_text", None)
        if not out:
            raise RuntimeError("No output_text returned from OpenAI response.")
        return out