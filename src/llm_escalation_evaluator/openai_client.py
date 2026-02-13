# openai_client.py
from __future__ import annotations
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIResponsesClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Set it in environment or pass to constructor."
            )

        self.client = OpenAI(api_key=api_key)

    def grade(self, *, model: str, system: str, user_payload: dict, schema: dict) -> str:
        resp = self.client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_payload},
            ],
            text={"format": {"type": "json_schema", "json_schema": schema}},
        )
        return resp.output_text