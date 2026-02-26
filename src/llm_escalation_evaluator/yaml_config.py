from __future__ import annotations
from pathlib import Path
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "grader_config.yaml"


def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_profile_config(profile: str, path: str | None = None) -> dict:
    """Load the config block for a specific profile from the YAML."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    profiles = raw.get("profiles", {})
    if profile not in profiles:
        raise ValueError(
            f"Unknown profile '{profile}'. Available: {sorted(profiles.keys())}"
        )
    return profiles[profile]


def build_schema_body(fields: list) -> dict:
    # OpenAI strict mode requires every property to be in required,
    # so only include fields that are explicitly marked required: true.
    properties = {}
    required = []
    for field in fields:
        if not field.get("required", False):
            continue
        name = field["name"]
        spec: dict = {"type": field["type"]}
        for key in ("enum", "minimum", "maximum", "minItems", "maxItems"):
            if key in field:
                spec[key] = field[key]
        if "items" in field:
            spec["items"] = field["items"]
        properties[name] = spec
        required.append(name)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }
