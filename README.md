# LLM Escalation Evaluator

Grades nurse statements for their impact on patient escalation using an LLM.

## Installation

```bash
pip install -e .
```

Requires an OpenAI API key in `.env` or the environment:

```
OPENAI_API_KEY=sk-...
```

## Quick start

```python
from llm_escalation_evaluator.data_types import Turn
from llm_escalation_evaluator.grader import EscalationGrader

grader = EscalationGrader()

result = grader.evaluate_nurse_turn(
    nurse_line="You need to calm down or I can't help you.",
    history=[
        Turn(role="patient", text="Nobody here cares. I've been waiting forever!"),
        Turn(role="nurse",   text="I'm sorry it's been a long wait. Tell me what's worrying you most."),
    ],
    context={"setting": "ED", "goal": "de-escalation training"},
)
# result: "escalatory" | "deescalatory" | "neutral"
```

## Configuration

All settings are controlled by a YAML file. The package ships with a default at
`src/llm_escalation_evaluator/grader_config.yaml`. Pass a custom path to override:

```python
grader = EscalationGrader(config_path="/path/to/my_config.yaml")
```

Any kwarg passed to `EscalationGrader()` overrides the YAML value for that run.

### Top-level settings

| Key | Default | Description |
|---|---|---|
| `model` | `gpt-4o-mini` | OpenAI model name |
| `temperature` | `0.0` | Sampling temperature |
| `seed` | `null` | Random seed for reproducibility |
| `max_turns` | `16` | Maximum conversation turns sent to the LLM |
| `use_schema` | `true` | Enforce structured JSON output via OpenAI strict schema |
| `log_level` | `WARNING` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `conversation_context_aware` | `true` | Include prior conversation turns in the prompt |
| `scene_aware` | `true` | Include clinical context (setting, goal, etc.) in the prompt |

### Context-awareness modes

The two flags control what is sent to the LLM alongside the nurse line:

| `conversation_context_aware` | `scene_aware` | Prompt includes |
|---|---|---|
| `true` | `true` | history + context + nurse line (default) |
| `true` | `false` | history + nurse line |
| `false` | `true` | context + nurse line |
| `false` | `false` | nurse line only |

When `conversation_context_aware` is `false`, `history` is not required:

```python
grader = EscalationGrader(conversation_context_aware=False, scene_aware=False)
result = grader.evaluate_nurse_turn(nurse_line="You need to calm down.")
```

### Structured output vs free text

```python
# Structured (default) — returns lowercase enum value
grader = EscalationGrader(use_schema=True)
# → "escalatory" | "deescalatory" | "neutral"

# Free text — returns raw LLM output stripped of whitespace
grader = EscalationGrader(use_schema=False)
# → "Escalation" | "De-Escalation" | "Neutral"
```

### Schema fields

Activate additional output fields by setting `required: true` in the YAML.
Fields with `required: false` are defined but not sent to the API.

| Field | Type | Description |
|---|---|---|
| `turn_label` | `string` enum | `escalatory` / `deescalatory` / `neutral` — always active |
| `nurse_impact` | `number [-1, 1]` | Nurse line's direct emotional impact |
| `patient_escalation_level` | `number [-1, 1]` | Estimated patient state after the line |
| `confidence` | `number [0, 1]` | Grader confidence in the label |
| `context_alignment` | `string` enum | Whether nurse addressed the patient's last message |
| `sarcasm_detected` | `string` enum | Whether nurse used sarcasm or ironic tone |
| `rationale` | `string` | Free-text explanation of the label |
| `signals` | `string[]` | Up to 8 textual signals that informed the label |
| `coaching` | `string[]` | 0–2 coaching suggestions for the nurse |

Example — activating `confidence` and `rationale`:

```yaml
schema:
  name: nurse_patient_escalation_grade
  fields:
    - name: turn_label
      type: string
      enum: [escalatory, deescalatory, neutral]
      required: true # ← flip to de-activate

    - name: confidence
      type: number
      minimum: 0.0
      maximum: 1.0
      required: false   # ← flip to activate

    - name: rationale
      type: string
      required: false   # ← flip to activate
```

### System prompt

The `system_prompt` key in the YAML is the full prompt sent to the LLM. Edit it
directly to change grading instructions without touching Python code.

## Running the smoke tests

```bash
pip install -e .
python -m llm_escalation_evaluator.main
```
