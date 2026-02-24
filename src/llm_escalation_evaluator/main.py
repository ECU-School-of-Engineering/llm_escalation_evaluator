import time
import tempfile
import textwrap
from pathlib import Path
from llm_escalation_evaluator.data_types import Turn
from llm_escalation_evaluator.grader import EscalationGrader
from llm_escalation_evaluator.yaml_config import load_config, build_schema_body, DEFAULT_CONFIG_PATH
from dotenv import load_dotenv

load_dotenv()

debug = False

history = [
    Turn(role="patient", text="Nobody here cares. I've been waiting forever!"),
    Turn(role="nurse", text="I'm sorry it's been a long wait. Tell me what's worrying you most."),
]
nurse_line = "You need to calm down or I can't help you."
context = {"setting": "ED", "goal": "de-escalation training"}

# ── Non-API YAML tests ─────────────────────────────────────────────────────────

print("=== YAML TEST 1: default config loads with expected keys ===")
cfg = load_config()
required_keys = {"model", "use_schema", "temperature", "max_turns", "log_level", "system_prompt", "schema"}
missing = required_keys - cfg.keys()
assert not missing, f"Missing keys: {missing}"
assert isinstance(cfg["system_prompt"], str) and len(cfg["system_prompt"]) > 20
assert isinstance(cfg["schema"]["fields"], list) and len(cfg["schema"]["fields"]) > 0
print(f"  OK  model={cfg['model']}  use_schema={cfg['use_schema']}  fields={len(cfg['schema']['fields'])}")
print(f"  default config path: {DEFAULT_CONFIG_PATH}")

print()
print("=== YAML TEST 2: build_schema_body only includes required:true fields ===")
fields = [
    {"name": "turn_label", "type": "string", "enum": ["escalatory", "deescalatory", "neutral"], "required": True},
    {"name": "confidence",  "type": "number", "minimum": 0.0, "maximum": 1.0, "required": False},
    {"name": "rationale",   "type": "string", "required": False},
]
body = build_schema_body(fields)
assert "turn_label" in body["properties"], "turn_label should be included"
assert "confidence" not in body["properties"], "confidence should be excluded (required:false)"
assert "rationale"  not in body["properties"], "rationale should be excluded (required:false)"
assert body["required"] == ["turn_label"]
print(f"  OK  properties={list(body['properties'].keys())}  required={body['required']}")

print()
print("=== YAML TEST 3: kwarg overrides YAML value ===")
grader = EscalationGrader(model="gpt-4o", temperature=0.7, use_schema=False)
assert grader.model == "gpt-4o",       f"expected gpt-4o, got {grader.model}"
assert grader.temperature == 0.7,      f"expected 0.7, got {grader.temperature}"
assert grader.use_schema is False,     f"expected False, got {grader.use_schema}"
assert grader.max_turns == cfg["max_turns"], "max_turns should come from YAML (not overridden)"
print(f"  OK  model={grader.model}  temperature={grader.temperature}  use_schema={grader.use_schema}  max_turns={grader.max_turns}")

print()
print("=== YAML TEST 4: custom YAML path is loaded ===")
custom_yaml = textwrap.dedent("""\
    model: gpt-4o-mini
    use_schema: true
    temperature: 0.5
    seed: 42
    max_turns: 8
    log_level: WARNING
    system_prompt: "Custom prompt for testing."
    schema:
      name: test_schema
      fields:
        - name: turn_label
          type: string
          enum: [escalatory, deescalatory, neutral]
          required: true
        - name: confidence
          type: number
          minimum: 0.0
          maximum: 1.0
          required: true
""")
with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
    f.write(custom_yaml)
    tmp_path = f.name

grader_custom = EscalationGrader(config_path=tmp_path)
assert grader_custom.temperature == 0.5,   f"expected 0.5, got {grader_custom.temperature}"
assert grader_custom.seed == 42,           f"expected 42, got {grader_custom.seed}"
assert grader_custom.max_turns == 8,       f"expected 8, got {grader_custom.max_turns}"
assert grader_custom._system == "Custom prompt for testing."
assert "confidence" in grader_custom._schema_body["properties"], "confidence should be active"
assert grader_custom._schema_name == "test_schema"
print(f"  OK  temperature={grader_custom.temperature}  seed={grader_custom.seed}  max_turns={grader_custom.max_turns}")
print(f"      schema_name={grader_custom._schema_name}  active_fields={list(grader_custom._schema_body['properties'].keys())}")
Path(tmp_path).unlink()

# ── Context-awareness payload tests (no API) ───────────────────────────────────

print()
print("=== YAML TEST 5: context-awareness flag combinations ===")

combos = [
    (True,  True,  {"current_nurse_line", "conversation_history", "context"}),
    (True,  False, {"current_nurse_line", "conversation_history"}),
    (False, True,  {"current_nurse_line", "context"}),
    (False, False, {"current_nurse_line"}),
]
for conv_aware, scene_aware, expected_keys in combos:
    g = EscalationGrader(conversation_context_aware=conv_aware, scene_aware=scene_aware)
    assert g.conversation_context_aware == conv_aware
    assert g.scene_aware == scene_aware
    # Reconstruct the payload the same way grader.py would
    payload: dict = {"current_nurse_line": nurse_line}
    if g.conversation_context_aware:
        from llm_escalation_evaluator.prompt import format_history
        payload["conversation_history"] = format_history(history, max_turns=g.max_turns)
    if g.scene_aware:
        payload["context"] = context or {}
    assert set(payload.keys()) == expected_keys, f"Expected {expected_keys}, got {set(payload.keys())}"
    print(f"  OK  conv={conv_aware}  scene={scene_aware}  keys={sorted(payload.keys())}")

# ── API tests ──────────────────────────────────────────────────────────────────

print()
print("=== TEST 1: use_schema=True (structured output) ===")
grader = EscalationGrader(model="gpt-4o-mini", debug=debug, use_schema=True)
t0 = time.perf_counter()
result = grader.evaluate_nurse_turn(history=history, nurse_line=nurse_line, context=context)
print(f"turn_label: {result}  ({time.perf_counter() - t0:.2f}s)")

print()
print("=== TEST 2: use_schema=False (free-text output) ===")
grader = EscalationGrader(model="gpt-4o-mini", debug=debug, use_schema=False)
t0 = time.perf_counter()
result = grader.evaluate_nurse_turn(history=history, nurse_line=nurse_line, context=context)
print(f"turn_label: {result}  ({time.perf_counter() - t0:.2f}s)")

print()
print("=== TEST 3: context passthrough ===")
grader = EscalationGrader(model="gpt-4o-mini", debug=debug, use_schema=True)

print("  -- with context --")
t0 = time.perf_counter()
result_with = grader.evaluate_nurse_turn(history=history, nurse_line=nurse_line, context=context)
print(f"turn_label: {result_with}  ({time.perf_counter() - t0:.2f}s)  context={context}")

print("  -- without context (None -> {{}}) --")
t0 = time.perf_counter()
result_without = grader.evaluate_nurse_turn(history=history, nurse_line=nurse_line, context=None)
print(f"turn_label: {result_without}  ({time.perf_counter() - t0:.2f}s)  context=None -> {{}}")
