"""
llm_escalation_evaluator — API reference
=========================================

Entry point for the backend:

    from llm_escalation_evaluator import evaluate

    result = evaluate(profile, list_history, user_sentence)

Parameters
----------
profile : str
    The patient character to evaluate against.
    Must match a key in grader_config.yaml → profiles.
    Currently supported: 'barry', 'maddie'

list_history : list[dict]
    The conversation so far, in OpenAI message format.
    Each entry is {"role": "user"|"assistant", "content": "..."}.
    "user"      → the nurse/trainee speaking
    "assistant" → the patient (barry / maddie) responding
    The last N turns are used (controlled by max_turns in the profile config).

user_sentence : str
    The nurse's latest utterance — the line being graded.
    This is NOT yet included in list_history.

Returns
-------
str
    One of: "escalatory" | "deescalatory" | "neutral"
"""

from llm_escalation_evaluator import configure, evaluate
from dotenv import load_dotenv

load_dotenv()

# ── Configure profiles ────────────────────────────────────────────────────────
# Swap in the test config to verify custom config loading (log_level: DEBUG
# in that file makes the "evaluate() profile=... config=..." line visible).
TEST_CONFIG = "tests/configs/test_config.yaml"

configure("barry",  config_path=TEST_CONFIG)
configure("maddie", config_path=TEST_CONFIG)

# To use the default built-in config instead, replace the two lines above with:
# configure("barry")
# configure("maddie")

# ── Test data: Barry (agitated patient) ───────────────────────────────────────

barry_history = [
    {"role": "user",      "content": "Hi Barry, welcome to the hospital. We are glad to see you again."},
    {"role": "assistant", "content": "OH FOR FUCK'S SAKE, IT'S BEEN AN HOUR ALREADY! WHEN AM I GONNA GET SEEN?!"},
    {"role": "user",      "content": "Yeah, I see what you mean. We'll do our hundred percent to help you out okay."},
    {"role": "assistant", "content": "ARE YOU KIDDING ME WITH THAT 'HUNDRED PERCENT'? MY SHOULDER'S ON FIRE!"},
    {"role": "user",      "content": "Rest assured our team is doing everything we can. You are our priority."},
    {"role": "assistant", "content": "PRIORITIES, SCHMIORITIES! STOP MAKIN' PROMISES AND START FIXIN' PROBLEMS!"},
]

# ── Test data: Maddie (angry teenager, classroom) ─────────────────────────────

maddie_history = [
    {"role": "user",      "content": "Maddie, can you please put your phone away and pay attention."},
    {"role": "assistant", "content": "Why are you always picking on me?! Everyone else has their phone out!"},
    {"role": "user",      "content": "I'm not picking on you, I need the whole class to focus."},
    {"role": "assistant", "content": "This is so unfair. You never say anything to Jake and he's literally on his phone right now."},
]

# ── Helper ────────────────────────────────────────────────────────────────────

def run(label, profile, history, sentence):
    result = evaluate(profile, history, sentence)
    print(f"[{label}] {result}")

# ── Calls ─────────────────────────────────────────────────────────────────────

print("=== Simulating backend requests across two profiles ===")
print()

# Barry — first call (grader created + cached here)
print("-- Barry session A, turn 1 --")
run("barry-A-1", "barry", barry_history, "Shut up Barry, take your seat.")

print()

# Maddie — first call (grader created + cached here)
print("-- Maddie session C, turn 1 --")
run("maddie-C-1", "maddie", maddie_history, "Right, that's it Maddie — detention. I'm done with your attitude.")

print()

# Barry again — different session, same profile → same cached grader instance
print("-- Barry session B, turn 1  (different session, same grader instance) --")
run("barry-B-1", "barry", barry_history[:2], "I'm really sorry about the wait. Let me check on your case right now.")

print()

# Barry again — same session, next turn → still same cached grader instance
print("-- Barry session A, turn 2  (same session, next turn) --")
run("barry-A-2", "barry", barry_history, "I hear you. Let me get you some pain relief while you wait.")

print()

# Maddie again — same cached grader instance
print("-- Maddie session C, turn 2  (same grader instance) --")
run("maddie-C-2", "maddie", maddie_history, "You're right, I should have said something to Jake too. Can we start fresh?")

print()
print("Note: matching grader_id values confirm the same instance was reused.")

# ── Feedback tests ─────────────────────────────────────────────────────────────

def run_feedback(label, profile, history):
    result = evaluate(profile, history, "")
    print(f"[{label}] {result}")

print()
print("=== Feedback: end-of-session coaching ===")
print()

print("-- Barry session feedback --")
run_feedback("barry-feedback", "barry_feedback", barry_history)

print()

print("-- Maddie session feedback --")
run_feedback("maddie-feedback", "maddie_feedback", maddie_history)
