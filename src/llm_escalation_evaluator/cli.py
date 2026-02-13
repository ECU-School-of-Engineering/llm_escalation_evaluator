from __future__ import annotations
import os

from .grader import EscalationGrader
from .session import TrainingSession

def main() -> None:
    # OPENAI_API_KEY must be set in env for OpenAI SDK
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY in your environment.")
        return

    grader = EscalationGrader(model="gpt-5-mini")
    session = TrainingSession(
        grader=grader,
        context={"setting": "ED", "goal": "de-escalation training"},
    )

    print("Nurse/Patient escalation grading CLI")
    print("Type: 'p: ...' for patient, 'n: ...' for nurse (nurse lines get graded).")
    print("Type 'export' to print session JSON, or 'quit'.")

    while True:
        line = input("> ").strip()
        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            break
        if line.lower() == "export":
            print(session.export_json())
            continue

        if line.startswith("p:"):
            session.add_patient(line[2:].strip())
            print("(patient added)")
            continue

        if line.startswith("n:"):
            nurse_text = line[2:].strip()
            result = session.add_nurse_and_grade(nurse_text)
            print(f"label={result.turn_label} impact={result.nurse_impact:.2f} "
                  f"escalation={result.patient_escalation_level:.2f} conf={result.confidence:.2f}")
            if result.coaching:
                print("coaching:")
                for c in result.coaching:
                    print(f" - {c}")
            continue

        print("Unknown input. Use 'p:' or 'n:'.")

if __name__ == "__main__":
    main()