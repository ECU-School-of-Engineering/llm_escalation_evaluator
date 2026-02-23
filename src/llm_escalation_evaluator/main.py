from llm_escalation_evaluator.data_types import Turn
from llm_escalation_evaluator.grader import EscalationGrader
from dotenv import load_dotenv

load_dotenv()

history = [
    Turn(role="patient", text="Nobody here cares. I've been waiting forever!"),
    Turn(role="nurse", text="I'm sorry it's been a long wait. Tell me what's worrying you most."),
]

grader = EscalationGrader(model="gpt-4o-mini", debug=True)

result = grader.evaluate_nurse_turn(
    history=history,
    nurse_line="You need to calm down or I can't help you.",
    context={"setting": "ED", "goal": "de-escalation training"},
)

print("turn_label:", result)
