from llm_escalation_evaluator.data_types import Turn, ConversationState
from llm_escalation_evaluator.grader import EscalationGrader

history = [
    Turn(role="patient", text="Nobody here cares. I’ve been waiting forever!"),
    Turn(role="nurse", text="I’m sorry it’s been a long wait. Tell me what’s worrying you most."),
]

state = ConversationState(escalation=0.55)

grader = EscalationGrader(model="gpt-5-mini")

result = grader.evaluate_nurse_turn(
    history=history,
    nurse_line="You need to calm down or I can’t help you.",
    state=state,
    context={"setting": "ED", "goal": "de-escalation training"},
)

state = grader.apply(state, result)

print(result)
print("New escalation:", state.escalation)