import pandas as pd
from llm_escalation_evaluator.grader import EscalationGrader
from llm_escalation_evaluator.session import TrainingSession
import time
TEST_NUMBER = 2
CSV_PATH = "tests/conversations_Feb_01.csv"
OUTPUT_PATH = f"tests/evaluation_results_test_{TEST_NUMBER}.csv"

# ---- FILTER HERE ----
SESSION_FILTER = [
    "cb2053bb-72a6-4542-bdf6-07de82a0dcdb",
    # add more session IDs if needed
]
# ---------------------

df = pd.read_csv(CSV_PATH)

# Filter dataset
if SESSION_FILTER:
    df = df[df["session_id"].isin(SESSION_FILTER)]

grader = EscalationGrader(model="gpt-5-mini")
MODEL_USED = grader.model  # store once

results = []

for session_id, session_df in df.groupby("session_id"):
    session_df = session_df.sort_values("turn_number")

    session = TrainingSession(
        grader=grader,
        context={"setting": "ED", "goal": "de-escalation training"},
    )

    for _, row in session_df.iterrows():
        role = str(row["role"]).strip().lower()
        text = str(row["content"])

        if role == "assistant":  # patient
            session.add_patient(text)

            results.append({
                "session_id": session_id,
                "turn_number": row["turn_number"],
                "role": role,
                "text": text,
                "model": MODEL_USED,
                "inference_ms": None,              # no inference for patient turns
                "predicted_label": None,
                "ground_truth_label": None,
                "correct": None,
                "predicted_impact": None,
                "predicted_escalation": None,
                "confidence": None,
                "ground_truth_score": row.get("escalation", None),
            })

        elif role == "user":  # nurse
            t0 = time.perf_counter()
            result = session.add_nurse_and_grade(text)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            ground_truth = row.get("escalation", None)

            # simple label mapping for comparison
            predicted_label = result.turn_label

            gt_label = None
            if ground_truth is not None and str(ground_truth) != "nan":
                gt = float(ground_truth)
                if gt > 0.1:
                    gt_label = "escalatory"
                elif gt < -0.1:
                    gt_label = "deescalatory"
                else:
                    gt_label = "neutral"

            correct = (predicted_label == gt_label) if gt_label is not None else None

            results.append({
                "session_id": session_id,
                "turn_number": row["turn_number"],
                "role": role,
                "text": text,
                "model": MODEL_USED,
                "inference_ms": round(inference_ms, 2),
                "predicted_label": predicted_label,
                "ground_truth_label": gt_label,
                "correct": correct,
                "predicted_impact": result.nurse_impact,
                "predicted_escalation": result.patient_escalation_level,
                "confidence": result.confidence,
                "ground_truth_score": ground_truth,
            })

out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_PATH, index=False)

print("Saved results to", OUTPUT_PATH)

# ---- accuracy summary (nurse rows only) ----
if "correct" in out_df.columns:
    valid = out_df.dropna(subset=["correct"])
    if len(valid) > 0:
        acc = valid["correct"].mean()
        print(f"Accuracy: {acc:.2%}")
    else:
        print("No comparable ground truth rows found.")
else:
    print("Column 'correct' not found — no nurse turns were evaluated.")

# ---- inference time summary (nurse rows only) ----
t = out_df["inference_ms"].dropna()
if len(t) > 0:
    print(
        f"Inference time (ms): mean={t.mean():.2f}  "
        f"p50={t.median():.2f}  p95={t.quantile(0.95):.2f}  "
        f"min={t.min():.2f}  max={t.max():.2f}  n={len(t)}"
    )
else:
    print("No inference timings collected (no nurse turns evaluated).")

#____________________________________________________________________________________________________________________
# Additional code to visualize escalation trajectories per session

import os
import matplotlib.pyplot as plt

PLOTS_DIR = "tests/trajectory_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

out_df = pd.read_csv(OUTPUT_PATH)

# plot only nurse turns for predicted escalation (patient rows have NaN)
for session_id, s in out_df.groupby("session_id"):
    s = s.sort_values("turn_number")

    # Use only rows that have predictions (nurse turns)
    s_pred = s.dropna(subset=["predicted_escalation"])

    x = s_pred["turn_number"].astype(int)
    y_pred = s_pred["predicted_escalation"].astype(float)
    y_gt = pd.to_numeric(s_pred["ground_truth_score"], errors="coerce")

    plt.figure()
    plt.plot(x, y_pred, marker="o", label="Predicted escalation (after nurse line)")
    plt.plot(x, y_gt, marker="o", label="Ground truth escalation")

    plt.title(f"Escalation Trajectory — {session_id}")
    plt.xlabel("Turn number (nurse turns only)")
    plt.ylabel("Escalation score")
    plt.ylim(-1.05, 1.05)
    plt.grid(True)
    plt.legend()

    path = os.path.join(PLOTS_DIR, f"{session_id}_trajectory_test{TEST_NUMBER}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

print(f"Saved trajectory plots to: {PLOTS_DIR}/")