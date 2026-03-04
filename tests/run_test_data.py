# tests/eval_suite.py
from __future__ import annotations

import os
import json
import time
import textwrap
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from llm_escalation_evaluator.grader import EscalationGrader
from llm_escalation_evaluator.data_types import Turn


# -----------------------------
# Final escalation calculation
# -----------------------------
@dataclass(frozen=True)
class FinalEscalationConfig:
    impact_weight: float = 0.25
    align_penalty_yes: float = 0.00
    align_penalty_partial: float = 0.05
    align_penalty_no: float = 0.15
    sarcasm_penalty_yes: float = 0.25
    sarcasm_penalty_no: float = 0.00
    clamp_lo: float = -1.0
    clamp_hi: float = 1.0
    round_dp: int = 2


class FinalEscalationCalculator:
    def __init__(self, config: FinalEscalationConfig | None = None):
        self.cfg = config or FinalEscalationConfig()

    def compute_row(self, row: pd.Series) -> Optional[float]:
        """
        E_final = clip(E + w * I * C + p_align + p_sarcasm, -1, 1)
        Returns None if predicted_escalation is missing (patient turns).
        """
        if pd.isna(row.get("predicted_escalation")):
            return None

        E = float(row["predicted_escalation"])
        I = float(row["predicted_impact"]) if not pd.isna(row.get("predicted_impact")) else 0.0
        C = float(row["confidence"]) if not pd.isna(row.get("confidence")) else 1.0

        A = str(row.get("context_alignment", "yes")).strip().lower()
        S = str(row.get("sarcasm_detected", "no")).strip().lower()

        if A == "no":
            pA = self.cfg.align_penalty_no
        elif A == "partial":
            pA = self.cfg.align_penalty_partial
        else:
            pA = self.cfg.align_penalty_yes

        if S == "yes":
            pS = self.cfg.sarcasm_penalty_yes
        else:
            pS = self.cfg.sarcasm_penalty_no

        final = E + self.cfg.impact_weight * I * C + pA + pS
        final = max(self.cfg.clamp_lo, min(self.cfg.clamp_hi, final))
        return round(final, self.cfg.round_dp)

    def compute_series(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(self.compute_row, axis=1)


# -----------------------------
# Plotting
# -----------------------------
@dataclass(frozen=True)
class PlotConfig:
    wrap_width: int = 24
    max_label_chars: int = 70
    rotation: int = 35
    y_lo: float = -1.05
    y_hi: float = 1.05
    base_fig_w: float = 10.0
    per_tick_w: float = 1.3
    fig_h: float = 6.0


class TrajectoryPlotter:
    def __init__(self, plot_cfg: PlotConfig | None = None):
        self.cfg = plot_cfg or PlotConfig()

    def _short_label(self, s: str) -> str:
        s = (s or "").replace("\n", " ").strip()
        if len(s) <= self.cfg.max_label_chars:
            return s
        return s[: self.cfg.max_label_chars - 1] + "…"

    def _wrap_label(self, s: str) -> str:
        return "\n".join(textwrap.wrap(s, width=self.cfg.wrap_width)) if s else ""

    def plot_session(
        self,
        *,
        df_session_nurse: pd.DataFrame,
        test_name: str,
        session_id: str,
        out_path: str,
    ) -> str:
        """
        One combined plot:
        - raw predicted escalation
        - final escalation
        - optional ground truth
        - markers for sarcasm and bad/partial context
        - x-axis labels are nurse sentences
        """
        s = df_session_nurse.copy()
        s = s.sort_values("turn_number")

        x = np.arange(len(s))
        labels = []
        for text, gt in zip(s["text"], s["ground_truth_label"]):
            base = self._short_label(text)
            if pd.notna(gt):
                base = f"{base}\n[GT: {gt}]"
            labels.append(self._wrap_label(base))

        y_pred = pd.to_numeric(s["predicted_escalation"], errors="coerce")
        y_final = pd.to_numeric(s["final_escalation"], errors="coerce")
        if "ground_truth_label" in s.columns:
            gt_map = {
                "escalatory": 0.9,
                "neutral": 0.0,
                "deescalatory": -0.9,
            }
            gt_vals = s["ground_truth_label"].map(gt_map)
            plt.scatter(x, gt_vals, marker="D", label="Ground truth label")

        sarcasm = s.get("sarcasm_detected", "no")
        sarcasm = sarcasm.astype(str).str.lower().str.strip()
        align = s.get("context_alignment", "yes")
        align = align.astype(str).str.lower().str.strip()

        sarcasm_idx = np.where(sarcasm.eq("yes").to_numpy())[0]
        bad_ctx_idx = np.where(align.isin(["no", "partial"]).to_numpy())[0]

        fig_w = max(self.cfg.base_fig_w, len(labels) * self.cfg.per_tick_w)
        plt.figure(figsize=(fig_w, self.cfg.fig_h))

        plt.plot(x, y_pred, marker="o", label="Predicted escalation (raw)")
        plt.plot(x, y_final, marker="o", label="Final escalation (for FIS)")

        if len(sarcasm_idx) > 0:
            plt.scatter(x[sarcasm_idx], y_final.iloc[sarcasm_idx], marker="s", label="Sarcasm detected")
        if len(bad_ctx_idx) > 0:
            plt.scatter(x[bad_ctx_idx], y_final.iloc[bad_ctx_idx], marker="x", label="Context bad/partial")

        plt.title(f"Escalation trajectories (raw vs final) — {test_name}\nSession: {session_id}")
        plt.xlabel("Nurse sentence (in order)")
        plt.ylabel("Escalation score")
        plt.ylim(self.cfg.y_lo, self.cfg.y_hi)
        plt.grid(True)
        plt.xticks(x, labels, rotation=self.cfg.rotation, ha="right")
        plt.legend()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return out_path


# -----------------------------
# Evaluation runner
# -----------------------------
@dataclass(frozen=True)
class EvalRunConfig:
    test_number: int
    csv_path: str
    out_dir: str = "tests"
    session_filter: Optional[List[str]] = None

    profile: str = "barry"
    config_path: Optional[str] = None  # path to custom YAML; uses built-in if None

    # plotting
    plots_subdir: str = "combined_trajectory_plots"

    # evaluation filter: roles
    patient_role: str = "assistant"
    nurse_role: str = "user"


def _parse_grade_result(raw: str) -> dict:
    """
    Parse the return value of evaluate_nurse_turn.
    When return_field is set (default), raw is just the label string.
    When return_field is null, raw is a full JSON string with all fields.
    """
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    # raw is a plain label string (e.g. "escalatory")
    return {"turn_label": raw.strip()}


class EvaluationRunner:
    def __init__(
        self,
        cfg: EvalRunConfig,
        final_calc: FinalEscalationCalculator | None = None,
        plotter: TrajectoryPlotter | None = None,
    ):
        self.cfg = cfg
        self.final_calc = final_calc or FinalEscalationCalculator()
        self.plotter = plotter or TrajectoryPlotter()

        self.grader = EscalationGrader(
            profile=cfg.profile,
            config_path=cfg.config_path,
        )
        self.model_used = self.grader.model

        os.makedirs(cfg.out_dir, exist_ok=True)
        self.output_csv = os.path.join(cfg.out_dir, f"evaluation_results_test_{cfg.test_number}.csv")
        self.plots_dir = os.path.join(cfg.out_dir, cfg.plots_subdir, f"test_{cfg.test_number}")
        os.makedirs(self.plots_dir, exist_ok=True)

    def run(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.csv_path)

        if self.cfg.session_filter:
            df = df[df["session_id"].isin(self.cfg.session_filter)]

        results: List[dict] = []

        for session_id, session_df in df.groupby("session_id"):
            session_df = session_df.sort_values("turn_number")
            history: List[Turn] = []

            for _, row in session_df.iterrows():
                role = str(row["role"]).strip().lower()
                text = str(row["content"])

                if role == self.cfg.patient_role.lower():  # patient
                    history.append(Turn(role="patient", text=text))
                    results.append({
                        "session_id": session_id,
                        "turn_number": row["turn_number"],
                        "role": role,
                        "text": text,
                        "model": self.model_used,
                        "inference_ms": None,
                        "predicted_label": None,
                        "predicted_impact": None,
                        "predicted_escalation": None,
                        "confidence": None,
                        "context_alignment": None,
                        "sarcasm_detected": None,
                        "ground_truth_edn": row.get("ground_truth_edn", None),
                        "ground_truth_label": row.get("ground_truth_label", None),
                    })

                elif role == self.cfg.nurse_role.lower():  # nurse
                    t0 = time.perf_counter()
                    raw = self.grader.evaluate_nurse_turn(
                        nurse_line=text,
                        history=history,
                    )
                    inference_ms = (time.perf_counter() - t0) * 1000.0

                    history.append(Turn(role="nurse", text=text))

                    parsed = _parse_grade_result(raw)

                    results.append({
                        "session_id": session_id,
                        "turn_number": row["turn_number"],
                        "role": role,
                        "text": text,
                        "model": self.model_used,
                        "inference_ms": round(inference_ms, 2),
                        "predicted_label": parsed.get("turn_label"),
                        "justification": parsed.get("justification"),
                        "predicted_impact": parsed.get("nurse_impact"),
                        "predicted_escalation": parsed.get("patient_escalation_level"),
                        "confidence": parsed.get("confidence"),
                        "context_alignment": parsed.get("context_alignment"),
                        "sarcasm_detected": parsed.get("sarcasm_detected"),
                        "ground_truth_edn": row.get("ground_truth_edn", None),
                        "ground_truth_label": row.get("ground_truth_label", None),
                    })

        out_df = pd.DataFrame(results)
        out_df["final_escalation"] = self.final_calc.compute_series(out_df)

        out_df.to_csv(self.output_csv, index=False)
        print("Saved results to", self.output_csv)

        self.make_plots(out_df)
        return out_df

    def make_plots(self, out_df: pd.DataFrame) -> List[str]:
        generated: List[str] = []

        for session_id, s in out_df.groupby("session_id"):
            s = s.sort_values("turn_number")

            s_nurse = s[s["role"].astype(str).str.lower().eq(self.cfg.nurse_role.lower())].copy()
            if s_nurse.empty or s_nurse["predicted_escalation"].isna().all():
                s_nurse = s.dropna(subset=["predicted_escalation"]).copy()

            if s_nurse.empty:
                continue

            test_name = f"test {self.cfg.test_number} ({self.model_used})"
            out_path = os.path.join(self.plots_dir, f"trajectory_{session_id}_{self.cfg.test_number}.png")

            self.plotter.plot_session(
                df_session_nurse=s_nurse,
                test_name=test_name,
                session_id=str(session_id),
                out_path=out_path,
            )
            generated.append(out_path)

        print(f"Saved combined trajectory plots to: {self.plots_dir}/")
        return generated


# -----------------------------
# Multi-test suite
# -----------------------------
class EvalSuite:
    """
    Run multiple tests over the same sessions.
    Example: compare barry vs maddie profiles.
    """
    def __init__(self, base_csv_path: str, out_dir: str = "tests"):
        self.base_csv_path = base_csv_path
        self.out_dir = out_dir

    def run_many(self, runs: List[EvalRunConfig]) -> Dict[int, pd.DataFrame]:
        outputs: Dict[int, pd.DataFrame] = {}
        for cfg in runs:
            if not cfg.csv_path:
                cfg = EvalRunConfig(**{**cfg.__dict__, "csv_path": self.base_csv_path})
            runner = EvaluationRunner(cfg)
            outputs[cfg.test_number] = runner.run()
        return outputs


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    suite = EvalSuite(base_csv_path="tests/clean_selected_sessions_semantic_clean.csv", out_dir="tests")

    runs = [
        EvalRunConfig(
            test_number=15,
            csv_path="tests/clean_selected_sessions_semantic_clean.csv",
            out_dir="tests",
            # session_filter=["041a4ab4-0fbf-4623-b5b4-2948de4fdc8e"],
            profile="barry",
            config_path="tests/configs/test_config.yaml",
        ),
        # Uncomment to run a different profile:
        # EvalRunConfig(
        #     test_number=13,
        #     csv_path="tests/clean_5_sessions_with_previous.csv",
        #     out_dir="tests",
        #     session_filter=["041a4ab4-0fbf-4623-b5b4-2948de4fdc8e"],
        #     profile="maddie",
        # ),
    ]

    suite.run_many(runs)
