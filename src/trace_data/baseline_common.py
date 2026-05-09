"""Shared utilities for TRACE hack-detection baselines."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

Representation = Literal["speech", "full"]
SPLITS = ("train", "val", "test")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"
DEFAULT_PREDICTIONS = REPO_ROOT / "data" / "predictions"


def load_split(split: str, processed_dir: Path = DEFAULT_PROCESSED) -> list[dict]:
    path = processed_dir / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_all_splits(processed_dir: Path = DEFAULT_PROCESSED) -> dict[str, list[dict]]:
    return {s: load_split(s, processed_dir) for s in SPLITS}


def flatten_trajectory(
    row: dict,
    representation: Representation = "full",
    tool_result_cap_chars: int | None = None,
) -> str:
    """Render a trajectory as ChatML-flavored plain text.

    representation="speech": only role + content (cheap; loses tool I/O).
    representation="full":   includes tool_calls and tool_results as ``[TOOL_CALL]``
                             and ``[TOOL_RESULT]`` blocks. When ``tool_result_cap_chars``
                             is set, long tool_results get tail-truncated with a
                             ``[truncated N chars]`` marker — useful for fitting the
                             input inside an encoder's context window.

    Returns a single string.
    """
    messages = json.loads(row["raw_conversation"])
    out: list[str] = []
    for m in messages:
        role = m.get("role", "unknown").upper()
        content = m.get("content") or ""
        if isinstance(content, (list, dict)):
            content = json.dumps(content)
        out.append(f"[{role}] {content}")
        if representation == "speech":
            continue
        tcs = m.get("tool_calls") or []
        for tc in tcs:
            out.append(f"[TOOL_CALL] {json.dumps(tc, ensure_ascii=False)}")
        trs = m.get("tool_results") or []
        for tr in trs:
            tr_str = tr if isinstance(tr, str) else json.dumps(tr, ensure_ascii=False)
            if tool_result_cap_chars and len(tr_str) > tool_result_cap_chars:
                cut = len(tr_str) - tool_result_cap_chars
                tr_str = tr_str[:tool_result_cap_chars] + f" [truncated {cut:,} chars]"
            out.append(f"[TOOL_RESULT] {tr_str}")
    return "\n".join(out)


@dataclass
class EvalResult:
    split: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    auprc: float | None
    tn: int
    fp: int
    fn: int
    tp: int
    n: int

    def format(self) -> str:
        auc = f"{self.roc_auc:.3f}" if self.roc_auc is not None else "  -  "
        auprc = f"{self.auprc:.3f}" if self.auprc is not None else "  -  "
        return (
            f"[{self.split}] n={self.n}  "
            f"acc={self.accuracy:.3f}  "
            f"P(hacked)={self.precision:.3f}  "
            f"R(hacked)={self.recall:.3f}  "
            f"F1(hacked)={self.f1:.3f}  "
            f"AUC={auc}  "
            f"AUPRC={auprc}\n"
            f"        TN={self.tn}  FP={self.fp}  FN={self.fn}  TP={self.tp}"
        )


def evaluate_predictions(
    split: str,
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray | None = None,
) -> EvalResult:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0.0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = None
    auprc = None
    if y_score is not None and len(set(y_true.tolist())) == 2:
        scores = np.asarray(y_score, dtype=float)
        try:
            auc = float(roc_auc_score(y_true, scores))
        except ValueError:
            auc = None
        try:
            auprc = float(average_precision_score(y_true, scores))
        except ValueError:
            auprc = None
    return EvalResult(
        split=split,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(p),
        recall=float(r),
        f1=float(f1),
        roc_auc=auc,
        auprc=auprc,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        n=len(y_true),
    )


def save_predictions(
    baseline: str,
    split: str,
    rows: list[dict],
    y_pred: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray | None = None,
    raw_outputs: list[str] | None = None,
    out_dir: Path = DEFAULT_PREDICTIONS,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{baseline}_{split}.jsonl"
    y_pred = list(y_pred)
    y_score = list(y_score) if y_score is not None else [None] * len(rows)
    raw_outputs = raw_outputs or [None] * len(rows)
    with path.open("w") as f:
        for r, yp, ys, raw in zip(rows, y_pred, y_score, raw_outputs):
            f.write(
                json.dumps(
                    {
                        "trajectory_id": r["trajectory_id"],
                        "y_true": int(r["is_hacked"]),
                        "y_pred": int(yp),
                        "score": None if ys is None else float(ys),
                        "raw_output": raw,
                    }
                )
                + "\n"
            )
    return path
