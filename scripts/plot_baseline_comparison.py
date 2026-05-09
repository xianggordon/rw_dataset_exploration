#!/usr/bin/env python3
"""Generate visual comparison of the three baseline approaches on the TRACE test split.

Reads test-set prediction JSONLs from data/predictions/, computes metrics and
ROC/PR curves, saves a 1x3 panel figure to results/figures/baseline_comparison.png.

Panels:
  1) Grouped bar chart of acc/F1/recall/precision/AUROC/AUPRC per approach
     (visualizes the markdown's cross-baseline comparison table)
  2) ROC curves overlaid (real ranking quality, comparable across all three)
  3) PR curves overlaid (alternative view, more sensitive to positive-class focus)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"
OUTPUT_DIR = REPO_ROOT / "results" / "figures"

APPROACHES = [
    {
        "name": "TF-IDF + LogReg",
        "predictions_file": "tfidf_full_test.jsonl",
        "color": "#4c72b0",
    },
    {
        "name": "TF-IDF + MLP",
        "predictions_file": "mlp_full_test.jsonl",
        "color": "#55a868",
    },
    {
        "name": "LLM (gpt-5.5)",
        "predictions_file": "llm_continuous_gpt-5.5_test.jsonl",
        "color": "#c44e52",
    },
]

METRIC_ORDER = ["acc", "F1", "recall", "precision", "AUROC", "AUPRC"]


def load_predictions(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in (PREDICTIONS_DIR / filename).read_text().splitlines()]
    y_true = np.array([r["y_true"] for r in rows], dtype=int)
    y_pred = np.array([r["y_pred"] for r in rows], dtype=int)
    y_score = np.array([float(r["score"]) if r["score"] is not None else 0.5 for r in rows])
    return y_true, y_pred, y_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0.0)),
        "AUROC": float(roc_auc_score(y_true, y_score)),
        "AUPRC": float(average_precision_score(y_true, y_score)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {}
    for app in APPROACHES:
        path = PREDICTIONS_DIR / app["predictions_file"]
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {app['name']}", file=sys.stderr)
            continue
        y_true, y_pred, y_score = load_predictions(app["predictions_file"])
        data[app["name"]] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score,
            "metrics": compute_metrics(y_true, y_pred, y_score),
            "color": app["color"],
        }
    if not data:
        sys.exit("No prediction files found.")

    # Figure: 1x3 horizontal layout
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # --- Panel 1: grouped bar chart ---
    ax = axes[0]
    n_approaches = len(data)
    n_metrics = len(METRIC_ORDER)
    x = np.arange(n_metrics)
    width = 0.8 / n_approaches
    for i, (name, d) in enumerate(data.items()):
        values = [d["metrics"][m] for m in METRIC_ORDER]
        offset = (i - (n_approaches - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name, color=d["color"])
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_ORDER)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Test-set metrics (n=104)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # --- Panel 2: ROC curves ---
    ax = axes[1]
    for name, d in data.items():
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_score"])
        ax.plot(
            fpr,
            tpr,
            color=d["color"],
            lw=2,
            label=f"{name}  (AUROC={d['metrics']['AUROC']:.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUROC=0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    # --- Panel 3: PR curves ---
    ax = axes[2]
    prevalence = next(iter(data.values()))["y_true"].mean()
    for name, d in data.items():
        precision, recall, _ = precision_recall_curve(d["y_true"], d["y_score"])
        ax.plot(
            recall,
            precision,
            color=d["color"],
            lw=2,
            label=f"{name}  (AUPRC={d['metrics']['AUPRC']:.3f})",
        )
    ax.axhline(prevalence, color="k", ls="--", lw=1, alpha=0.5, label=f"Random ({prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    fig.suptitle(
        "Three approaches on TRACE test split — cross-baseline comparison",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    output_path = OUTPUT_DIR / "baseline_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
