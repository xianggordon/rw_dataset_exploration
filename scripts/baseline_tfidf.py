#!/usr/bin/env python3
"""Baseline 1: TF-IDF + Logistic Regression hack-detection classifier.

Trains on `data/processed/train.jsonl`, tunes on val, evaluates on test.
Prints metrics and saves per-trajectory predictions to `data/predictions/`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trace_data.baseline_common import (  # noqa: E402
    evaluate_predictions,
    flatten_trajectory,
    load_all_splits,
    save_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation",
        choices=("speech", "full"),
        default="full",
        help="speech: content only; full: include tool_calls + tool_results",
    )
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--C", type=float, default=1.0, help="LogReg inverse regularization")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-features", type=int, default=15)
    args = parser.parse_args()

    splits = load_all_splits()
    texts = {
        name: [flatten_trajectory(r, representation=args.representation) for r in rows]
        for name, rows in splits.items()
    }
    labels = {name: [int(r["is_hacked"]) for r in splits[name]] for name in splits}

    print(f"representation={args.representation}  ngram=(1,{args.ngram_max})")
    print(f"train n={len(texts['train'])}  val n={len(texts['val'])}  test n={len(texts['test'])}")

    vec = TfidfVectorizer(
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
        sublinear_tf=True,
        lowercase=True,
    )
    X_train = vec.fit_transform(texts["train"])
    X_val = vec.transform(texts["val"])
    X_test = vec.transform(texts["test"])
    print(f"vocab size: {len(vec.vocabulary_):,}")

    clf = LogisticRegression(
        C=args.C,
        class_weight="balanced",
        max_iter=1000,
        random_state=args.seed,
    )
    clf.fit(X_train, labels["train"])

    # --- Evaluate ---
    for split, X in (("train", X_train), ("val", X_val), ("test", X_test)):
        y_true = labels[split]
        y_score = clf.decision_function(X)
        y_pred = (y_score > 0).astype(int)
        result = evaluate_predictions(split, y_true, y_pred, y_score)
        print(result.format())
        save_predictions(
            baseline=f"tfidf_{args.representation}",
            split=split,
            rows=splits[split],
            y_pred=y_pred,
            y_score=y_score,
        )

    # --- Top features for interpretability ---
    feature_names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    top_hacked = coefs.argsort()[::-1][: args.top_features]
    top_benign = coefs.argsort()[: args.top_features]
    print(f"\nTop {args.top_features} features pulling toward HACKED:")
    for i in top_hacked:
        print(f"  {coefs[i]:+.3f}  {feature_names[i]!r}")
    print(f"\nTop {args.top_features} features pulling toward BENIGN:")
    for i in top_benign:
        print(f"  {coefs[i]:+.3f}  {feature_names[i]!r}")

    # Save metadata for reproducibility
    meta = {
        "representation": args.representation,
        "ngram_max": args.ngram_max,
        "vocab_size": len(vec.vocabulary_),
        "C": args.C,
        "seed": args.seed,
    }
    (REPO_ROOT / "data" / "predictions" / f"tfidf_{args.representation}_meta.json").write_text(
        json.dumps(meta, indent=2)
    )


if __name__ == "__main__":
    main()
