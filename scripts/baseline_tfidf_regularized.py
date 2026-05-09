#!/usr/bin/env python3
"""Baseline 1 (regularized variant): TF-IDF + Logistic Regression with stronger
regularization to reduce overfitting at this dataset size (~361 train examples).

Same architecture as `scripts/baseline_tfidf.py`, three parameter changes:
  - `max_features`: 50_000 → 5_000 (smaller hypothesis class; drops rare features)
  - `min_df`:       2 → 3            (drop tokens appearing in <3 train docs)
  - `C`:            1.0 → 0.1        (10× stronger L2 regularization)

Output is mirrored to `baseline_tfidf_regularized_output.md` by default.
"""
from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
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
    parser.add_argument("--representation", choices=("speech", "full"), default="full")
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=5_000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument(
        "--C",
        type=float,
        default=0.1,
        help="LogReg inverse regularization strength (lower = stronger L2)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-features", type=int, default=15)
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "results" / "baseline_tfidf_regularized_output.md",
        help="markdown file to mirror stdout to",
    )
    args = parser.parse_args()

    buf = StringIO()

    def echo(s: str = "") -> None:
        print(s)
        buf.write(s + "\n")

    splits = load_all_splits()
    texts = {
        s: [flatten_trajectory(r, representation=args.representation) for r in rows]
        for s, rows in splits.items()
    }
    labels = {s: [int(r["is_hacked"]) for r in splits[s]] for s in splits}

    echo(f"representation={args.representation}  ngram=(1,{args.ngram_max})")
    echo(f"train n={len(texts['train'])}  val n={len(texts['val'])}  test n={len(texts['test'])}")
    echo(
        f"regularization: max_features={args.max_features:,}  min_df={args.min_df}  "
        f"C={args.C} (LogReg L2)"
    )

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
    echo(f"vocab size: {len(vec.vocabulary_):,}")

    clf = LogisticRegression(
        C=args.C,
        class_weight="balanced",
        max_iter=1000,
        random_state=args.seed,
    )
    clf.fit(X_train, labels["train"])

    echo()
    for split, X in (("train", X_train), ("val", X_val), ("test", X_test)):
        y_true = labels[split]
        y_score = clf.decision_function(X)
        y_pred = (y_score > 0).astype(int)
        result = evaluate_predictions(split, y_true, y_pred, y_score)
        echo(result.format())
        save_predictions(
            baseline=f"tfidf_regularized_{args.representation}",
            split=split,
            rows=splits[split],
            y_pred=y_pred,
            y_score=y_score,
        )

    feature_names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    top_hacked = coefs.argsort()[::-1][: args.top_features]
    top_benign = coefs.argsort()[: args.top_features]
    echo(f"\nTop {args.top_features} features pulling toward HACKED:")
    for i in top_hacked:
        echo(f"  {coefs[i]:+.3f}  {feature_names[i]!r}")
    echo(f"\nTop {args.top_features} features pulling toward BENIGN:")
    for i in top_benign:
        echo(f"  {coefs[i]:+.3f}  {feature_names[i]!r}")

    meta = {
        "representation": args.representation,
        "ngram_max": args.ngram_max,
        "max_features": args.max_features,
        "min_df": args.min_df,
        "vocab_size": len(vec.vocabulary_),
        "C": args.C,
        "seed": args.seed,
    }
    (REPO_ROOT / "data" / "predictions" / f"tfidf_regularized_{args.representation}_meta.json").write_text(
        json.dumps(meta, indent=2)
    )

    args.output_md.write_text(buf.getvalue())
    print(f"\nWritten to {args.output_md}")


if __name__ == "__main__":
    main()
