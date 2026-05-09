#!/usr/bin/env python3
"""Baseline 2: TF-IDF features + small MLP classifier.

Same input space as Baseline 1, but a single-hidden-layer MLP head instead of
linear logistic regression. Designed to be small and CPU-friendly: ~1.3M params,
trains end-to-end in roughly 30 seconds on a laptop CPU.

Predicts hacked (1) vs. not-hacked (0). Output format mirrors the TF-IDF
baseline and is saved to `baseline_mlp_output.md` by default.
"""
from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trace_data.baseline_common import (  # noqa: E402
    evaluate_predictions,
    flatten_trajectory,
    load_all_splits,
    save_predictions,
)


class TinyMLP(nn.Module):
    """Linear → ReLU → Dropout → Linear. As simple as it gets."""

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.dropout(torch.relu(self.layer1(x))))


def sparse_batch(X_sparse, indices) -> torch.Tensor:
    return torch.from_numpy(X_sparse[indices].toarray()).float()


@torch.no_grad()
def predict(model: nn.Module, X_sparse, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    n = X_sparse.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n))
        scores[idx] = torch.softmax(model(sparse_batch(X_sparse, idx)), dim=-1)[:, 1].numpy()
    preds = (scores > 0.5).astype(int)
    return preds, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=("speech", "full"), default="full")
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument(
        "--max-features",
        type=int,
        default=5_000,
        help="capped lower than Baseline 1 to keep the MLP small",
    )
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-features", type=int, default=15)
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "results" / "baseline_mlp_output.md",
        help="markdown file to mirror stdout to",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    buf = StringIO()

    def echo(s: str = "") -> None:
        print(s)
        buf.write(s + "\n")

    splits = load_all_splits()
    texts = {
        s: [flatten_trajectory(r, representation=args.representation) for r in rows]
        for s, rows in splits.items()
    }
    labels = {s: np.array([int(r["is_hacked"]) for r in splits[s]]) for s in splits}

    echo(f"representation={args.representation}  ngram=(1,{args.ngram_max})")
    echo(f"train n={len(texts['train'])}  val n={len(texts['val'])}  test n={len(texts['test'])}")

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
    in_dim = X_train.shape[1]
    echo(f"vocab size: {in_dim:,}")

    model = TinyMLP(in_dim=in_dim, hidden=args.hidden, dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    echo(f"MLP: {in_dim} → {args.hidden} → 2  ({n_params/1e6:.2f}M params, dropout={args.dropout})")
    echo(f"optim: AdamW lr={args.lr} weight_decay={args.weight_decay}  batch={args.batch_size}")
    echo()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)
    n_train = X_train.shape[0]

    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        running_loss, n_b = 0.0, 0
        for start in range(0, n_train, args.batch_size):
            idx = perm[start : start + args.batch_size]
            x = sparse_batch(X_train, idx)
            y = torch.from_numpy(labels["train"][idx]).long()
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_b += 1

        val_preds, val_scores = predict(model, X_val, args.batch_size)
        val_res = evaluate_predictions("val", labels["val"], val_preds, val_scores)
        improved = (val_res.roc_auc or -1.0) > best_val_auc
        marker = "  *" if improved else ""
        echo(
            f"epoch {epoch:>2}/{args.epochs}  "
            f"loss={running_loss/n_b:.3f}  "
            f"val_acc={val_res.accuracy:.3f}  "
            f"val_F1={val_res.f1:.3f}  "
            f"val_AUC={val_res.roc_auc:.3f}{marker}"
        )
        if improved:
            best_val_auc = val_res.roc_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                echo(f"early stop at epoch {epoch} (no val-AUC improvement for {args.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    echo(f"best val AUC={best_val_auc:.3f} at epoch {best_epoch}")
    echo()

    # Final eval on all splits using the best-val checkpoint — same format as TF-IDF baseline
    for split, X in (("train", X_train), ("val", X_val), ("test", X_test)):
        preds, scores = predict(model, X, args.batch_size)
        result = evaluate_predictions(split, labels[split], preds, scores)
        echo(result.format())
        save_predictions(
            baseline=f"mlp_{args.representation}",
            split=split,
            rows=splits[split],
            y_pred=preds,
            y_score=scores,
        )

    # Top features by linearized influence on the (hacked − benign) logit.
    # For an MLP without dropout: y = W2 @ relu(W1 x + b1) + b2.
    # Approximating relu as identity gives effective per-input weight (W2[1]-W2[0]) @ W1.
    # This loses the gating effect of ReLU but gives a useful global feature ranking.
    feature_names = vec.get_feature_names_out()
    W1 = model.layer1.weight.detach().cpu().numpy()  # (hidden, in_dim)
    W2 = model.layer2.weight.detach().cpu().numpy()  # (2, hidden)
    contribution = (W2[1] - W2[0]) @ W1  # (in_dim,)
    top_hacked = contribution.argsort()[::-1][: args.top_features]
    top_benign = contribution.argsort()[: args.top_features]
    echo(f"\nTop {args.top_features} features pulling toward HACKED (linearized influence):")
    for i in top_hacked:
        echo(f"  {contribution[i]:+.3f}  {feature_names[i]!r}")
    echo(f"\nTop {args.top_features} features pulling toward BENIGN (linearized influence):")
    for i in top_benign:
        echo(f"  {contribution[i]:+.3f}  {feature_names[i]!r}")

    args.output_md.write_text(buf.getvalue())
    print(f"\nWritten to {args.output_md}")


if __name__ == "__main__":
    main()
