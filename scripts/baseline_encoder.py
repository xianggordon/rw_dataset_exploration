#!/usr/bin/env python3
"""Baseline 2: fine-tuned long-context encoder for TRACE hack detection.

Default model: ModernBERT-base (8192 token context). Falls back to any
HuggingFace `AutoModelForSequenceClassification`-compatible checkpoint via
``--model``.

Usage:
  --dry-run   Validate the pipeline (load model, tokenize, 1 fwd+bwd pass).
              Does not train. ~1-2 minutes including first-time model download.
  --train     Run full fine-tuning. Recommended on GPU/MPS, painful on CPU.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trace_data.baseline_common import (  # noqa: E402
    evaluate_predictions,
    flatten_trajectory,
    load_all_splits,
    save_predictions,
)


class TrajectoryDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int, tool_result_cap: int | None):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length
        self.tool_result_cap = tool_result_cap

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        text = flatten_trajectory(
            row, representation="full", tool_result_cap_chars=self.tool_result_cap
        )
        enc = self.tok(text, max_length=self.max_length, truncation=True)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": int(row["is_hacked"]),
        }


def get_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_loader(model, loader, device) -> tuple[list[int], list[float], list[int]]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        all_logits.append(out.logits.detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    scores = torch.softmax(logits, dim=-1)[:, 1].numpy()
    preds = (scores > 0.5).astype(int).tolist()
    return preds, scores.tolist(), labels.numpy().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--tool-result-cap",
        type=int,
        default=2000,
        help="cap chars per tool_result (None to disable)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-frac", type=float, default=0.1)
    parser.add_argument("--device", default=None, help="cuda|mps|cpu (auto-detect if omitted)")
    parser.add_argument("--seed", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="validate pipeline; no training")
    mode.add_argument("--train", action="store_true", help="run full fine-tuning")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    print(f"device={device}  model={args.model}  max_length={args.max_length}")
    print(f"tool_result_cap={args.tool_result_cap} chars  batch_size={args.batch_size} (grad_accum={args.grad_accum})")

    print("loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.1f}M")

    splits = load_all_splits()
    datasets = {
        name: TrajectoryDataset(rows, tok, args.max_length, args.tool_result_cap)
        for name, rows in splits.items()
    }
    print(f"train={len(datasets['train'])}  val={len(datasets['val'])}  test={len(datasets['test'])}")

    collator = DataCollatorWithPadding(tok)

    if args.dry_run:
        sample = datasets["train"][0]
        print(f"\nsample input_ids length: {len(sample['input_ids'])}")
        loader = DataLoader(datasets["train"], batch_size=args.batch_size, collate_fn=collator)
        batch = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        print(f"batch input_ids shape: {tuple(batch['input_ids'].shape)}")
        t0 = time.time()
        out = model(**batch)
        print(f"forward pass ok: loss={out.loss.item():.3f}  ({time.time()-t0:.1f}s)")
        t1 = time.time()
        out.loss.backward()
        print(f"backward pass ok ({time.time()-t1:.1f}s)")
        print("\nDry run complete. Use --train to fine-tune.")
        return

    # --- training ---
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, collate_fn=collator)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, collate_fn=collator)

    total_optim_steps = (len(train_loader) * args.epochs) // args.grad_accum
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_optim_steps * args.warmup_frac),
        num_training_steps=total_optim_steps,
    )
    print(f"\ntraining for {args.epochs} epochs, ~{total_optim_steps} optimizer steps")

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss, running_n = 0.0, 0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / args.grad_accum).backward()
            running_loss += out.loss.item()
            running_n += 1
            if (i + 1) % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if (i + 1) % 50 == 0:
                print(
                    f"  epoch {epoch+1} step {i+1}/{len(train_loader)}  "
                    f"loss={running_loss/running_n:.3f}  elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
        preds, scores, labels = evaluate_loader(model, val_loader, device)
        result = evaluate_predictions("val", labels, preds, scores)
        print(f"epoch {epoch+1} val: {result.format()}")

    preds, scores, labels = evaluate_loader(model, test_loader, device)
    result = evaluate_predictions("test", labels, preds, scores)
    print(f"\nFinal test: {result.format()}")

    tag = args.model.replace("/", "_")
    save_predictions(
        baseline=f"encoder_{tag}",
        split="test",
        rows=splits["test"],
        y_pred=preds,
        y_score=scores,
    )


if __name__ == "__main__":
    main()
