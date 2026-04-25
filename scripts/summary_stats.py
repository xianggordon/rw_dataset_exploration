#!/usr/bin/env python3
"""Summary statistics + sample pretty-prints for the processed TRACE splits."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def load_split(processed_dir: Path, split: str) -> list[dict]:
    path = processed_dir / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _fmt_num(n: float) -> str:
    return f"{n:,.1f}" if isinstance(n, float) else f"{n:,}"


def _print_header(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def print_split_table(all_rows: dict[str, list[dict]]) -> None:
    _print_header("Split composition")
    print(f"{'split':<10}{'n':>8}{'hacked':>10}{'benign':>10}{'%hacked':>10}")
    for split in SPLITS:
        rows = all_rows[split]
        n = len(rows)
        h = sum(r["is_hacked"] for r in rows)
        print(f"{split:<10}{n:>8}{h:>10}{n - h:>10}{100 * h / n:>9.1f}%")
    total = [r for split in SPLITS for r in all_rows[split]]
    h = sum(r["is_hacked"] for r in total)
    print(f"{'total':<10}{len(total):>8}{h:>10}{len(total) - h:>10}{100 * h / len(total):>9.1f}%")


def print_turn_stats(all_rows: dict[str, list[dict]]) -> None:
    _print_header("Turns per trajectory")
    rows = [r for split in SPLITS for r in all_rows[split]]
    by_label = {
        "all": [r["num_turns"] for r in rows],
        "hacked": [r["num_turns"] for r in rows if r["is_hacked"]],
        "benign": [r["num_turns"] for r in rows if not r["is_hacked"]],
    }
    print(f"{'group':<10}{'min':>8}{'p25':>8}{'med':>8}{'mean':>10}{'p75':>8}{'max':>8}")
    for name, xs in by_label.items():
        xs_sorted = sorted(xs)
        q = statistics.quantiles(xs_sorted, n=4)
        print(
            f"{name:<10}{min(xs):>8}{int(q[0]):>8}{int(q[1]):>8}"
            f"{statistics.mean(xs):>10.1f}{int(q[2]):>8}{max(xs):>8}"
        )


def print_role_distribution(all_rows: dict[str, list[dict]]) -> None:
    _print_header("Role distribution across all turns")
    counter: Counter[str] = Counter()
    for split in SPLITS:
        for r in all_rows[split]:
            for t in r["turns"]:
                counter[t["role"]] += 1
    total = sum(counter.values())
    print(f"{'role':<15}{'count':>10}{'%':>8}")
    for role, c in counter.most_common():
        print(f"{role:<15}{c:>10,}{100 * c / total:>7.1f}%")


def print_category_stats(all_rows: dict[str, list[dict]]) -> None:
    _print_header("Hack categories (hacked trajectories only)")
    cat_counter: Counter[str] = Counter()
    multi_label = 0
    hacked_total = 0
    for split in SPLITS:
        for r in all_rows[split]:
            if not r["is_hacked"]:
                continue
            hacked_total += 1
            if len(r["categories"]) > 1:
                multi_label += 1
            for c in r["categories"]:
                cat_counter[c] += 1
    print(f"hacked trajectories: {hacked_total}")
    print(f"multi-label (>=2 categories): {multi_label} ({100 * multi_label / hacked_total:.1f}%)")
    print(f"unique categories: {len(cat_counter)}")
    print()
    print(f"{'category':<12}{'count':>8}{'% of hacked':>14}")
    for cat, c in cat_counter.most_common():
        print(f"{cat:<12}{c:>8}{100 * c / hacked_total:>13.1f}%")


def print_length_stats(all_rows: dict[str, list[dict]]) -> None:
    _print_header("Conversation text length (sum of turn content chars)")
    rows = [r for split in SPLITS for r in all_rows[split]]
    lens_all = [sum(len(t["content"]) for t in r["turns"]) for r in rows]
    lens_hacked = [lens_all[i] for i, r in enumerate(rows) if r["is_hacked"]]
    lens_benign = [lens_all[i] for i, r in enumerate(rows) if not r["is_hacked"]]
    print(f"{'group':<10}{'min':>8}{'med':>10}{'mean':>12}{'max':>10}")
    for name, xs in [("all", lens_all), ("hacked", lens_hacked), ("benign", lens_benign)]:
        print(
            f"{name:<10}{min(xs):>8,}{int(statistics.median(xs)):>10,}"
            f"{statistics.mean(xs):>12,.0f}{max(xs):>10,}"
        )


def make_figures(all_rows: dict[str, list[dict]], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for split in SPLITS for r in all_rows[split]]
    turns_h = [r["num_turns"] for r in rows if r["is_hacked"]]
    turns_b = [r["num_turns"] for r in rows if not r["is_hacked"]]

    # 1. Label split stacked bar
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(SPLITS)
    hacked = [sum(r["is_hacked"] for r in all_rows[s]) for s in SPLITS]
    benign = [len(all_rows[s]) - h for h, s in zip(hacked, SPLITS)]
    ax.bar(names, benign, label="benign", color="#4c72b0")
    ax.bar(names, hacked, bottom=benign, label="hacked", color="#c44e52")
    ax.set_ylabel("trajectories")
    ax.set_title("TRACE split composition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "01_split_composition.png", dpi=140)
    plt.close(fig)

    # 2. Turn count histogram (overlaid)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = range(min(turns_h + turns_b), max(turns_h + turns_b) + 2)
    ax.hist(turns_b, bins=bins, alpha=0.6, label="benign", color="#4c72b0")
    ax.hist(turns_h, bins=bins, alpha=0.6, label="hacked", color="#c44e52")
    ax.set_xlabel("turns per trajectory")
    ax.set_ylabel("count")
    ax.set_title("Trajectory length by label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "02_turn_distribution.png", dpi=140)
    plt.close(fig)

    # 3. Category frequency bar
    cat_counter: Counter[str] = Counter()
    for r in rows:
        if r["is_hacked"]:
            for c in r["categories"]:
                cat_counter[c] += 1
    if cat_counter:
        cats, counts = zip(*sorted(cat_counter.items()))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(cats, counts, color="#c44e52")
        ax.set_xlabel("category code")
        ax.set_ylabel("hacked trajectories")
        ax.set_title("Hack category frequency (multi-label)")
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        fig.savefig(fig_dir / "03_category_frequency.png", dpi=140)
        plt.close(fig)

    # 4. Char-length boxplot
    lens_h = [sum(len(t["content"]) for t in r["turns"]) for r in rows if r["is_hacked"]]
    lens_b = [sum(len(t["content"]) for t in r["turns"]) for r in rows if not r["is_hacked"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([lens_b, lens_h], tick_labels=["benign", "hacked"], showfliers=False)
    ax.set_ylabel("conversation length (chars)")
    ax.set_title("Content length by label (outliers hidden)")
    fig.tight_layout()
    fig.savefig(fig_dir / "04_length_boxplot.png", dpi=140)
    plt.close(fig)

    print(f"\nSaved 4 figures to {fig_dir}/")


def pretty_print_processed(row: dict) -> None:
    print(f"trajectory_id: {row['trajectory_id']}")
    print(f"is_hacked:     {row['is_hacked']}")
    print(f"categories:    {row['categories'] or '[]'}")
    print(f"raw_label:     {row['raw_label']!r}")
    print(f"num_turns:     {row['num_turns']}")
    print(f"raw_conversation: <string, {len(row['raw_conversation']):,} chars, not shown>")
    print("turns:")
    for i, t in enumerate(row["turns"]):
        meta = ""
        if t.get("tool_call_id"):
            meta += f" tool_call_id={t['tool_call_id']}"
        if t.get("name"):
            meta += f" name={t['name']}"
        print(f"  [{i:02d}] role={t['role']}{meta}")
        for line in textwrap.indent(t["content"], "       ").splitlines() or ["       "]:
            print(line)


def pretty_print_raw_conversation(raw: str) -> None:
    messages = json.loads(raw)
    for i, m in enumerate(messages):
        extras = {k: v for k, v in m.items() if k not in ("role", "content")}
        extra_str = f"  extras={list(extras.keys())}" if extras else ""
        print(f"  [{i:02d}] role={m.get('role')}{extra_str}")
        content = m.get("content") or ""
        if isinstance(content, (list, dict)):
            content = json.dumps(content, indent=2)
        for line in textwrap.indent(str(content), "       ").splitlines() or ["       "]:
            print(line)
        for key, val in extras.items():
            val_str = json.dumps(val, indent=2) if not isinstance(val, str) else val
            print(f"       -- {key} --")
            for line in textwrap.indent(val_str, "         ").splitlines():
                print(line)


def print_samples(all_rows: dict[str, list[dict]], seed: int) -> None:
    pool = all_rows["train"]
    hacked = [r for r in pool if r["is_hacked"]]
    benign = [r for r in pool if not r["is_hacked"]]
    rng = random.Random(seed)
    sample_h = rng.choice(hacked)
    sample_b = rng.choice(benign)

    for label, row in (("HACKED", sample_h), ("BENIGN", sample_b)):
        _print_header(f"Sample {label} trajectory — processed row")
        pretty_print_processed(row)
        _print_header(f"Sample {label} trajectory — full raw conversation")
        pretty_print_raw_conversation(row["raw_conversation"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "data" / "figures")
    parser.add_argument("--seed", type=int, default=0, help="sample-selection seed")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    all_rows = {s: load_split(args.processed_dir, s) for s in SPLITS}

    print_split_table(all_rows)
    print_turn_stats(all_rows)
    print_role_distribution(all_rows)
    print_category_stats(all_rows)
    print_length_stats(all_rows)
    if not args.no_figures:
        make_figures(all_rows, args.fig_dir)
    print_samples(all_rows, seed=args.seed)


if __name__ == "__main__":
    main()
