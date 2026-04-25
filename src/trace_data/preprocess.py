import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .schema import Trajectory, Turn

logger = logging.getLogger(__name__)


def _parse_conversation(raw: str) -> list[Turn]:
    """TRACE stores `conversation` as ChatML-shaped JSON (string).

    Shape is either a list of {role, content, ...} messages, or a wrapper
    dict with a `messages`/`conversation` key. We normalize to a list of Turn.
    """
    data = json.loads(raw)
    if isinstance(data, dict):
        for key in ("messages", "conversation", "turns"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError(f"Unrecognized conversation dict keys: {list(data)}")
    if not isinstance(data, list):
        raise ValueError(f"Expected list of messages, got {type(data).__name__}")

    turns: list[Turn] = []
    for msg in data:
        turns.append(
            Turn(
                role=msg.get("role", "unknown"),
                content=_coerce_content(msg.get("content", "")),
                tool_call_id=msg.get("tool_call_id"),
                name=msg.get("name"),
            )
        )
    return turns


def _coerce_content(content: Any) -> str:
    """Some ChatML entries use list-of-parts content; flatten to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or json.dumps(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return json.dumps(content)


def _parse_label(raw: str) -> tuple[bool, list[str]]:
    """TRACE labels: '0' = benign; otherwise comma-separated hack category codes
    like '1.1.2, 1.2.1'. Returns (is_hacked, categories).
    """
    codes = [c.strip() for c in raw.split(",") if c.strip()]
    if codes == ["0"]:
        return False, []
    return True, codes


def _iter_trajectories(df: pd.DataFrame) -> Iterable[Trajectory]:
    for row in df.itertuples(index=False):
        turns = _parse_conversation(row.conversation)
        is_hacked, categories = _parse_label(row.label)
        yield Trajectory(
            trajectory_id=str(row.trajectory_id),
            is_hacked=is_hacked,
            categories=categories,
            turns=turns,
            num_turns=len(turns),
            raw_conversation=row.conversation,
            raw_label=row.label,
        )


def _stratified_split(
    trajectories: list[Trajectory],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[Trajectory]]:
    rng = random.Random(seed)
    by_label: dict[str, list[Trajectory]] = {}
    for t in trajectories:
        by_label.setdefault("hacked" if t.is_hacked else "benign", []).append(t)

    splits: dict[str, list[Trajectory]] = {"train": [], "val": [], "test": []}
    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        splits["test"].extend(items[:n_test])
        splits["val"].extend(items[n_test : n_test + n_val])
        splits["train"].extend(items[n_test + n_val :])
        logger.info(
            "label=%s total=%d train=%d val=%d test=%d",
            label,
            n,
            n - n_test - n_val,
            n_val,
            n_test,
        )
    return splits


def _write_split(name: str, items: list[Trajectory], out_dir: Path) -> Path:
    path = out_dir / f"{name}.jsonl"
    with path.open("w") as f:
        for traj in items:
            f.write(json.dumps(traj.to_dict()) + "\n")
    logger.info("Wrote %d trajectories to %s", len(items), path)
    return path


def preprocess(
    raw_parquet: Path,
    out_dir: Path,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    seed: int = 0,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(raw_parquet)
    logger.info("Loaded raw parquet: %d rows, columns=%s", len(df), list(df.columns))

    trajectories = list(_iter_trajectories(df))
    splits = _stratified_split(trajectories, val_frac, test_frac, seed)
    return {name: _write_split(name, items, out_dir) for name, items in splits.items()}
