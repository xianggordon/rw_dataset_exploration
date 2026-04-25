#!/usr/bin/env python3
"""End-to-end: download TRACE, then preprocess into train/val/test JSONL."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trace_data import download_raw, preprocess  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--split", default="train", help="HF split to download")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv(REPO_ROOT / ".env")

    raw_dir = args.data_dir / "raw"
    processed_dir = args.data_dir / "processed"
    raw_parquet = raw_dir / f"{args.split}.parquet"

    if not args.skip_download or not raw_parquet.exists():
        raw_parquet = download_raw(raw_dir, split=args.split)

    preprocess(
        raw_parquet=raw_parquet,
        out_dir=processed_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
