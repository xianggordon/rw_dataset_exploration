import logging
import os
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import login

logger = logging.getLogger(__name__)

REPO_ID = "PatronusAI/trace-dataset"
DEFAULT_SPLIT = "train"


def _authenticate() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. TRACE is gated — accept terms at "
            f"https://huggingface.co/datasets/{REPO_ID} and export a token."
        )
    login(token=token, add_to_git_credential=False)


def download_raw(
    out_dir: Path,
    split: str = DEFAULT_SPLIT,
    cache_dir: Path | None = None,
) -> Path:
    """Fetch TRACE from the Hub and persist the raw split as Parquet.

    Returns the path to the written Parquet file.
    """
    _authenticate()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s:%s from Hugging Face", REPO_ID, split)
    ds: Dataset = load_dataset(
        REPO_ID,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    target = out_dir / f"{split}.parquet"
    ds.to_parquet(str(target))
    logger.info("Wrote %d rows to %s", len(ds), target)
    return target
