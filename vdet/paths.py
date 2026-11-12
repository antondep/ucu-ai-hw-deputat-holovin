"""Find the dataset root so the same code runs on Colab, Kaggle and locally.

The dataset root is the folder that contains train/, val/, test/ and test_images.csv.
Resolution order:
  1. the VDET_DATA environment variable
  2. common locations (/content/data, ./data, ../data)
  3. any folder under /kaggle/input that contains train/annotations_coco.json
"""
from __future__ import annotations

import os
from pathlib import Path

_MARKER = Path("train") / "annotations_coco.json"


def _is_root(p: Path) -> bool:
    return (p / _MARKER).exists()


def find_data_root(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("VDET_DATA"):
        candidates.append(Path(os.environ["VDET_DATA"]))
    candidates += [Path("/content/data"), Path("data"), Path("../data")]

    for c in candidates:
        if _is_root(c):
            return c.resolve()
        # the archive is sometimes extracted one level deeper
        for sub in c.glob("*") if c.is_dir() else []:
            if _is_root(sub):
                return sub.resolve()

    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for p in sorted(kaggle.glob("**/annotations_coco.json")):
            if p.parent.name == "train":
                return p.parent.parent.resolve()

    raise FileNotFoundError(
        "Dataset not found. Set VDET_DATA to the folder that contains train/, val/, test/."
    )


def default_work_dir() -> Path:
    """Where to write splits, exported datasets, checkpoints and predictions."""
    if os.environ.get("VDET_WORK"):
        p = Path(os.environ["VDET_WORK"])
    elif Path("/kaggle/working").exists():
        p = Path("/kaggle/working")
    elif Path("/content").exists():
        p = Path("/content/work")
    else:
        p = Path("work")
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()
