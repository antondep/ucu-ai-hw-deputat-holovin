"""Write and validate the Kaggle submission.

Format: one row per test image,
    image_id,PredictionString
    clip_001_0000001,2 0.91 1366 580 26 44 0 0.55 809 625 29 66
groups of `class score x y w h`, class 0-based, (x, y) = top-left corner in pixels of the
ORIGINAL image. Empty string for images without detections. Max 500 detections are scored.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import NUM_CLASSES

MAX_DETS = 500


def submission_order(data_root: str | Path) -> pd.Series:
    """Row order of the submission: sample_submission.csv if the competition ships it,
    otherwise test_images.csv (both list the same image ids)."""
    data_root = Path(data_root)
    f = data_root / "sample_submission.csv"
    if not f.exists():
        f = data_root / "test_images.csv"
    return pd.read_csv(f)["image_id"]


def clean_predictions(preds: pd.DataFrame, sizes: pd.DataFrame, max_dets: int = MAX_DETS,
                      min_size: float = 1.0) -> pd.DataFrame:
    """Clip boxes to the image, drop degenerate ones, keep the top `max_dets` per image.

    sizes: DataFrame with key, width, height for every image.
    """
    p = preds.merge(sizes[["key", "width", "height"]], on="key", how="inner")
    x1 = p.x.clip(0, p.width)
    y1 = p.y.clip(0, p.height)
    x2 = (p.x + p.w).clip(0, p.width)
    y2 = (p.y + p.h).clip(0, p.height)
    p = p.assign(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
    p = p[(p.w >= min_size) & (p.h >= min_size) & p.cat.between(0, NUM_CLASSES - 1)]
    p = p.sort_values(["key", "score"], ascending=[True, False]).groupby("key").head(max_dets)
    return p[["key", "cat", "score", "x", "y", "w", "h"]].reset_index(drop=True)


def _format(g: pd.DataFrame) -> str:
    return " ".join(f"{int(c)} {s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}"
                    for c, s, x, y, w, h in g[["cat", "score", "x", "y", "w", "h"]].itertuples(index=False))


def write_submission(preds: pd.DataFrame, data_root: str | Path, out_path: str | Path,
                     max_dets: int = MAX_DETS) -> pd.DataFrame:
    """Build the submission in sample_submission.csv order and save it without an index."""
    data_root = Path(data_root)
    test = pd.read_csv(data_root / "test_images.csv").rename(columns={"image_id": "key"})
    order = submission_order(data_root)
    p = clean_predictions(preds, test, max_dets)
    strings = pd.Series({k: _format(g) for k, g in p.groupby("key")}, dtype=object)
    sub = pd.DataFrame({"image_id": order})
    sub["PredictionString"] = sub["image_id"].map(strings).fillna("")
    check_submission(sub, data_root)
    sub.to_csv(out_path, index=False)
    print(f"saved {out_path}: {len(sub)} rows, {len(p)} detections, "
          f"{(sub.PredictionString == '').sum()} empty rows")
    return sub


def check_submission(sub: pd.DataFrame, data_root: str | Path) -> None:
    """Raise if the submission would be rejected or silently mis-scored."""
    data_root = Path(data_root)
    order = submission_order(data_root)
    sizes = pd.read_csv(data_root / "test_images.csv").set_index("image_id")
    assert list(sub.columns) == ["image_id", "PredictionString"], sub.columns
    assert sub.image_id.tolist() == order.tolist(), "rows must follow the sample submission order"
    for iid, s in zip(sub.image_id, sub.PredictionString.fillna("")):
        if not s:
            continue
        v = np.array(s.split(), dtype=float).reshape(-1, 6)
        W, H = sizes.loc[iid, "width"], sizes.loc[iid, "height"]
        assert len(v) <= MAX_DETS, f"{iid}: {len(v)} > {MAX_DETS} detections"
        assert np.isfinite(v).all(), iid
        assert ((v[:, 0] >= 0) & (v[:, 0] < NUM_CLASSES)).all(), f"{iid}: bad class id"
        assert (v[:, 2] >= 0).all() and (v[:, 3] >= 0).all(), f"{iid}: negative x/y"
        assert (v[:, 2] + v[:, 4] <= W + 1).all() and (v[:, 3] + v[:, 5] <= H + 1).all(), f"{iid}: box outside image"


def read_submission(path: str | Path) -> pd.DataFrame:
    """Parse a submission back into the prediction table (useful for ensembling saved runs)."""
    sub = pd.read_csv(path).fillna("")
    rows = []
    for iid, s in zip(sub.image_id, sub.PredictionString):
        if s:
            v = np.array(s.split(), dtype=float).reshape(-1, 6)
            rows.append(pd.DataFrame(v, columns=["cat", "score", "x", "y", "w", "h"]).assign(key=iid))
    if not rows:
        return pd.DataFrame(columns=["key", "cat", "score", "x", "y", "w", "h"])
    out = pd.concat(rows, ignore_index=True)
    out["cat"] = out["cat"].astype(int)
    return out[["key", "cat", "score", "x", "y", "w", "h"]]
