"""Load VisDrone-VID annotations into pandas tables and build leak-free splits.

Every split is represented by three tables:
  images : one row per frame  (key, split, clip, frame, width, height, path, coco_id)
  boxes  : one row per object (key, cat [0-based], x, y, w, h, area, occlusion, truncation, track_id, ...)
  ignore : one row per ignored region (key, x, y, w, h)

`key` is "<clip>_<frame stem>", e.g. "clip_001_0000001", the same format as image_id
in test_images.csv, so predictions on train/val/test are handled identically.
Boxes are always in pixels, (x, y) = top-left corner.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


@dataclass
class SplitData:
    images: pd.DataFrame
    boxes: pd.DataFrame
    ignore: pd.DataFrame

    def subset(self, keys) -> "SplitData":
        keys = set(keys)
        return SplitData(
            self.images[self.images.key.isin(keys)].reset_index(drop=True),
            self.boxes[self.boxes.key.isin(keys)].reset_index(drop=True),
            self.ignore[self.ignore.key.isin(keys)].reset_index(drop=True),
        )

    def __repr__(self) -> str:
        return (f"SplitData(images={len(self.images)}, boxes={len(self.boxes)}, "
                f"ignore={len(self.ignore)}, clips={self.images['clip'].nunique()})")


# --------------------------------------------------------------------------- loading
def _resolve_image_path(split_root: Path, file_name: str) -> Path:
    """COCO file_name may be relative to the split folder, to images/, or to the data root."""
    for base in (split_root, split_root / "images", split_root.parent):
        p = base / file_name
        if p.exists():
            return p
    return split_root / "images" / file_name  # best guess, existence checked later


def load_split(data_root: str | Path, split: str) -> SplitData:
    """Read <data_root>/<split>/annotations_coco.json into a SplitData."""
    data_root = Path(data_root)
    split_root = data_root / split
    with open(split_root / "annotations_coco.json") as f:
        coco = json.load(f)

    # --- images
    im = pd.DataFrame(coco["images"]).rename(columns={"id": "coco_id"})
    fn = im["file_name"].map(lambda s: Path(s))
    im["clip"] = im["video"] if "video" in im else fn.map(lambda p: p.parent.name)
    stem = fn.map(lambda p: p.stem)
    im["frame"] = (im["frame_id"] if "frame_id" in im else stem).astype(int)
    im["key"] = im["clip"].astype(str) + "_" + stem
    im["split"] = split
    # resolve the path pattern once on the first image, then apply to all
    first = _resolve_image_path(split_root, im["file_name"].iloc[0])
    prefix = Path(str(first)[: len(str(first)) - len(im["file_name"].iloc[0])])
    im["path"] = im["file_name"].map(lambda s: str(prefix / s))
    images = im[["key", "split", "clip", "frame", "width", "height", "path", "coco_id"]]
    images = images.sort_values(["clip", "frame"]).reset_index(drop=True)

    # --- annotations
    an = pd.DataFrame(coco["annotations"])
    id2key = dict(zip(im["coco_id"], im["key"]))
    an["key"] = an["image_id"].map(id2key)
    xywh = np.array(an["bbox"].tolist(), dtype=np.float64).reshape(-1, 4)
    an[["x", "y", "w", "h"]] = xywh
    crowd = an.get("iscrowd", pd.Series(0, index=an.index)).fillna(0).astype(int) == 1

    extra = [c for c in ("occlusion", "truncation", "track_id", "orig_category_id") if c in an]
    boxes = an.loc[~crowd, ["key", "category_id", "x", "y", "w", "h", *extra]].copy()
    boxes["cat"] = boxes.pop("category_id").astype(int) - 1  # COCO is 1-based -> 0-based
    boxes["area"] = boxes.w * boxes.h
    # The json stores every ignored region once per class (8 identical copies with category_id 1..8),
    # the same trick pycocotools needs for class-agnostic crowd regions. Keep one copy per region;
    # the evaluator re-creates the per-class copies itself.
    ign_cols = ["key", "x", "y", "w", "h"] + [c for c in ("orig_category_id",) if c in an]
    ignore = (an.loc[crowd, ign_cols]
              .drop_duplicates(["key", "x", "y", "w", "h"])
              .reset_index(drop=True))

    return SplitData(images, boxes.reset_index(drop=True), ignore)


def load_train_val(data_root: str | Path) -> SplitData:
    """Train and val concatenated; the `split` column keeps the official assignment."""
    tr, va = load_split(data_root, "train"), load_split(data_root, "val")
    return SplitData(
        pd.concat([tr.images, va.images], ignore_index=True),
        pd.concat([tr.boxes, va.boxes], ignore_index=True),
        pd.concat([tr.ignore, va.ignore], ignore_index=True),
    )


def load_test_images(data_root: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_root) / "test_images.csv")
    df = df.rename(columns={"image_id": "key"})
    df["path"] = df["file_name"].map(lambda s: str(Path(data_root) / s))
    df["split"] = "test"
    return df


# --------------------------------------------------------------------------- YOLO txt files
def read_yolo_txt(path: Path, width: int, height: int, has_class: bool = True) -> np.ndarray:
    """Read a normalised YOLO txt file into pixel [cls, x, y, w, h] (top-left) rows."""
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5 if has_class else 4))
    a = np.loadtxt(path, ndmin=2)
    off = 1 if has_class else 0
    cx, cy, w, h = (a[:, off] * width, a[:, off + 1] * height, a[:, off + 2] * width, a[:, off + 3] * height)
    out = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
    return np.concatenate([a[:, :1], out], 1) if has_class else out


def verify_yolo_vs_coco(data: SplitData, data_root: str | Path, split: str,
                        n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Compare the YOLO txt labels with the COCO json on n random frames.

    Returns one row per frame with object counts from both sources and the largest
    coordinate disagreement in pixels (should be < 1 px, i.e. rounding only).
    """
    root = Path(data_root) / split
    imgs = data.images[data.images.split == split]
    imgs = imgs.sample(min(n, len(imgs)), random_state=seed)
    bx = {k: g for k, g in data.boxes.groupby("key")}
    ig = {k: g for k, g in data.ignore.groupby("key")}
    rows = []
    for r in imgs.itertuples():
        stem = Path(r.path).stem
        yl = read_yolo_txt(root / "labels" / r.clip / f"{stem}.txt", r.width, r.height)
        yi = read_yolo_txt(root / "ignore" / r.clip / f"{stem}.txt", r.width, r.height, has_class=False)
        cb = bx.get(r.key, data.boxes.iloc[:0])
        ci = ig.get(r.key, data.ignore.iloc[:0])
        err = np.nan
        if len(yl) == len(cb) and len(cb):
            a = np.sort(yl[:, 1:], axis=0)
            b = np.sort(cb[["x", "y", "w", "h"]].to_numpy(), axis=0)
            err = float(np.abs(a - b).max())
        # ignore regions: which ones appear in both sources (same box within 1 px)?
        cxy = ci[["x", "y", "w", "h"]].to_numpy()
        same = (np.abs(yi[:, None, :] - cxy[None, :, :]).max(-1) < 1.0) if len(yi) and len(cxy) \
            else np.zeros((len(yi), len(cxy)), bool)
        coco_only = ci[~same.any(0)] if len(cxy) else ci
        dup = int(ci.duplicated(["x", "y", "w", "h"]).sum())
        rows.append(dict(key=r.key, yolo_objs=len(yl), coco_objs=len(cb),
                         yolo_ignore=len(yi), coco_ignore=len(ci),
                         yolo_ign_in_coco=int(same.any(1).sum()), coco_ign_in_yolo=int(same.any(0).sum()),
                         coco_ign_duplicates=dup,
                         coco_only_orig_cats=coco_only["orig_category_id"].tolist() if "orig_category_id" in ci else [],
                         max_px_err=err))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- splits
def make_clip_folds(images: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Assign every clip to a fold with GroupKFold (groups = clip), balancing frame counts.

    All frames of a clip land in the same fold, so no near-duplicate frames leak
    between training and validation.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold = np.full(len(images), -1)
    for f, (_, va) in enumerate(gkf.split(images, groups=images["clip"])):
        fold[va] = f
    out = images[["clip"]].assign(fold=fold).drop_duplicates("clip")
    return out.sort_values("clip").reset_index(drop=True)


def subsample_frames(images: pd.DataFrame, step: int, offset: int = 0) -> pd.DataFrame:
    """Keep every `step`-th frame of each clip (frame order), starting at `offset`."""
    if step <= 1:
        return images.copy()
    im = images.sort_values(["clip", "frame"])
    rank = im.groupby("clip").cumcount()
    return im[(rank % step) == offset].reset_index(drop=True)


# --------------------------------------------------------------------------- geometry helpers
def box_iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise IoU between two aligned arrays of [x, y, w, h] boxes."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter + 1e-9)


def overlap_fraction(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """For each box, the largest fraction of its area covered by any single region."""
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes))
    b, r = boxes[:, None, :], regions[None, :, :]
    iw = np.clip(np.minimum(b[..., 0] + b[..., 2], r[..., 0] + r[..., 2]) - np.maximum(b[..., 0], r[..., 0]), 0, None)
    ih = np.clip(np.minimum(b[..., 1] + b[..., 3], r[..., 1] + r[..., 3]) - np.maximum(b[..., 1], r[..., 1]), 0, None)
    return ((iw * ih) / (boxes[:, None, 2] * boxes[:, None, 3] + 1e-9)).max(1)


# --------------------------------------------------------------------------- temporal analysis
def track_iou_vs_gap(data: SplitData, gaps=(1, 2, 5, 10, 20, 50)) -> pd.DataFrame:
    """How similar is the same object `gap` frames later? Used to justify frame subsampling.

    For every track present in frames t and t+gap, compute the IoU of its two boxes.
    High IoU at gap k means frames k apart carry almost the same supervision.
    """
    if "track_id" not in data.boxes:
        raise ValueError("annotations have no track_id")
    b = data.boxes.merge(data.images[["key", "clip", "frame"]], on="key")
    b = b[["clip", "track_id", "frame", "x", "y", "w", "h"]]
    rows = []
    for g in gaps:
        nxt = b.assign(frame=b.frame - g)
        m = b.merge(nxt, on=["clip", "track_id", "frame"], suffixes=("", "_n"))
        iou = box_iou_xywh(m[["x", "y", "w", "h"]].to_numpy(), m[["x_n", "y_n", "w_n", "h_n"]].to_numpy())
        rows.append(dict(gap=g, pairs=len(m), median_iou=np.median(iou) if len(iou) else np.nan,
                         frac_iou_gt_0_7=(iou > 0.7).mean() if len(iou) else np.nan))
    return pd.DataFrame(rows)
