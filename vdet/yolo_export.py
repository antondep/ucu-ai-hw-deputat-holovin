"""Build an Ultralytics-ready dataset from a SplitData selection.

Why not use the provided train/data.yaml directly?
  1. It contains every frame; neighbouring frames are near duplicates, so we train on a
     subsampled set (step chosen from the temporal analysis in the EDA notebook).
  2. Ultralytics has no notion of ignored regions. Unmasked, objects inside crowds and
     parking lots would be learned as background. We paint ignored regions with the
     image's mean colour and drop GT boxes mostly covered by them (VisDrone practice).
  3. Our train/val assignment is by clip and must match the shared evaluator.

Layout written:
  out_dir/images/{train,val}/<key>.jpg   (masked and/or downscaled copy; symlink if neither is needed)
  out_dir/labels/{train,val}/<key>.txt   (class cx cy w h, normalised)
  out_dir/data.yaml
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import CLASSES
from .data import SplitData, overlap_fraction


def mask_ignored(img: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Fill [x, y, w, h] pixel regions with the image mean colour."""
    out = img.copy()
    fill = img.reshape(-1, img.shape[2]).mean(0).astype(img.dtype)
    H, W = img.shape[:2]
    for x, y, w, h in regions:
        x1, y1 = max(int(np.floor(x)), 0), max(int(np.floor(y)), 0)
        x2, y2 = min(int(np.ceil(x + w)), W), min(int(np.ceil(y + h)), H)
        out[y1:y2, x1:x2] = fill
    return out


def export_split(data: SplitData, out_dir: str | Path, subset: str, mask: bool = True,
                 drop_thr: float = 0.5, max_side: int | None = None) -> dict:
    """Write one subset ('train' or 'val'). Returns counts for the report.

    max_side: downscale frames whose longer side exceeds it. Labels are normalised, so they
    do not change. Decoding 2.7K JPEGs every epoch is the bottleneck on Kaggle's 4 CPUs, and the
    model resizes to imgsz anyway, so pre-resizing to imgsz loses nothing.
    """
    out_dir = Path(out_dir)
    img_dir, lab_dir = out_dir / "images" / subset, out_dir / "labels" / subset
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    boxes = {k: g for k, g in data.boxes.groupby("key")}
    ignore = {k: g[["x", "y", "w", "h"]].to_numpy() for k, g in data.ignore.groupby("key")}
    stats = dict(images=0, masked_images=0, boxes=0, dropped_boxes=0)

    for r in tqdm(data.images.itertuples(), total=len(data.images), desc=f"export {subset}"):
        b = boxes.get(r.key, data.boxes.iloc[:0])
        xywh = b[["x", "y", "w", "h"]].to_numpy()
        cls = b["cat"].to_numpy()
        reg = ignore.get(r.key) if mask else None
        dst = img_dir / f"{r.key}.jpg"

        has_reg = reg is not None and len(reg) > 0
        if has_reg:
            keep = overlap_fraction(xywh, reg) < drop_thr
            stats["dropped_boxes"] += int((~keep).sum())
            xywh, cls = xywh[keep], cls[keep]
            stats["masked_images"] += 1
        scale = min(1.0, max_side / max(r.width, r.height)) if max_side else 1.0
        if not dst.exists():
            if has_reg or scale < 1.0:
                img = cv2.imread(r.path)
                if has_reg:
                    img = mask_ignored(img, reg)
                if scale < 1.0:
                    img = cv2.resize(img, (round(r.width * scale), round(r.height * scale)),
                                     interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                os.symlink(os.path.abspath(r.path), dst)

        W, H = float(r.width), float(r.height)
        lines = [f"{c} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
                 for c, (x, y, w, h) in zip(cls, xywh)]
        (lab_dir / f"{r.key}.txt").write_text("\n".join(lines))
        stats["images"] += 1
        stats["boxes"] += len(lines)
    return stats


def export_yolo_dataset(train: SplitData, val: SplitData, out_dir: str | Path,
                        mask_train: bool = True, mask_val: bool = False,
                        max_side: int | None = None) -> pd.DataFrame:
    """Export train/val and write data.yaml.

    Validation images are NOT masked by default: Ultralytics' built-in val metric is only
    a training monitor, the numbers we report come from vdet.evaluate on original frames,
    where ignore regions are handled as iscrowd.
    """
    out_dir = Path(out_dir)
    st = {"train": export_split(train, out_dir, "train", mask_train, max_side=max_side),
          "val": export_split(val, out_dir, "val", mask_val, max_side=max_side)}
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return pd.DataFrame(st).T
