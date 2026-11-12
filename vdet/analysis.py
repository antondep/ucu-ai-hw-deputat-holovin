"""Failure analysis: a simple TIDE-style breakdown of what goes wrong.

Every confident detection (score >= thr) is assigned to ONE error type, greedily by score:
  tp          IoU >= 0.5 with an unmatched GT of the same class
  duplicate   IoU >= 0.5 with a same-class GT that is already matched
  class       IoU >= 0.5 with a GT of another class (class confusion)
  location    0.1 <= best IoU < 0.5 with any GT (right object, poor box)
  ignored     not near any GT but mostly inside an ignored region (not penalised by the metric)
  background  none of the above (false alarm on background)
Every GT left unmatched is a miss, reported by size bucket.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import CLASSES
from .data import SplitData, overlap_fraction

SIZE_BINS = [0, 16, 32, 96, 1e9]
SIZE_LABELS = ["<16px", "16-32px", "32-96px", ">96px"]


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between [N, 4] and [M, 4] xywh boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None] - inter + 1e-9)


def error_breakdown(preds: pd.DataFrame, data: SplitData, score_thr: float = 0.3) -> dict:
    """Returns {'detections': counts per type, 'confusions': class pairs, 'misses': misses by size and class,
    'per_frame': recall and false positives per frame}."""
    P = preds[preds.score >= score_thr].sort_values("score", ascending=False)
    pg = {k: g for k, g in P.groupby("key")}
    gg = {k: g for k, g in data.boxes.groupby("key")}
    ig = {k: g[["x", "y", "w", "h"]].to_numpy() for k, g in data.ignore.groupby("key")}
    kinds, conf, missed, frames = [], [], [], []
    for key in data.images.key:
        g = gg.get(key); p = pg.get(key)
        gb = g[["x", "y", "w", "h"]].to_numpy() if g is not None else np.zeros((0, 4))
        gc = g["cat"].to_numpy() if g is not None else np.zeros(0, int)
        matched = np.zeros(len(gb), bool)
        n_fp = 0
        if p is not None:
            pb, pc = p[["x", "y", "w", "h"]].to_numpy(), p["cat"].to_numpy().astype(int)
            iou = _iou_matrix(pb, gb)
            inside = overlap_fraction(pb, ig[key]) if key in ig else np.zeros(len(pb))
            for i in range(len(pb)):
                same = np.where(gc == pc[i], iou[i], 0.0) if len(gb) else np.zeros(0)
                free = np.where(~matched, same, 0.0) if len(gb) else same
                if len(gb) and free.max() >= 0.5:
                    j = int(free.argmax()); matched[j] = True; kinds.append("tp"); continue
                if len(gb) and same.max() >= 0.5:
                    kinds.append("duplicate"); n_fp += 1; continue
                if len(gb) and iou[i].max() >= 0.5:
                    j = int(iou[i].argmax()); kinds.append("class"); conf.append((CLASSES[gc[j]], CLASSES[pc[i]])); n_fp += 1; continue
                if len(gb) and iou[i].max() >= 0.1:
                    kinds.append("location"); n_fp += 1; continue
                if inside[i] > 0.5:
                    kinds.append("ignored"); continue
                kinds.append("background"); n_fp += 1
        if len(gb):
            side = np.sqrt(gb[~matched, 2] * gb[~matched, 3])
            for s, c in zip(side, gc[~matched]):
                missed.append((s, CLASSES[c]))
        frames.append({"key": key, "n_gt": len(gb), "recall": matched.mean() if len(gb) else np.nan, "false_pos": n_fp})

    det = pd.Series(kinds).value_counts().rename("count").to_frame()
    det["share_%"] = (det["count"] / det["count"].sum() * 100).round(1)
    cf = (pd.DataFrame(conf, columns=["true", "predicted"]).value_counts().rename("count").reset_index()
          if conf else pd.DataFrame(columns=["true", "predicted", "count"]))
    gt_side = np.sqrt(data.boxes.w * data.boxes.h)
    total_by_size = pd.cut(gt_side, SIZE_BINS, labels=SIZE_LABELS).value_counts().sort_index()
    ms = pd.DataFrame(missed, columns=["side", "cls"])
    miss_by_size = pd.cut(ms.side, SIZE_BINS, labels=SIZE_LABELS).value_counts().sort_index() if len(ms) else total_by_size * 0
    misses = pd.DataFrame({"n_gt": total_by_size, "missed": miss_by_size})
    misses["miss_rate_%"] = (misses["missed"] / misses["n_gt"] * 100).round(1)
    miss_cls = (ms.cls.value_counts() / data.boxes.cat.map(dict(enumerate(CLASSES))).value_counts() * 100).round(1)
    return {"detections": det, "confusions": cf, "misses": misses,
            "miss_rate_by_class_%": miss_cls.rename("miss_rate_%").sort_values(ascending=False).to_frame(),
            "per_frame": pd.DataFrame(frames)}
