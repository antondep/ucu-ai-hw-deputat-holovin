"""Anchor analysis grounded in the box statistics (YOLOv2-style k-means with 1 - IoU distance).

Everything works on box shapes only (w, h), as if anchor and box shared a centre.
Use `scale_to_input` first so that sizes match the resolution the model actually sees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def scale_to_input(boxes: pd.DataFrame, images: pd.DataFrame, long_side: int) -> np.ndarray:
    """(w, h) of every box after resizing its image so the longer side equals `long_side`."""
    b = boxes.merge(images[["key", "width", "height"]], on="key")
    s = (long_side / np.maximum(b.width, b.height)).to_numpy()
    return np.stack([b.w.to_numpy() * s, b.h.to_numpy() * s], 1)


def shape_iou(wh: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """IoU between every box shape and every anchor shape, both centred: [N, K]."""
    inter = np.minimum(wh[:, None, 0], anchors[None, :, 0]) * np.minimum(wh[:, None, 1], anchors[None, :, 1])
    union = wh[:, 0:1] * wh[:, 1:2] + (anchors[:, 0] * anchors[:, 1])[None] - inter
    return inter / union


def kmeans_anchors(wh: np.ndarray, k: int = 9, iters: int = 300, seed: int = 0,
                   sample: int = 100_000) -> tuple[np.ndarray, float]:
    """k-means on box shapes with distance 1 - IoU. Returns anchors sorted by area and mean best IoU."""
    rng = np.random.default_rng(seed)
    x = wh[rng.choice(len(wh), min(sample, len(wh)), replace=False)]
    centers = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        assign = shape_iou(x, centers).argmax(1)
        new = np.array([np.median(x[assign == j], 0) if (assign == j).any() else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    centers = centers[np.argsort(centers.prod(1))]
    return centers, float(shape_iou(wh, centers).max(1).mean())


def anchor_grid(sizes, ratios, scales=(1.0,)) -> np.ndarray:
    """Expand (size, ratio, scale) into (w, h) shapes the way torchvision's AnchorGenerator does:
    ratio = h / w, area = (size * scale)^2."""
    out = []
    for s in sizes:
        for sc in scales:
            for r in ratios:
                a = (s * sc) ** 2
                w = np.sqrt(a / r)
                out.append((w, w * r))
    return np.array(out)


def anchor_coverage(wh: np.ndarray, anchors: np.ndarray, thr: float = 0.5) -> dict:
    """Share of boxes whose best-matching anchor shape reaches IoU >= thr.

    Boxes below the threshold can never become positives under an IoU-based
    assignment (e.g. RetinaNet's 0.5), so they get no direct training signal.
    """
    best = shape_iou(wh, anchors).max(1)
    side = np.sqrt(wh.prod(1))
    out = {"coverage": float((best >= thr).mean()), "mean_best_iou": float(best.mean())}
    for lo, hi, name in [(0, 16, "<16px"), (16, 32, "16-32px"), (32, 96, "32-96px"), (96, 1e9, ">96px")]:
        m = (side >= lo) & (side < hi)
        out[f"coverage_{name}"] = float((best[m] >= thr).mean()) if m.any() else float("nan")
    return out


# torchvision RetinaNet default: 5 FPN levels, 3 octave scales, 3 ratios
TORCHVISION_RETINA_SIZES = (32, 64, 128, 256, 512)
OCTAVE_SCALES = (1.0, 2 ** (1 / 3), 2 ** (2 / 3))
DEFAULT_RATIOS = (0.5, 1.0, 2.0)
