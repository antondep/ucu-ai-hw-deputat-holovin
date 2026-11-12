"""Draw ground truth, ignored regions and predictions on frames."""
from __future__ import annotations

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import CLASSES
from .data import SplitData

COLORS = plt.get_cmap("tab10").colors


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_frame(ax, data: SplitData, key: str, preds: pd.DataFrame | None = None,
               score_thr: float = 0.3, title: str | None = None, lw: float = 1.0) -> None:
    """GT boxes as solid class-coloured rectangles, ignore regions as grey hatched areas,
    predictions (optional) as dashed rectangles."""
    row = data.images.set_index("key").loc[key]
    ax.imshow(load_rgb(row.path))
    for r in data.ignore[data.ignore.key == key].itertuples():
        ax.add_patch(mpatches.Rectangle((r.x, r.y), r.w, r.h, fill=True, fc=(0.5, 0.5, 0.5, 0.35),
                                        ec="white", hatch="//", lw=0.8))
    for r in data.boxes[data.boxes.key == key].itertuples():
        ax.add_patch(mpatches.Rectangle((r.x, r.y), r.w, r.h, fill=False, ec=COLORS[r.cat], lw=lw))
    if preds is not None:
        for r in preds[(preds.key == key) & (preds.score >= score_thr)].itertuples():
            ax.add_patch(mpatches.Rectangle((r.x, r.y), r.w, r.h, fill=False, ec=COLORS[int(r.cat)],
                                            lw=lw, ls="--"))
    n = (data.boxes.key == key).sum()
    ax.set_title(title or f"{key}  ({int(row.width)}x{int(row.height)}, {n} objects)", fontsize=9)
    ax.axis("off")


def class_legend(fig) -> None:
    handles = [mpatches.Patch(color=COLORS[i], label=c) for i, c in enumerate(CLASSES)]
    handles.append(mpatches.Patch(fc=(0.5, 0.5, 0.5, 0.35), ec="white", hatch="//", label="ignored"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)


def show_frames(data: SplitData, keys, preds=None, cols: int = 2, size: float = 7.0, **kw):
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows * 0.6))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, k in zip(np.atleast_1d(axes).ravel(), keys):
        draw_frame(ax, data, k, preds, **kw)
    class_legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def show_crop(ax, data: SplitData, key: str, cx: float, cy: float, half: int = 100, preds=None, **kw):
    """Zoomed view around (cx, cy): useful for failure-case analysis of tiny objects."""
    draw_frame(ax, data, key, preds, **kw)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy + half, cy - half)
