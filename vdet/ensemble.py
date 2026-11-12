"""Combining predictions: Weighted Boxes Fusion (WBF) for model ensembles and test-time views.

WBF (Solovyev et al., 2019, https://arxiv.org/abs/1910.13302; package `ensemble-boxes`) clusters
same-class boxes from several sources by IoU and replaces each cluster by a score-weighted average
box. Unlike NMS, which keeps one box and discards the rest, WBF uses every box in the cluster, so
models that localise differently correct each other. With conf_type='avg' a box found by only some
sources gets its score scaled down by the share of sources that found it, which rewards agreement.

Everything works on the shared prediction table (key, cat, score, x, y, w, h) in original pixels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

COLS = ["key", "cat", "score", "x", "y", "w", "h"]


def _by_key(p: pd.DataFrame, top_k: int | None) -> dict:
    """{key: (xyxy, score, cat)} with at most top_k highest-scoring boxes per image."""
    p = p.sort_values("score", ascending=False)
    if top_k:
        p = p.groupby("key", sort=False).head(top_k)
    out = {}
    for k, g in p.groupby("key", sort=False):
        a = g[["x", "y", "w", "h"]].to_numpy(np.float64)
        out[k] = (np.stack([a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]], 1),
                  g["score"].to_numpy(np.float64), g["cat"].to_numpy(np.int64))
    return out


def wbf(preds_list: list[pd.DataFrame], sizes: pd.DataFrame, weights=None, iou_thr: float = 0.6,
        skip_box_thr: float = 0.001, conf_type: str = "avg", top_k: int | None = 300,
        max_dets: int = 500, desc: str = "wbf") -> pd.DataFrame:
    """Fuse several prediction tables image by image.

    sizes: DataFrame with key, width, height (WBF needs coordinates normalised to [0, 1]).
    top_k: boxes kept per source and image before fusing (bounds the run time; scores below the
    top 300 hardly change mAP). max_dets: boxes kept per image after fusing (leaderboard limit).
    """
    from ensemble_boxes import weighted_boxes_fusion

    groups = [_by_key(p, top_k) for p in preds_list]
    size = sizes.set_index("key")[["width", "height"]]
    keys = sorted(set().union(*[g.keys() for g in groups]))
    rows = []
    for k in tqdm(keys, desc=desc, leave=False):
        W, H = float(size.at[k, "width"]), float(size.at[k, "height"])
        norm = np.array([W, H, W, H])
        bl, sl, ll = [], [], []
        for g in groups:
            if k in g:
                b, s, c = g[k]
                bl.append(np.clip(b / norm, 0, 1)); sl.append(s); ll.append(c)
            else:   # this source found nothing in this image: it still counts in the averaging
                bl.append(np.zeros((0, 4))); sl.append(np.zeros(0)); ll.append(np.zeros(0))
        b, s, c = weighted_boxes_fusion(bl, sl, ll, weights=weights, iou_thr=iou_thr,
                                        skip_box_thr=skip_box_thr, conf_type=conf_type)
        if len(b) == 0:
            continue
        order = np.argsort(-s)[:max_dets]
        b, s, c = b[order] * norm, s[order], c[order]
        rows.append(pd.DataFrame({"key": k, "cat": c.astype(int), "score": s,
                                  "x": b[:, 0], "y": b[:, 1], "w": b[:, 2] - b[:, 0], "h": b[:, 3] - b[:, 1]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=COLS)
