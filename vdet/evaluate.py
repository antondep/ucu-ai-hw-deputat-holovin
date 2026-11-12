"""One COCO-style evaluator shared by every model, so all numbers are comparable.

Predictions are a DataFrame with columns
    key, cat (0-based), score, x, y, w, h   (pixels, top-left corner)
which is the same table the submission writer consumes.

Design decisions (explain these in the report):
* Ignored regions become iscrowd=1 ground truth. pycocotools only matches a crowd box
  with detections of the same category, but VisDrone ignore regions have no class,
  so each region is duplicated once per class. Detections falling inside them are
  then neither true nor false positives.
* maxDets = (1, 100, 500): frames are dense and the competition scores up to 500
  detections per image, so the headline numbers use the 500-detection limit.
* Size buckets follow COCO: small < 32^2, medium 32^2..96^2, large > 96^2 px^2.
  An extra "tiny" bucket (< 16^2) is reported because many VisDrone objects are tiny.
"""
from __future__ import annotations

import contextlib
import io
import time

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from . import CLASSES, NUM_CLASSES
from .data import SplitData

AREA_RANGES = {
    "all": [0, 1e10],
    "tiny": [0, 16 ** 2],
    "small": [0, 32 ** 2],
    "medium": [32 ** 2, 96 ** 2],
    "large": [96 ** 2, 1e10],
}


class DetectionEvaluator:
    def __init__(self, data: SplitData, max_dets=(1, 100, 500)):
        self.data = data
        self.max_dets = list(max_dets)
        self.key2id = {k: i + 1 for i, k in enumerate(data.images.key)}
        self.coco_gt = self._build_gt()

    # ------------------------------------------------------------------ ground truth
    def _build_gt(self) -> COCO:
        images = [dict(id=self.key2id[r.key], width=int(r.width), height=int(r.height), file_name=r.key)
                  for r in self.data.images.itertuples()]
        anns, aid = [], 1
        for r in self.data.boxes.itertuples():
            anns.append(dict(id=aid, image_id=self.key2id[r.key], category_id=int(r.cat) + 1,
                             bbox=[r.x, r.y, r.w, r.h], area=float(r.w * r.h), iscrowd=0))
            aid += 1
        for r in self.data.ignore.itertuples():  # class-agnostic "don't care" regions
            for c in range(NUM_CLASSES):
                anns.append(dict(id=aid, image_id=self.key2id[r.key], category_id=c + 1,
                                 bbox=[r.x, r.y, r.w, r.h], area=float(r.w * r.h), iscrowd=1))
                aid += 1
        gt = COCO()
        gt.dataset = dict(images=images, annotations=anns,
                          categories=[dict(id=i + 1, name=n) for i, n in enumerate(CLASSES)])
        with contextlib.redirect_stdout(io.StringIO()):
            gt.createIndex()
        return gt

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, preds: pd.DataFrame, per_class: bool = True) -> dict:
        """Return a flat dict of metrics: mAP, AP50, AP75, AP_tiny/small/medium/large, AR500, per-class AP."""
        t0 = time.time()
        p = preds[preds.key.isin(self.key2id)]
        if len(p) == 0:   # no detections at all: every metric is 0, same keys as a normal result
            out = {k: 0.0 for k in ("mAP", "AP50", "AP75", "AP_tiny", "AP_small", "AP_medium", "AP_large",
                                    f"AR{self.max_dets[-1]}")}
            if per_class:
                out.update({f"AP_{name}": 0.0 for name in CLASSES})
            return {**out, "n_dets": 0, "eval_sec": 0.0}
        # keep the top max_dets[-1] per image, as the leaderboard does
        p = p.sort_values("score", ascending=False).groupby("key").head(self.max_dets[-1])
        dets = [dict(image_id=self.key2id[r.key], category_id=int(r.cat) + 1,
                     bbox=[float(r.x), float(r.y), float(r.w), float(r.h)], score=float(r.score))
                for r in p.itertuples()]
        with contextlib.redirect_stdout(io.StringIO()):
            dt = self.coco_gt.loadRes(dets)
            ev = COCOeval(self.coco_gt, dt, iouType="bbox")
            ev.params.maxDets = self.max_dets
            ev.params.areaRng = list(AREA_RANGES.values())
            ev.params.areaRngLbl = list(AREA_RANGES.keys())
            ev.evaluate()
            ev.accumulate()

        prec = ev.eval["precision"]  # [T iou, R recall, K class, A area, M maxdet]
        rec = ev.eval["recall"]      # [T, K, A, M]
        iou = ev.params.iouThrs

        def ap(t=None, k=None, a="all"):
            ai = list(AREA_RANGES).index(a)
            x = prec[:, :, :, ai, -1]
            if t is not None:
                x = x[[int(np.argmin(np.abs(iou - t)))]]
            if k is not None:
                x = x[:, :, [k]]
            x = x[x > -1]
            return float(x.mean()) if x.size else float("nan")

        r = rec[:, :, 0, -1]
        out = {
            "mAP": ap(), "AP50": ap(0.5), "AP75": ap(0.75),
            "AP_tiny": ap(a="tiny"), "AP_small": ap(a="small"),
            "AP_medium": ap(a="medium"), "AP_large": ap(a="large"),
            f"AR{self.max_dets[-1]}": float(r[r > -1].mean()) if (r > -1).any() else float("nan"),
            "n_dets": len(p), "eval_sec": round(time.time() - t0, 1),
        }
        if per_class:
            for k, name in enumerate(CLASSES):
                out[f"AP_{name}"] = ap(k=k)
        self.last_eval = ev
        return out


def results_table(results: dict[str, dict], cols=None) -> pd.DataFrame:
    """Turn {run_name: metrics} into a tidy comparison table (percent, 1 decimal)."""
    cols = cols or ["mAP", "AP50", "AP75", "AP_tiny", "AP_small", "AP_medium", "AP_large"]
    df = pd.DataFrame(results).T
    keep = [c for c in cols if c in df] + [c for c in df if c not in cols and not c.startswith("AP_") and c not in ("n_dets", "eval_sec")]
    out = df[keep].copy()
    for c in out:
        if c in cols:
            out[c] = (out[c].astype(float) * 100).round(1)
    return out


def gt_as_predictions(data: SplitData, jitter_px: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Ground truth turned into predictions (score 1). Sanity check: jitter 0 must give mAP = 1.0."""
    rng = np.random.default_rng(seed)
    p = data.boxes[["key", "cat", "x", "y", "w", "h"]].copy()
    if jitter_px:
        p[["x", "y"]] += rng.normal(0, jitter_px, size=(len(p), 2))
    p["score"] = 1.0
    return p
