"""Thin helpers around Ultralytics (YOLO and RT-DETR share the same API).

Training metrics printed by Ultralytics are only a monitor: they use its own matcher, do not know
about ignored regions and are computed on the downscaled export. Every number we report comes from
`predict_table` -> `vdet.evaluate.DetectionEvaluator` on the original validation frames.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def predict_table(model, images: pd.DataFrame, imgsz: int = 1280, conf: float = 0.001,
                  iou: float = 0.6, max_det: int = 500, batch: int = 8, half: bool = True,
                  augment: bool = False, desc: str = "predict") -> tuple[pd.DataFrame, float]:
    """Run a Ultralytics model on `images` (needs columns key, path).

    Returns (predictions table, seconds per image). Boxes come back in original-image pixels.
    A low `conf` is used on purpose: mAP integrates over the whole precision/recall curve,
    so cutting low-score detections can only lower it.
    """
    rows, paths, keys = [], images["path"].tolist(), images["key"].tolist()
    t0 = time.time()
    for i in tqdm(range(0, len(paths), batch), desc=desc):
        res = model.predict(paths[i:i + batch], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                            half=half, augment=augment, verbose=False)
        for k, r in zip(keys[i:i + batch], res):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            rows.append(pd.DataFrame({
                "key": k, "cat": b.cls.cpu().numpy().astype(int), "score": b.conf.cpu().numpy(),
                "x": xyxy[:, 0], "y": xyxy[:, 1], "w": xyxy[:, 2] - xyxy[:, 0], "h": xyxy[:, 3] - xyxy[:, 1],
            }))
    sec = (time.time() - t0) / max(len(paths), 1)
    cols = ["key", "cat", "score", "x", "y", "w", "h"]
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)), sec


def _tile_boxes(W: int, H: int, n: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """n x n overlapping tiles covering the frame: (x0, y0, x1, y1)."""
    tw, th = W / (n - (n - 1) * overlap), H / (n - (n - 1) * overlap)
    out = []
    for j in range(n):
        for i in range(n):
            x0, y0 = round(i * tw * (1 - overlap)), round(j * th * (1 - overlap))
            out.append((x0, y0, min(W, round(x0 + tw)), min(H, round(y0 + th))))
    return out


def predict_views(model, images: pd.DataFrame, views=("full", "flip", "tiles"), imgsz: int = 1280,
                  grid: int = 2, overlap: float = 0.2, conf: float = 0.001, iou: float = 0.6,
                  max_det: int = 500, half: bool = True, desc: str = "views") -> tuple[dict, float]:
    """Test-time views of each frame, one table per view, all in original-frame pixels.

    full  : the whole frame (same as predict_table)
    flip  : the horizontally mirrored frame, boxes mirrored back (test-time augmentation)
    tiles : grid x grid overlapping crops, each resized to imgsz by the model, boxes shifted back;
            small objects are seen ~grid times larger (the slicing idea of SAHI, arXiv:2202.06934)
    The views of one frame go through the model as one batch. Merge them with vdet.ensemble.wbf.
    Returns ({view: table}, seconds per frame for all views together).
    """
    import cv2
    rows = {v: [] for v in views}
    t0 = time.time()
    for r in tqdm(images[["key", "path"]].itertuples(index=False), total=len(images), desc=desc):
        img = cv2.imread(r.path)
        H, W = img.shape[:2]
        crops, meta = [], []
        if "full" in views:
            crops.append(img); meta.append(("full", 0, 0, False))
        if "flip" in views:
            crops.append(np.ascontiguousarray(img[:, ::-1])); meta.append(("flip", 0, 0, True))
        if "tiles" in views:
            for x0, y0, x1, y1 in _tile_boxes(W, H, grid, overlap):
                crops.append(np.ascontiguousarray(img[y0:y1, x0:x1])); meta.append(("tiles", x0, y0, False))
        res = model.predict(crops, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, half=half, verbose=False)
        for (view, dx, dy, flipped), out in zip(meta, res):
            b = out.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy().astype(np.float64)
            if flipped:
                xyxy[:, [0, 2]] = W - xyxy[:, [2, 0]]
            xyxy[:, [0, 2]] += dx; xyxy[:, [1, 3]] += dy
            rows[view].append(pd.DataFrame({
                "key": r.key, "cat": b.cls.cpu().numpy().astype(int), "score": b.conf.cpu().numpy(),
                "x": xyxy[:, 0], "y": xyxy[:, 1], "w": xyxy[:, 2] - xyxy[:, 0], "h": xyxy[:, 3] - xyxy[:, 1]}))
    sec = (time.time() - t0) / max(len(images), 1)
    cols = ["key", "cat", "score", "x", "y", "w", "h"]
    return {v: (pd.concat(rows[v], ignore_index=True) if rows[v] else pd.DataFrame(columns=cols)) for v in views}, sec


def training_curve(run_dir: str | Path) -> pd.DataFrame:
    """Per-epoch losses and Ultralytics' own val mAP from results.csv (for plots)."""
    df = pd.read_csv(Path(run_dir) / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    return df


_RTDETR_OVERRIDES: dict = {}


def set_rtdetr_decoder(**overrides) -> None:
    """Change RT-DETR decoder settings that the model yaml cannot express, e.g.
    set_rtdetr_decoder(nq=500) (number of object queries) or set_rtdetr_decoder(nd=0) (no denoising).

    Ultralytics rebuilds the model from its yaml when training starts and when it resumes, and the
    yaml only passes `nc`, so the overrides are injected into RTDETRDecoder.__init__. COCO weights still
    load fully: neither setting changes the shape of any parameter. Call it before every train/resume;
    set_rtdetr_decoder() with no arguments restores the defaults (300 queries, 100 denoising groups).
    """
    from ultralytics.nn.modules.head import RTDETRDecoder
    if not hasattr(RTDETRDecoder, "_vdet_orig_init"):
        RTDETRDecoder._vdet_orig_init = RTDETRDecoder.__init__

        def _init(self, *args, **kwargs):
            RTDETRDecoder._vdet_orig_init(self, *args, **{**_RTDETR_OVERRIDES, **kwargs})
        RTDETRDecoder.__init__ = _init
    _RTDETR_OVERRIDES.clear()
    _RTDETR_OVERRIDES.update(overrides)


class ResultsLog:
    """Append-only JSON log of every experiment: config + metrics, survives across sessions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def add(self, name: str, config: dict, metrics: dict) -> None:
        # Re-read before writing: another session (e.g. a second Colab account training a different
        # experiment into the same shared folder) may have added entries since this one started.
        if self.path.exists():
            self.data = {**self.data, **json.loads(self.path.read_text())}
        clean = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in metrics.items()}
        self.data[name] = {"config": config, "metrics": clean}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str))
        os.replace(tmp, self.path)

    def table(self) -> pd.DataFrame:
        if self.path.exists():
            self.data = {**self.data, **json.loads(self.path.read_text())}
        return pd.DataFrame({k: v["metrics"] for k, v in self.data.items()}).T
