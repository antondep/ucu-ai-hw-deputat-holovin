"""Training and inference for torchvision detectors: Faster R-CNN and RetinaNet.

Data: the same exported dataset as the YOLO notebook (vdet.yolo_export): every 5th training frame,
ignored regions masked, frames pre-resized to a 1280 px longer side, YOLO txt labels. Using the
identical files keeps the training data exactly the same across detector families.

Predictions are returned as the shared table (key, cat, score, x, y, w, h) in ORIGINAL image pixels,
so they go straight into vdet.evaluate and vdet.submission.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import roi_heads as tv_roi_heads
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import box_iou, generalized_box_iou_loss
from tqdm.auto import tqdm

from . import NUM_CLASSES


# ============================================================================ data
def _read_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255)


class ExportedDetDataset(Dataset):
    """Training frames from the exported YOLO-format folder.

    Targets follow torchvision: boxes in xyxy pixels of the loaded image, labels 1..8
    (0 is background). Augmentation: horizontal flip only, so that differences between
    detectors come from the models, not from different augmentation pipelines.
    """

    def __init__(self, export_dir: str | Path, subset: str = "train", hflip: float = 0.5):
        self.img_dir = Path(export_dir) / "images" / subset
        self.lab_dir = Path(export_dir) / "labels" / subset
        self.keys = sorted(p.stem for p in self.lab_dir.glob("*.txt"))
        self.hflip = hflip

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        img = _read_rgb(self.img_dir / f"{k}.jpg")
        H, W = img.shape[:2]
        txt = (self.lab_dir / f"{k}.txt").read_text().split()
        a = np.array(txt, dtype=np.float32).reshape(-1, 5)
        cls = a[:, 0].astype(np.int64)
        cx, cy, w, h = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        boxes = np.clip(boxes, 0, [W, H, W, H])
        keep = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)  # torchvision rejects empty boxes
        boxes, cls = boxes[keep], cls[keep]
        if self.hflip and np.random.rand() < self.hflip:
            img = img[:, ::-1]
            boxes = np.stack([W - boxes[:, 2], boxes[:, 1], W - boxes[:, 0], boxes[:, 3]], 1)
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.as_tensor(cls + 1, dtype=torch.int64)}
        return _to_tensor(img), target


class InferenceDataset(Dataset):
    """Frames for prediction. Reads the downscaled export when available (fast), otherwise the
    original file; returns the factors that map boxes back to original pixels."""

    def __init__(self, images: pd.DataFrame, export_img_dir: str | Path | None = None):
        self.rows = images[["key", "path", "width", "height"]].to_dict("records")
        self.export_img_dir = Path(export_img_dir) if export_img_dir else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p = self.export_img_dir / f"{r['key']}.jpg" if self.export_img_dir else None
        img = _read_rgb(p if p is not None and p.exists() else r["path"])
        h, w = img.shape[:2]
        return _to_tensor(img), (r["key"], r["width"] / w, r["height"] / h)


def collate(batch):
    return tuple(zip(*batch))


# ============================================================================ GIoU box loss for the RoI head
_ORIG_FASTRCNN_LOSS = tv_roi_heads.fastrcnn_loss
_BOX_LOSS = {"name": "smooth_l1", "weight": 1.0}
_BBOX_CLIP = math.log(1000.0 / 16)


def _decode_unit(d: torch.Tensor, weights=(10.0, 10.0, 5.0, 5.0)) -> torch.Tensor:
    """Decode Faster R-CNN deltas against a unit reference box [0, 0, 1, 1].

    Deltas are normalised by the proposal's width and height, so decoding prediction and target
    against the same unit box gives boxes that differ from the real ones only by a per-axis scale
    and shift. IoU and GIoU are invariant to that, so the loss is exact without the proposals.
    """
    dx, dy = d[:, 0] / weights[0], d[:, 1] / weights[1]
    dw = (d[:, 2] / weights[2]).clamp(max=_BBOX_CLIP)
    dh = (d[:, 3] / weights[3]).clamp(max=_BBOX_CLIP)
    cx, cy, w, h = 0.5 + dx, 0.5 + dy, dw.exp(), dh.exp()
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def _fastrcnn_loss_switchable(class_logits, box_regression, labels, regression_targets):
    if _BOX_LOSS["name"] == "smooth_l1":
        return _ORIG_FASTRCNN_LOSS(class_logits, box_regression, labels, regression_targets)
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    cls_loss = F.cross_entropy(class_logits, labels)
    pos = torch.where(labels > 0)[0]
    N = class_logits.shape[0]
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    pred = _decode_unit(box_regression[pos, labels[pos]].float())
    tgt = _decode_unit(regression_targets[pos].float())
    box_loss = generalized_box_iou_loss(pred, tgt, reduction="sum") / labels.numel()
    return cls_loss, _BOX_LOSS["weight"] * box_loss


tv_roi_heads.fastrcnn_loss = _fastrcnn_loss_switchable  # RoIHeads looks the function up in this module


def set_box_loss(name: str, weight: float = 1.0) -> None:
    """'smooth_l1' (torchvision default) or 'giou'. The GIoU weight of 10 follows MMDetection's configs."""
    assert name in ("smooth_l1", "giou")
    _BOX_LOSS.update(name=name, weight=weight)


# ============================================================================ models
@dataclass
class FRCNNConfig:
    anchor_sizes: tuple = (32, 64, 128, 256, 512)       # one size per FPN level P2..P6 (torchvision default)
    aspect_ratios: tuple = (0.5, 1.0, 2.0)
    min_size: int = 720                                   # 1280x720 for 16:9 frames, as in the YOLO runs
    max_size: int = 1280
    rpn_pre_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_test: int = 1000
    box_batch_size_per_image: int = 512                   # RoIs sampled per image for the second stage
    box_positive_fraction: float = 0.25
    box_detections_per_img: int = 100
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_loss: str = "smooth_l1"
    box_loss_weight: float = 1.0
    trainable_backbone_layers: int = 3


def build_faster_rcnn(cfg: FRCNNConfig, pretrained: bool = True):
    """Faster R-CNN R50-FPN with COCO weights; only the class predictor is new (8 classes + bg).
    Changing anchor sizes keeps 3 anchors per location, so the pretrained RPN head still fits."""
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    anchors = AnchorGenerator(tuple((s,) for s in cfg.anchor_sizes), (tuple(cfg.aspect_ratios),) * len(cfg.anchor_sizes))
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None,
        trainable_backbone_layers=cfg.trainable_backbone_layers if pretrained else None,
        rpn_anchor_generator=anchors, min_size=cfg.min_size, max_size=cfg.max_size,
        rpn_pre_nms_top_n_train=cfg.rpn_pre_nms_top_n_train, rpn_post_nms_top_n_train=cfg.rpn_post_nms_top_n_train,
        rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_top_n_test, rpn_post_nms_top_n_test=cfg.rpn_post_nms_top_n_test,
        box_batch_size_per_image=cfg.box_batch_size_per_image, box_positive_fraction=cfg.box_positive_fraction,
        box_detections_per_img=cfg.box_detections_per_img, box_score_thresh=cfg.box_score_thresh,
        box_nms_thresh=cfg.box_nms_thresh)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)
    return model


def set_inference_params(model, detections_per_img=None, score_thresh=None, nms_thresh=None,
                         rpn_pre_nms_top_n=None, rpn_post_nms_top_n=None, topk_candidates=None) -> None:
    """Change test-time settings of an already trained Faster R-CNN or RetinaNet without retraining."""
    if hasattr(model, "roi_heads"):              # Faster R-CNN: the settings live in the RoI head and the RPN
        rh, rpn = model.roi_heads, model.rpn
        if rpn_pre_nms_top_n is not None: rpn._pre_nms_top_n["testing"] = rpn_pre_nms_top_n
        if rpn_post_nms_top_n is not None: rpn._post_nms_top_n["testing"] = rpn_post_nms_top_n
    else:                                        # RetinaNet: on the model itself
        rh = model
        if topk_candidates is not None: model.topk_candidates = topk_candidates
    if detections_per_img is not None: rh.detections_per_img = detections_per_img
    if score_thresh is not None: rh.score_thresh = score_thresh
    if nms_thresh is not None: rh.nms_thresh = nms_thresh


# ============================================================================ RetinaNet
from torchvision.models.detection import retinanet as tv_retina

_ORIG_FOCAL = tv_retina.sigmoid_focal_loss
_FOCAL = {"alpha": 0.25, "gamma": 2.0}


def _focal_switchable(inputs, targets, alpha: float = 0.25, gamma: float = 2, reduction: str = "none"):
    return _ORIG_FOCAL(inputs, targets, alpha=_FOCAL["alpha"], gamma=_FOCAL["gamma"], reduction=reduction)


tv_retina.sigmoid_focal_loss = _focal_switchable   # the classification head looks it up in this module


def set_focal(alpha: float = 0.25, gamma: float = 2.0) -> None:
    """Focal loss parameters for RetinaNet (torchvision hard-codes alpha=0.25, gamma=2)."""
    _FOCAL.update(alpha=alpha, gamma=gamma)


@dataclass
class RetinaConfig:
    anchor_sizes: tuple = (32, 64, 128, 256, 512)   # base size per FPN level P3..P7 (torchvision default)
    octave_scales: tuple = (1.0, 2 ** (1 / 3), 2 ** (2 / 3))
    aspect_ratios: tuple = (0.5, 1.0, 2.0)
    min_size: int = 720
    max_size: int = 1280
    fg_iou_thresh: float = 0.5                      # anchor is positive if IoU >= this
    bg_iou_thresh: float = 0.4                      # negative if IoU < this; in between = ignored
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    box_loss: str = "l1"                            # torchvision supports l1, smooth_l1, giou, ciou, diou
    detections_per_img: int = 300
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    topk_candidates: int = 1000
    trainable_backbone_layers: int = 3


def build_retinanet(cfg: RetinaConfig, pretrained: bool = True):
    """RetinaNet R50-FPN with COCO weights. Only the last classification conv is replaced (8 classes + an
    unused background index, so labels 1..8 match Faster R-CNN). Changing the anchor sizes keeps 9 anchors
    per location, so the pretrained box-regression head still fits."""
    from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn

    sizes = tuple(tuple(b * s for s in cfg.octave_scales) for b in cfg.anchor_sizes)
    anchors = AnchorGenerator(sizes, (tuple(cfg.aspect_ratios),) * len(sizes))
    model = retinanet_resnet50_fpn(
        weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1 if pretrained else None, weights_backbone=None,
        trainable_backbone_layers=cfg.trainable_backbone_layers if pretrained else None, anchor_generator=anchors,
        min_size=cfg.min_size, max_size=cfg.max_size, fg_iou_thresh=cfg.fg_iou_thresh, bg_iou_thresh=cfg.bg_iou_thresh,
        score_thresh=cfg.score_thresh, nms_thresh=cfg.nms_thresh, detections_per_img=cfg.detections_per_img,
        topk_candidates=cfg.topk_candidates)
    head = model.head.classification_head
    n_cls = NUM_CLASSES + 1
    conv = torch.nn.Conv2d(head.cls_logits.in_channels, head.num_anchors * n_cls, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(conv.weight, std=0.01)
    torch.nn.init.constant_(conv.bias, -math.log((1 - 0.01) / 0.01))   # prior 0.01, as in the RetinaNet paper
    head.cls_logits, head.num_classes = conv, n_cls
    model.head.regression_head._loss_type = cfg.box_loss
    return model


@torch.no_grad()
def anchor_assignment_stats(model, dataset: Dataset, device, n_images: int = 100, seed: int = 0) -> pd.DataFrame:
    """How many anchors per image are positive / ignored / negative under the model's IoU thresholds.

    This is the foreground-background imbalance a one-stage detector faces: every anchor is classified,
    while a two-stage detector only classifies a few hundred sampled proposals (RPN: 256, RoI head: 512 per image).
    Also: share of ground-truth objects that get no positive anchor at all.
    """
    from torchvision.ops import box_iou as _iou
    model.eval().to(device)
    rng = np.random.default_rng(seed)
    rows = []
    for i in rng.choice(len(dataset), min(n_images, len(dataset)), replace=False):
        img, tgt = dataset[int(i)]
        il, tl = model.transform([img.to(device)], [{k: v.to(device) for k, v in tgt.items()}])
        feats = list(model.backbone(il.tensors).values())
        anchors = model.anchor_generator(il, feats)[0]
        gt = tl[0]["boxes"]
        if len(gt) == 0:
            continue
        iou = _iou(gt, anchors)
        matched = model.proposal_matcher(iou)          # >=0 positive, -1 negative, -2 ignored
        pos = matched >= 0
        covered = torch.zeros(len(gt), dtype=torch.bool, device=device)
        covered[matched[pos]] = True
        side = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).sqrt()
        rows.append({"anchors": len(anchors), "positive": int(pos.sum()), "ignored": int((matched == -2).sum()),
                     "negative": int((matched == -1).sum()), "objects": len(gt),
                     "objects_without_positive_%": float((~covered).float().mean() * 100),
                     "small_objects_without_positive_%": float((~covered[side < 32]).float().mean() * 100) if (side < 32).any() else float("nan")})
    df = pd.DataFrame(rows)
    out = df.mean().to_frame("mean per image").T
    out["negative : positive"] = (df.negative.sum() / max(df.positive.sum(), 1))
    return out.round(1)


# ============================================================================ training
@dataclass
class TrainConfig:
    epochs: int = 8
    batch: int = 4
    lr: float = 0.005            # linear scaling rule: 0.02 for batch 16 -> 0.005 for batch 4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_iters: int = 500
    milestones: tuple = (6,)     # epochs after which lr is multiplied by 0.1
    amp: bool = True
    workers: int = 2
    seed: int = 42
    max_iters_per_epoch: int | None = None   # only for smoke tests


def _lr_lambda(tc: TrainConfig, iters_per_epoch: int):
    def f(it):
        warm = min(1.0, (it + 1) / tc.warmup_iters) if tc.warmup_iters else 1.0
        warm = 0.001 + (1 - 0.001) * warm
        decay = 0.1 ** sum(it >= m * iters_per_epoch for m in tc.milestones)
        return warm * decay
    return f


def train(model, dataset: Dataset, tc: TrainConfig, run_dir: str | Path, device,
          monitor=None, model_cfg=None) -> pd.DataFrame:
    """Train with SGD + warmup + step decay, AMP, and a checkpoint after every epoch.

    If run_dir/last.pth exists, training resumes from it (Colab/Kaggle sessions end).
    `monitor(model) -> dict` is called after each epoch (validation mAP over training).
    Returns the per-epoch history, also written to run_dir/history.csv.
    """
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model_cfg, FRCNNConfig):
        set_box_loss(model_cfg.box_loss, model_cfg.box_loss_weight)
    elif isinstance(model_cfg, RetinaConfig):
        set_focal(model_cfg.focal_alpha, model_cfg.focal_gamma)
    torch.manual_seed(tc.seed); np.random.seed(tc.seed)
    g = torch.Generator(); g.manual_seed(tc.seed)
    loader = DataLoader(dataset, batch_size=tc.batch, shuffle=True, num_workers=tc.workers,
                        collate_fn=collate, generator=g, drop_last=True, persistent_workers=tc.workers > 0,
                        worker_init_fn=lambda w: np.random.seed(torch.initial_seed() % 2 ** 32))  # flips differ per worker/epoch
    ipe = min(len(loader), tc.max_iters_per_epoch or len(loader))
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=tc.lr, momentum=tc.momentum, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(tc, ipe))
    use_amp = tc.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, start = [], 0

    ckpt = run_dir / "last.pth"
    if ckpt.exists():
        s = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); scaler.load_state_dict(s["scaler"])
        history, start = s["history"], s["epoch"] + 1
        print(f"resumed from epoch {start} / {tc.epochs}")

    for epoch in range(start, tc.epochs):
        model.train()
        t0, sums, n, skipped = time.time(), {}, 0, 0
        pbar = tqdm(loader, total=ipe, desc=f"epoch {epoch + 1}/{tc.epochs}", leave=False)
        for it, (imgs, tgts) in enumerate(pbar):
            if it >= ipe:
                break
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                losses = model(imgs, tgts)
            loss = sum(losses.values())
            if not torch.isfinite(loss):          # rare AMP overflow: skip the step instead of crashing
                skipped += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n += 1
            if it % 50 == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}")
        row = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], "train_min": round((time.time() - t0) / 60, 1),
               "skipped_steps": skipped, **{k: v / max(n, 1) for k, v in sums.items()}}
        row["loss"] = sum(v for k, v in row.items() if k.startswith("loss_"))
        if monitor is not None:
            row.update(monitor(model))
            model.train()
        history.append(row)
        tmp = run_dir / "last.pth.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "history": history,
                    "model_cfg": asdict(model_cfg) if model_cfg is not None else None, "train_cfg": asdict(tc)}, tmp)
        os.replace(tmp, ckpt)                     # atomic: a disconnect never leaves a broken checkpoint
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        mon = f"  monitor mAP {row['mAP'] * 100:.1f}" if "mAP" in row else ""
        print(f"epoch {epoch + 1}/{tc.epochs}  loss {row['loss']:.3f}{mon}  ({row['train_min']} min)")

    torch.save(model.state_dict(), run_dir / "final.pth")
    return pd.DataFrame(history)


def is_finished(run_dir: str | Path, epochs: int) -> bool:
    p = Path(run_dir) / "history.csv"
    return (Path(run_dir) / "final.pth").exists() and p.exists() and len(pd.read_csv(p)) >= epochs


# ============================================================================ inference
@torch.no_grad()
def predict_table(model, images: pd.DataFrame, device, export_img_dir=None, batch: int = 4,
                  workers: int = 2, amp: bool = True, desc: str = "predict") -> tuple[pd.DataFrame, float]:
    """Run a torchvision detector; boxes come back in original-image pixels.
    Returns (table, seconds per image incl. loading)."""
    model.eval().to(device)
    dl = DataLoader(InferenceDataset(images, export_img_dir), batch_size=batch, num_workers=workers,
                    collate_fn=collate)
    use_amp = amp and device.type == "cuda"
    rows, t0 = [], time.time()
    for imgs, metas in tqdm(dl, desc=desc, leave=False):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outs = model([i.to(device, non_blocking=True) for i in imgs])
        for o, (key, sx, sy) in zip(outs, metas):
            b = o["boxes"].float().cpu().numpy()
            if len(b) == 0:
                continue
            b[:, [0, 2]] *= sx; b[:, [1, 3]] *= sy
            rows.append(pd.DataFrame({"key": key, "cat": o["labels"].cpu().numpy() - 1,
                                      "score": o["scores"].float().cpu().numpy(),
                                      "x": b[:, 0], "y": b[:, 1], "w": b[:, 2] - b[:, 0], "h": b[:, 3] - b[:, 1]}))
    sec = (time.time() - t0) / max(len(images), 1)
    cols = ["key", "cat", "score", "x", "y", "w", "h"]
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)), sec


@torch.no_grad()
def proposal_recall(model, data, device, n_list=(100, 300, 1000, 2000), iou_thr: float = 0.5,
                    max_images: int = 300, seed: int = 0) -> pd.DataFrame:
    """Class-agnostic recall of the RPN's top-N proposals, by object size.

    Answers "what does the proposal stage contribute": the second stage can only classify and
    refine objects that some proposal already covers, so this recall is its upper bound.
    """
    model.eval().to(device)
    old = model.rpn._post_nms_top_n["testing"], model.rpn._pre_nms_top_n["testing"]
    model.rpn._post_nms_top_n["testing"] = max(n_list)
    model.rpn._pre_nms_top_n["testing"] = max(max(n_list), old[1])
    imgs = data.images.sample(min(max_images, len(data.images)), random_state=seed)
    gtb = {k: g for k, g in data.boxes.groupby("key")}
    best = {n: [] for n in n_list}; sides = []
    for r in tqdm(imgs.itertuples(), total=len(imgs), desc="proposal recall", leave=False):
        g = gtb.get(r.key)
        if g is None:
            continue
        boxes = torch.tensor(np.stack([g.x, g.y, g.x + g.w, g.y + g.h], 1), dtype=torch.float32)
        img = _to_tensor(_read_rgb(r.path)).to(device)
        il, tl = model.transform([img], [{"boxes": boxes.to(device), "labels": torch.ones(len(boxes), dtype=torch.int64, device=device)}])
        props, _ = model.rpn(il, model.backbone(il.tensors))
        p, gt = props[0], tl[0]["boxes"]
        for n in n_list:
            best[n].append(box_iou(gt, p[:n]).max(1).values.cpu().numpy() if len(p) else np.zeros(len(gt)))
        sides.append(np.sqrt(g.w.to_numpy() * g.h.to_numpy()))
    model.rpn._post_nms_top_n["testing"], model.rpn._pre_nms_top_n["testing"] = old
    side = np.concatenate(sides)
    mean = lambda x: float(x.mean()) if x.size else float("nan")
    out = {}
    for n in n_list:
        hit = np.concatenate(best[n]) >= iou_thr
        out[n] = {"all": mean(hit), "<16px": mean(hit[side < 16]), "16-32px": mean(hit[(side >= 16) & (side < 32)]),
                  "32-96px": mean(hit[(side >= 32) & (side < 96)]), ">96px": mean(hit[side >= 96])}
    return pd.DataFrame(out).T.rename_axis("top-N proposals")
