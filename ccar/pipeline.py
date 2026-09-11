"""Faithful reimplementation of the ClearCLIP/mmseg inference & eval protocol.

Replicates (ClearCLIP repo @ ad68a40):
  - mmseg test pipeline: LoadImageFromFile (cv2, BGR) -> Resize(scale=(2048,448),
    keep_ratio=True, bilinear) -> GT loaded at ORIGINAL resolution -> SegDataPreProcessor
    (bgr_to_rgb, mean/std in 0-255 RGB domain, fp16 at model input)
  - clearclip_segmentor.py: forward_slide (crop 448 / stride 224, pad-to-/16 centered,
    count_mat averaging at resized resolution, final bilinear to ori_shape),
    postprocess_result (x logit_scale=50 -> softmax -> argmax; prob_thd=0.0 no-op;
    VOC20/Context59/ADE20K cls files contain no synonym commas -> no query merging)
  - mmseg IoUMetric: dataset-aggregated intersect/union, ignore_index=255,
    reduce_zero_label (0=bg -> 255 ignore, v -> v-1)

Dtype policy (deliberate, documented):
  - baseline path: fp16 end-to-end, exactly like the official segmentor.
  - head path: backbone in fp16 (official), per-head logits/accumulation in fp32
    (robustness of the 12-way decomposition; argmax-level equivalence holds).
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

CCAR_PKG = Path(__file__).resolve().parent
PROJECT_ROOT = CCAR_PKG.parent
DATA_ROOT = Path(
    os.environ.get("CCAR_DATA_ROOT", str(PROJECT_ROOT / "data"))
).expanduser().resolve()
OFFICIAL_CFG = PROJECT_ROOT / "repos/ClearCLIP/configs"

# Cityscapes labelIds -> trainIds. The official ClearCLIP/mmseg evaluator
# uses the 19 trainIds and ignores all other labelIds.
CITYSCAPES_LABEL_MAP = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}

MEAN = np.array([122.771, 116.746, 104.094], dtype=np.float32)   # RGB, 0-255 domain
STD = np.array([68.501, 66.632, 70.323], dtype=np.float32)       # RGB, 0-255 domain

DATASETS = {
    "voc20": dict(
        img_dir=DATA_ROOT / "VOCdevkit/VOC2012/JPEGImages",
        gt_dir=DATA_ROOT / "VOCdevkit/VOC2012/SegmentationClass",
        split=DATA_ROOT / "VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
        train_split=DATA_ROOT / "VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt",
        cls_file=CCAR_PKG / "cls/cls_voc20.txt",
        num_classes=20, reduce_zero_label=True, img_suffix=".jpg"),
    "voc21": dict(
        img_dir=DATA_ROOT / "VOCdevkit/VOC2012/JPEGImages",
        gt_dir=DATA_ROOT / "VOCdevkit/VOC2012/SegmentationClass",
        split=DATA_ROOT / "VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
        train_split=DATA_ROOT / "VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt",
        cls_file=OFFICIAL_CFG / "cls_voc21.txt",
        num_classes=21, reduce_zero_label=False, img_suffix=".jpg"),
    "context59": dict(
        img_dir=DATA_ROOT / "VOCdevkit/VOC2010/JPEGImages",
        gt_dir=DATA_ROOT / "VOCdevkit/VOC2010/SegmentationClassContext",
        split=DATA_ROOT / "VOCdevkit/VOC2010/ImageSets/SegmentationContext/val.txt",
        train_split=DATA_ROOT / "VOCdevkit/VOC2010/ImageSets/SegmentationContext/train.txt",
        cls_file=CCAR_PKG / "cls/cls_context59.txt",
        num_classes=59, reduce_zero_label=True, img_suffix=".jpg"),
    "context60": dict(
        img_dir=DATA_ROOT / "VOCdevkit/VOC2010/JPEGImages",
        gt_dir=DATA_ROOT / "VOCdevkit/VOC2010/SegmentationClassContext",
        split=DATA_ROOT / "VOCdevkit/VOC2010/ImageSets/SegmentationContext/val.txt",
        train_split=DATA_ROOT / "VOCdevkit/VOC2010/ImageSets/SegmentationContext/train.txt",
        cls_file=OFFICIAL_CFG / "cls_context60.txt",
        num_classes=60, reduce_zero_label=False, img_suffix=".jpg"),
    "ade20k": dict(
        img_dir=DATA_ROOT / "ADEChallengeData2016/images/validation",
        gt_dir=DATA_ROOT / "ADEChallengeData2016/annotations/validation",
        split=None,   # whole validation dir
        train_img_dir=DATA_ROOT / "ADEChallengeData2016/images/training",
        train_gt_dir=DATA_ROOT / "ADEChallengeData2016/annotations/training",
        cls_file=CCAR_PKG / "cls/cls_ade20k.txt",
        num_classes=150, reduce_zero_label=True, img_suffix=".jpg"),
    # v1.8 P2: stuffthingmaps PNGs carry RAW class ids with gaps (0-181, 255
    # unlabeled); label_map = official mmseg coco_stuff164k converter table
    # (raw->trainId, unmapped->ignore). reduce_zero_label=False (raw 0 = person).
    "cocostuff": dict(
        img_dir=DATA_ROOT / "coco/val2017",
        gt_dir=DATA_ROOT / "coco/stuffthingmaps/val2017",
        split=None,   # whole validation dir
        cls_file=CCAR_PKG / "cls/cls_coco_stuff.txt",
        num_classes=171, reduce_zero_label=False, img_suffix=".jpg",
        train_img_dir=DATA_ROOT / "coco/train2017",
        train_gt_dir=DATA_ROOT / "coco/stuffthingmaps/train2017",
        train_calibration_frac=0.1,
        label_map=json.load(open(CCAR_PKG / "cls/cocostuff_rawid_to_trainid.json"))),
    "cocoobject": dict(
        img_dir=DATA_ROOT / "coco/val2017",
        gt_dir=DATA_ROOT / "coco_object/annotations/val2017",
        split=None,
        cls_file=OFFICIAL_CFG / "cls_coco_object.txt",
        num_classes=81, reduce_zero_label=False, img_suffix=".jpg",
        gt_suffix="_instanceTrainIds.png",
        train_img_dir=DATA_ROOT / "coco/train2017",
        train_gt_dir=DATA_ROOT / "coco_object/annotations/train2017",
        train_calibration_frac=0.1),
    "cityscapes": dict(
        img_dir=DATA_ROOT / "cityscapes/leftImg8bit/val",
        gt_dir=DATA_ROOT / "cityscapes/gtFine/val",
        split=None,
        cls_file=OFFICIAL_CFG / "cls_city_scapes.txt",
        num_classes=19, reduce_zero_label=False,
        img_suffix="_leftImg8bit.png", gt_suffix="_gtFine_labelTrainIds.png",
        train_img_dir=DATA_ROOT / "cityscapes/leftImg8bit/train",
        train_gt_dir=DATA_ROOT / "cityscapes/gtFine/train",
        train_calibration_frac=0.1,
        recursive=True, slide_crop=224, slide_stride=224),
}


def dataset_items(name, limit=None, seed=42):
    cfg = DATASETS[name]
    if cfg["split"] is not None:
        ids = [l.strip() for l in open(cfg["split"]) if l.strip()]
        gt_suffix = cfg.get("gt_suffix", ".png")
        items = [(cfg["img_dir"] / (i + cfg["img_suffix"]),
                  cfg["gt_dir"] / (i + gt_suffix)) for i in ids]
    else:
        gt_suffix = cfg.get("gt_suffix", ".png")
        globber = cfg["img_dir"].rglob if cfg.get("recursive", False) else cfg["img_dir"].glob
        items = []
        for img_path in globber("*" + cfg["img_suffix"]):
            rel = img_path.relative_to(cfg["img_dir"])
            gt_name = rel.name[:-len(cfg["img_suffix"])] + gt_suffix
            items.append((img_path, cfg["gt_dir"] / rel.parent / gt_name))
        items.sort()
    if limit is not None and limit < len(items):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(items), size=limit, replace=False)
        items = [items[i] for i in sorted(idx)]
    return items


def rescale_size(h, w, scale=(2048, 448)):
    """mmcv rescale_size(factor=False)."""
    max_long_edge, max_short_edge = max(scale), min(scale)
    sf = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    return int(h * sf + 0.5), int(w * sf + 0.5)


def preprocess(img_path, short_side=448):
    """BGR -> keep-ratio resize -> RGB -> normalize.

    ``short_side=448`` preserves the audited ClearCLIP path. Protocol-specific
    runners may select another short side while retaining the same max-long
    edge, OpenCV bilinear interpolation, normalization, and return contract.
    Returns float32 (1,3,h,w) tensor and original (h,w).
    """
    if short_side <= 0:
        raise ValueError(f"short_side must be positive, got {short_side}")
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    assert img is not None, f"failed to read {img_path}"
    h, w = img.shape[:2]
    nh, nw = rescale_size(h, w, scale=(2048, short_side))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - MEAN) / STD
    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0)
    return t, (h, w)


def load_gt(gt_path, reduce_zero_label=True, label_map=None):
    """Palette-index GT at original resolution; mmseg reduce_zero_label semantics.
    label_map (optional): dict raw_id -> train_id applied via a 256-entry LUT;
    raw ids absent from the dict become 255 (ignore). Used for COCO-Stuff
    stuffthingmaps, whose PNGs carry RAW class ids with gaps (0-181 + 255):
    the mapping is the official mmseg coco_stuff164k converter table verbatim
    (scripts/ref/mmseg_coco_stuff164k_converter.py, vendored json in
    ccar/cls/cocostuff_rawid_to_trainid.json; PROVENANCE 2026-09-08)."""
    gt = np.array(Image.open(gt_path))
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    if reduce_zero_label:
        gt[gt == 0] = 255
        gt = gt - 1
        gt[gt == 254] = 255
    if label_map is not None:
        lut = np.full(256, 255, dtype=np.uint8)
        for k, v in label_map.items():
            lut[int(k)] = int(v)
        gt = lut[gt]
    return gt


def compute_padsize(H, W, patch_size):
    """Verbatim from clearclip_segmentor.py."""
    l, r, t, b = 0, 0, 0, 0
    if W % patch_size:
        lr = patch_size - (W % patch_size)
        l = lr // 2
        r = lr - l
    if H % patch_size:
        tb = patch_size - (H % patch_size)
        t = tb // 2
        b = tb - t
    return l, r, t, b


def slide_crops(h_img, w_img, crop=448, stride=224):
    """(y1,x1,y2,x2) exactly as clearclip_segmentor.forward_slide iterates."""
    h_grids = max(h_img - crop + stride - 1, 0) // stride + 1
    w_grids = max(w_img - crop + stride - 1, 0) // stride + 1
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * stride
            x1 = w_idx * stride
            y2 = min(y1 + crop, h_img)
            x2 = min(x1 + crop, w_img)
            y1 = max(y2 - crop, 0)
            x1 = max(x2 - crop, 0)
            yield y1, x1, y2, x2


def _place(field_qhw, y1, x1, y2, x2, h_img, w_img):
    """Place (Q,h,w) crop field into full-res canvas coords (padding semantics of
    official lines 181-183); returns a full-canvas (Q,h_img,w_img) tensor."""
    return nn.functional.pad(field_qhw.unsqueeze(0),
                             (int(x1), int(w_img - x2), int(y1), int(h_img - y2)))[0]


@torch.no_grad()
def slide_inference(model, text_features, img_t, mode, num_heads=12,
                    crop=448, stride=224, patch=16, device="cuda",
                    model_type="ClearCLIP"):
    """Official sliding-window replication.

    mode='baseline': official encode_image path, fp16 end-to-end.
                     -> dict(logits=(Q,h,w) fp16 @ resized res after count_mat avg)
    mode='heads':    per-head decomposition (backbone fp16, head math fp32).
        head_logits (H,Q,h,w) fp32: exact official geometry (per-crop bilinear to
            crop res, per-pixel count_mat averaging) per head;
        head_grid (H,Q,gh,gw) fp16: common-patch-grid approximation with per-cell
            counts (for the oracle cache; equivalence validated on VOC20).
    """
    _, _, h_img, w_img = img_t.shape
    Q = text_features.shape[0]
    heads_mode = (mode == "heads")
    model_dtype = model.visual.conv1.weight.dtype   # authoritative input dtype
    from .clip_model import forward_head_decomposed

    if heads_mode:
        head_preds = torch.zeros((num_heads, Q, h_img, w_img), dtype=torch.float32, device=device)
    else:
        preds = torch.zeros((1, Q, h_img, w_img), dtype=torch.float32, device=device)
    count_mat = torch.zeros((1, 1, h_img, w_img), dtype=torch.float32, device=device)

    gh, gw = (h_img + patch - 1) // patch, (w_img + patch - 1) // patch
    if heads_mode:
        head_grid_acc = torch.zeros((num_heads, Q, gh, gw), dtype=torch.float32, device=device)
        grid_count = torch.zeros((gh, gw), dtype=torch.float32, device=device)

    for (y1, x1, y2, x2) in slide_crops(h_img, w_img, crop, stride):
        crop_img = img_t[:, :, y1:y2, x1:x2]
        Hr, Wr = crop_img.shape[2:]
        pad = compute_padsize(Hr, Wr, patch)
        if any(pad):
            crop_img = nn.functional.pad(crop_img, pad)
        crop_half = crop_img.to(model_dtype).to(device)

        if heads_mode:
            out = forward_head_decomposed(
                model, crop_half, return_official=False,
                model_type=model_type,
            )
            feats = out["ln_feats"]          # (H, N, D) fp32, exact LN-affine per-head features
            Hn = feats.shape[0]
            hl = torch.einsum("hnd,qd->hqn", feats.float(), text_features.float())
            cg, cgd = crop_img.shape[2] // patch, crop_img.shape[3] // patch
            hl = hl.reshape(Hn, Q, cg, cgd)
            up = F.interpolate(hl.reshape(Hn * Q, cg, cgd).unsqueeze(0),
                               size=crop_img.shape[-2:], mode="bilinear")[0]
            up = up.view(Hn, Q, crop_img.shape[2], crop_img.shape[3])
            l, r, t, b = pad
            up_real = up[:, :, t:t + Hr, l:l + Wr]
            for h in range(Hn):
                head_preds[h] += _place(up_real[h], y1, x1, y2, x2, h_img, w_img)
            # common-grid accumulation: cells whose center lies in the real region
            ci = torch.arange(cg, device=device, dtype=torch.float32)
            cj = torch.arange(cgd, device=device, dtype=torch.float32)
            mask_i = ((ci + 0.5) * patch - t) < Hr
            mask_j = ((cj + 0.5) * patch - l) < Wr
            oy, ox = int(round(y1 / patch)), int(round(x1 / patch))
            ti = torch.clamp(oy + torch.nonzero(mask_i).squeeze(-1), max=gh - 1)
            tj = torch.clamp(ox + torch.nonzero(mask_j).squeeze(-1), max=gw - 1)
            src = hl[:, :, mask_i][:, :, :, mask_j]                   # (H,Q,gi,gj)
            head_grid_acc[:, :, ti.unsqueeze(1), tj.unsqueeze(0)] += src
            grid_count[ti.unsqueeze(1), tj.unsqueeze(0)] += 1
        elif mode == "recompose":
            out = forward_head_decomposed(
                model, crop_half, return_official=False,
                model_type=model_type,
            )
            feats = out["feats_recomposed"]                          # (1, N, D) official path
            logits = feats[0] @ text_features.T                      # (N, Q)
            cg, cgd = crop_img.shape[2] // patch, crop_img.shape[3] // patch
            logits = logits.permute(1, 0).reshape(1, Q, cg, cgd)
            up = F.interpolate(logits, size=crop_img.shape[-2:], mode="bilinear")
            l, r, t, b = pad
            up_real = up[:, :, t:t + Hr, l:l + Wr]
            preds += _place(up_real[0], y1, x1, y2, x2, h_img, w_img).unsqueeze(0)
        else:
            feats = model.encode_image(crop_half, model_type, True)  # (1, N, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = feats @ text_features.T                          # (1, N, Q) fp16
            cg, cgd = crop_img.shape[2] // patch, crop_img.shape[3] // patch
            logits = logits.permute(0, 2, 1).reshape(-1, Q, cg, cgd)
            up = F.interpolate(logits, size=crop_img.shape[-2:], mode="bilinear")
            l, r, t, b = pad
            up_real = up[:, :, t:t + Hr, l:l + Wr]
            preds += _place(up_real[0], y1, x1, y2, x2, h_img, w_img).unsqueeze(0)

        count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0

    if heads_mode:
        head_preds = head_preds / count_mat[0]
        head_grid = (head_grid_acc / grid_count.clamp(min=1)).half().cpu()
        return dict(head_logits=head_preds, head_grid=head_grid,
                    grid_count=grid_count.cpu())
    preds = preds / count_mat
    return dict(logits=preds[0])


def official_postprocess(seg_logits, logit_scale=50):
    """clearclip_segmentor.postprocess_result (no synonyms, prob_thd=0).
    3-D input (Q,H,W): softmax/argmax over class dim 0.
    4-D input (h,Q,H,W): per-head, softmax/argmax over class dim 1."""
    if seg_logits.dim() == 3:
        return (seg_logits.float() * logit_scale).softmax(dim=0).argmax(dim=0)
    return (seg_logits.float() * logit_scale).softmax(dim=1).argmax(dim=1)


def confusion_from_pred(pred, gt, num_classes):
    """Aggregated confusion, row=GT, col=pred over gt != 255."""
    m = gt != 255
    idx = gt[m].astype(np.int64) * num_classes + pred[m].astype(np.int64)
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def iou_from_confusion(conf):
    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(1) + conf.sum(0) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return inter / union


def miou(iou):
    return float(np.nanmean(iou))
