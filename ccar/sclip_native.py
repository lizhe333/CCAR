"""Official SCLIP per-dataset inference contracts and post-processing.

The public SCLIP configurations do not use one common input scale or score
calibration.  This module keeps those contracts separate from the legacy
ClearCLIP/CCAR protocols and provides the query-level post-processing used by
the official implementation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ccar.clip_model import forward_perturbation_variants
from ccar.pipeline import compute_padsize, slide_crops


NATIVE_DATASET_CONFIG = {
    "voc21": {
        "short_side": 336,
        "crop": 224,
        "stride": 112,
        "logit_scale": 65.0,
        "prob_thd": 0.1,
        "area_thd": 0.1,
    },
    "context60": {
        "short_side": 336,
        "crop": 224,
        "stride": 112,
        "logit_scale": 50.0,
        "prob_thd": 0.1,
        "area_thd": None,
    },
    "cocoobject": {
        "short_side": 336,
        "crop": 224,
        "stride": 112,
        "logit_scale": 50.0,
        "prob_thd": 0.1,
        "area_thd": None,
    },
    "voc20": {
        "short_side": 336,
        "crop": 224,
        "stride": 112,
        "logit_scale": 40.0,
        "prob_thd": 0.0,
        "area_thd": None,
    },
    "context59": {
        "short_side": 336,
        "crop": 224,
        "stride": 112,
        "logit_scale": 40.0,
        "prob_thd": 0.0,
        "area_thd": None,
    },
    "ade20k": {
        "short_side": 336,
        "crop": 224,
        "stride": 112,
        "logit_scale": 40.0,
        "prob_thd": 0.0,
        "area_thd": None,
    },
    "cocostuff": {
        "short_side": 448,
        "crop": 224,
        "stride": 112,
        "logit_scale": 40.0,
        "prob_thd": 0.0,
        "area_thd": None,
    },
    "cityscapes": {
        "short_side": 560,
        "crop": 224,
        "stride": 112,
        "logit_scale": 40.0,
        "prob_thd": 0.0,
        "area_thd": None,
    },
}


def official_query_prediction(
    field: torch.Tensor,
    ori_hw: tuple[int, int],
    query_idx: torch.Tensor,
    num_classes: int,
    *,
    logit_scale: float,
    prob_thd: float,
    area_thd: float | None,
) -> torch.Tensor:
    """Apply ``clip_segmentor.py`` post-processing to query-level logits.

    ``field`` is ``(num_queries, H, W)`` after the slide-window average.  The
    official code first softmaxes over queries, then max-pools synonyms into
    classes, applies the optional area filter, and finally applies the
    confidence threshold.
    """
    probs = F.interpolate(
        field.unsqueeze(0), size=ori_hw, mode="bilinear", align_corners=False
    )[0].float()
    probs = (probs * float(logit_scale)).softmax(dim=0)

    q_idx = torch.as_tensor(query_idx, device=probs.device, dtype=torch.long)
    if probs.shape[0] == num_classes:
        class_probs = probs
    else:
        shape = (num_classes, *probs.shape[1:])
        class_probs = torch.zeros(shape, device=probs.device, dtype=probs.dtype)
        class_probs.scatter_reduce_(
            0,
            q_idx.reshape(-1, 1, 1).expand_as(probs),
            probs,
            reduce="amax",
            include_self=True,
        )

    if area_thd is not None:
        predictions = class_probs.argmax(dim=0)
        areas = torch.bincount(
            predictions.reshape(-1), minlength=num_classes
        ).to(class_probs.dtype)
        foreground = areas[1:]
        keep = foreground > float(area_thd) * foreground.sum()
        class_probs = class_probs.clone()
        class_probs[1:] *= keep.reshape(-1, 1, 1)

    pred = class_probs.argmax(dim=0)
    if prob_thd > 0:
        pred = pred.clone()
        pred[class_probs.max(dim=0).values < float(prob_thd)] = 0
    return pred


@torch.no_grad()
def native_perturbation_confusion(
    model,
    tf32: torch.Tensor,
    img_t: torch.Tensor,
    ori_hw: tuple[int, int],
    gt,
    num_classes: int,
    *,
    model_type: str,
    crop: int,
    stride: int,
    logit_scale: float,
    prob_thd: float,
    area_thd: float | None,
    query_idx: torch.Tensor,
    patch: int = 16,
    chunk: int = 6,
) -> object:
    """Build the 25-variant training cache under official SCLIP semantics."""
    import numpy as np

    h_img, w_img = img_t.shape[2:]
    crops = list(slide_crops(h_img, w_img, crop=crop, stride=stride))
    gt_t = torch.as_tensor(gt, device="cuda", dtype=torch.long)
    valid = gt_t != 255
    gv = gt_t[valid].long()
    n_heads = model.visual.transformer.resblocks[-1].attn.num_heads
    n_variants = 2 * n_heads + 1
    conf = torch.zeros(
        (n_variants, num_classes, num_classes),
        dtype=torch.int64,
        device="cuda",
    )
    count_mat = torch.zeros(
        (1, 1, h_img, w_img), dtype=torch.float32, device="cuda"
    )
    staged = []

    for y1, x1, y2, x2 in crops:
        crop_img = img_t[:, :, y1:y2, x1:x2]
        hr, wr = crop_img.shape[2:]
        pad = compute_padsize(hr, wr, patch)
        if any(pad):
            crop_img = F.pad(crop_img, pad)
        logits = forward_perturbation_variants(
            model, crop_img.half().to("cuda"), tf32, model_type=model_type
        )
        query_count = logits.shape[1]
        gh, gw = crop_img.shape[2] // patch, crop_img.shape[3] // patch
        staged.append(
            (
                logits.reshape(n_variants, query_count, gh, gw).cpu(),
                pad, y1, x1, y2, x2, hr, wr, crop_img.shape[2], crop_img.shape[3],
            )
        )
        count_mat[:, :, y1:y2, x1:x2] += 1

    if torch.any(count_mat == 0):
        raise AssertionError("native SCLIP cache has uncovered pixels")

    for c0 in range(0, n_variants, chunk):
        n_chunk = min(chunk, n_variants - c0)
        query_count = staged[0][0].shape[1]
        canvas = torch.zeros(
            (n_chunk, query_count, h_img, w_img),
            dtype=torch.float32,
            device="cuda",
        )
        for logits_cpu, pad, y1, x1, y2, x2, hr, wr, hp, wp in staged:
            logits = logits_cpu[c0:c0 + n_chunk].to("cuda", non_blocking=True)
            up = F.interpolate(logits, size=(hp, wp), mode="bilinear")
            left, right, top, bottom = pad
            up_real = up[:, :, top:top + hr, left:left + wr]
            for k in range(up_real.shape[0]):
                canvas[k] += F.pad(
                    up_real[k].unsqueeze(0),
                    (int(x1), int(w_img - x2), int(y1), int(h_img - y2)),
                )[0]
        canvas /= count_mat
        for k in range(n_chunk):
            up_ori = F.interpolate(canvas[k:k + 1], size=ori_hw, mode="bilinear")[0]
            pred = official_query_prediction(
                up_ori,
                ori_hw,
                query_idx,
                num_classes,
                logit_scale=logit_scale,
                prob_thd=prob_thd,
                area_thd=area_thd,
            )
            idx = gv * num_classes + pred[valid].long()
            conf[c0 + k] += torch.bincount(
                idx, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    return conf.cpu().numpy().astype(np.uint32)

