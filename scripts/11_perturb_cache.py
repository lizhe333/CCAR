"""Perturbation-oracle cache: per-image confusions for 25 full-MHA variants
(base, remove_1..12, boost_1..12) under the exact official slide geometry.

Variant indexing: 0 = base; 1..12 = remove_h (x_full - F_h); 13..24 = boost_h.
All recomposition in fp32 feature domain (see ccar/clip_model.forward_perturbation_variants).
Canvas chunking (6 variants at a time) keeps peak VRAM bounded on large ADE images.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ccar.clip_model import (build_model, build_text_features,
                             forward_perturbation_variants,
                             aggregate_query_logits)
from ccar.pipeline import (DATASETS, dataset_items, preprocess, load_gt,
                           slide_crops, compute_padsize)

CACHE = {"voc20": "outputs/cache_pert_voc20", "context59": "outputs/cache_pert_context59",
         "ade20k": "outputs/cache_pert_ade20k"}
CHUNK = 2   # small: this job coexists with other GPU work on the box


def image_confusion(model, tf32, img_t, ori_hw, gt, C,
                   model_type="ClearCLIP", patch=16,
                   crop=448, stride=224, logit_scale=50.0,
                   query_idx=None):
    """(2H+1, C, C) confusion for one image, official geometry per variant.
    Phase 1: backbone once per crop (variant logits staged to CPU).
    Phase 2: CHUNK-variant canvases, then /count_mat, upsample to ori, argmax."""
    h_img, w_img = img_t.shape[2], img_t.shape[3]
    Q = C
    crops = list(slide_crops(h_img, w_img, crop=crop, stride=stride))
    count_mat = torch.zeros((1, 1, h_img, w_img), dtype=torch.float32, device="cuda")
    n_heads = model.visual.transformer.resblocks[-1].attn.num_heads
    n_variants = 2 * n_heads + 1
    conf = torch.zeros((n_variants, C, C), dtype=torch.int64, device="cuda")
    gt_t = torch.from_numpy(gt).cuda()
    valid = gt_t != 255
    gv = gt_t[valid].long()

    staged = []
    for (y1, x1, y2, x2) in crops:
        crop = img_t[:, :, y1:y2, x1:x2]
        Hr, Wr = crop.shape[2:]
        pad = compute_padsize(Hr, Wr, patch)
        if any(pad):
            crop = nn.functional.pad(crop, pad)
        with torch.no_grad():
            logits = forward_perturbation_variants(
                model, crop.half().to("cuda"), tf32, model_type=model_type
            )
            logits = aggregate_query_logits(logits, query_idx, C)
        cg, cgd = crop.shape[2] // patch, crop.shape[3] // patch
        assert logits.shape[0] == n_variants, f"expected {n_variants} perturbation variants, got {logits.shape[0]}"
        staged.append((logits.reshape(n_variants, -1, cg, cgd).cpu(), pad,
                       y1, x1, y2, x2, Hr, Wr, crop.shape[2], crop.shape[3]))
        count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0

    for c0 in range(0, n_variants, CHUNK):
        canvas = torch.zeros((CHUNK, Q, h_img, w_img), dtype=torch.float32, device="cuda")
        for (lg_cpu, pad, y1, x1, y2, x2, Hr, Wr, Hp, Wp) in staged:
            lg = lg_cpu[c0:c0 + CHUNK].to("cuda", non_blocking=True)
            up = F.interpolate(lg, size=(Hp, Wp), mode="bilinear")
            l, r, t, b = pad
            up_real = up[:, :, t:t + Hr, l:l + Wr]
            for k in range(up_real.shape[0]):
                canvas[k] += nn.functional.pad(
                    up_real[k].unsqueeze(0),
                    (int(x1), int(w_img - x2), int(y1), int(h_img - y2)))[0]
        canvas /= count_mat
        for k in range(min(CHUNK, n_variants - c0)):
            v = c0 + k
            up_ori = F.interpolate(canvas[k:k + 1], size=ori_hw, mode="bilinear")[0]
            pred = ((up_ori * logit_scale).softmax(0)).argmax(0)
            idx = gv * C + pred[valid].long()
            conf[v] += torch.bincount(idx, minlength=C * C).reshape(C, C)
    return conf.cpu().numpy().astype(np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(DATASETS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    ds = args.dataset
    cfg = DATASETS[ds]
    out_dir = Path(CACHE[ds]); out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(precision="fp16", device="cuda")
    tf, _ = build_text_features(model, str(cfg["cls_file"]))
    items = dataset_items(ds, limit=args.limit)
    C = cfg["num_classes"]

    t0 = time.time(); peak = 0
    for i, (img_path, gt_path) in enumerate(items):
        npz_path = out_dir / (img_path.stem + ".npz")
        if npz_path.exists():
            continue
        img_t, ori_hw = preprocess(img_path)
        torch.cuda.reset_peak_memory_stats()
        gt = load_gt(gt_path, cfg["reduce_zero_label"],
                         label_map=cfg.get("label_map"))
        conf = image_confusion(model, tf, img_t.to("cuda"), ori_hw, gt, C)
        np.savez_compressed(npz_path, conf=conf)
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**20)
        if (i + 1) % 250 == 0:
            print(f"  [{i+1}/{len(items)}] {((i+1)/(time.time()-t0)):.1f} img/s "
                  f"peak {peak:.0f}MiB", flush=True)

    n = len(list(out_dir.glob("*.npz")))
    dt = time.time() - t0
    print(f"=== {ds}: {n} images cached, {dt:.0f}s, cache "
          f"{sum(f.stat().st_size for f in out_dir.glob('*.npz'))/2**20:.0f}MiB ===")
    with open(f"outputs/results/cache_pert_{ds}_meta.json", "w") as f:
        json.dump(dict(dataset=ds, n_cached=n, seconds=dt, peak_vram_mib=peak), f, indent=1)


if __name__ == "__main__":
    main()
