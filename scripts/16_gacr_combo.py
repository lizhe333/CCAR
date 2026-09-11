"""GACR combination test: global anchor + uncertainty-gated class residual.

  anchor  A = L^{+h_a}        (single global boost variant, h_a selected on split A only)
  double  B_c = L^{+h_a + h_c*}   (feature F0 + F_{h_a} + F_{h_c*} -> official LN/proj/norm/cos;
                                   classes grouped by h_c*, <=12 distinct combos)
  final   L' = A + u * (B - A),  u = H(softmax_c(50 A))/log C   (frozen, from the winning
                                                                 nocalib recipe of oracle 15)
References in the same pass: anchor-only (A), and double-boost ungated (B, u=1).
Selections are cross-fit: split A's h_c* applied to split B images and vice versa;
anchor h_a from split A applied to everything.
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

from ccar.clip_model import (build_model, build_text_features, _trunk_last_block,
                             _manual_ln_fp32, aggregate_query_logits)
from ccar.pipeline import (DATASETS, dataset_items, preprocess, load_gt,
                           slide_crops, compute_padsize, confusion_from_pred,
                           iou_from_confusion, miou)
from ccar.analysis_utils import split_stems

EPS = 1e-6
LOGIT_SCALE = 50.0


@torch.no_grad()
def gacr_crop_logits(model, tf32, crop_half, h_anchor, h_star,
                     model_type="ClearCLIP", query_idx=None):
    """Returns (A (C,N) anchor logits, B (C,N) double-boost logits per class).
    model_type: "ClearCLIP" (default, unchanged) or "SCLIP" (config v1.7)."""
    if model_type == "SCLIP":
        head_pre_ln, attn, trunk_x = _trunk_last_block(
            model, crop_half, model_type=model_type,
            return_trunk_input=True,
        )
    else:
        head_pre_ln, attn = _trunk_last_block(
            model, crop_half, model_type=model_type
        )
        trunk_x = None
    h32 = head_pre_ln.float()
    full = h32.sum(0) + attn.out_proj.bias.float().unsqueeze(0)      # (L,768)

    def complete(attn_variant):
        if model_type != "SCLIP":
            return attn_variant
        blk = model.visual.transformer.resblocks[-1]
        y = trunk_x.squeeze(1) + attn_variant.to(trunk_x.dtype)
        y = y.unsqueeze(1)
        y = y + blk.mlp(blk.ln_2(y))
        return y.squeeze(1).float()

    A_feat = complete(full + h32[h_anchor])
    X = _manual_ln_fp32(A_feat.unsqueeze(0), model.visual.ln_post)[0]
    toks = X[1:] @ model.visual.proj.float()
    A = ((toks / toks.norm(dim=-1, keepdim=True)) @ tf32.T).T
    C = int(h_star.numel())
    A = aggregate_query_logits(A, query_idx, C)

    N = A.shape[1]
    B = torch.empty(C, N, device=A.device, dtype=A.dtype)
    for h in torch.unique(h_star):
        feats = complete(full + h32[h_anchor] + h32[h])
        X = _manual_ln_fp32(feats.unsqueeze(0), model.visual.ln_post)[0]
        toks = X[1:] @ model.visual.proj.float()
        logits = ((toks / toks.norm(dim=-1, keepdim=True)) @ tf32.T).T
        logits = aggregate_query_logits(logits, query_idx, C)
        mask = h_star == h
        B[mask] = logits[mask]
    return A, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(DATASETS))
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    ds = args.dataset
    cfg = DATASETS[ds]
    C = cfg["num_classes"]
    logC = float(np.log(C))

    # anchor: best boost variant on split A (frozen mechanical rule)
    files = sorted(Path(f"outputs/cache_pert_{ds}").glob("*.npz"))
    stems = [f.stem for f in files]
    a_stems, b_stems = split_stems(ds, stems)
    sa = set(a_stems)
    Cfg = C
    conf_a = np.zeros((25, Cfg, Cfg), dtype=np.int64)
    for s in a_stems:
        conf_a += np.load(f"outputs/cache_pert_{ds}/{s}.npz")["conf"].astype(np.int64)
    inter = np.einsum("vcc->vc", conf_a).astype(float)
    union = conf_a.sum(2) + conf_a.sum(1) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        vm = np.nanmean(inter / union, axis=1) * 100
    v_anchor = 13 + int(np.argmax(vm[13:25]))       # best boost variant on split A
    h_anchor = v_anchor - 13

    sel = json.load(open(f"outputs/results/perturb_sel_{ds}.json"))
    h_bo_A = torch.tensor(sel["h_bo_A"]).long().cuda()   # select on A -> apply on B
    h_bo_B = torch.tensor(sel["h_bo_B"]).long().cuda()   # select on B -> apply on A

    model = build_model(precision="fp16", device="cuda")
    tf, _ = build_text_features(model, str(cfg["cls_file"]))
    tf32 = tf.float()
    items = [(p, g) for p, g in dataset_items(ds) if p.stem in set(stems)]

    keys = ["anchor-only", "double-ungated", "gacr-combined"]
    conf = {k: np.zeros((C, C), dtype=np.int64) for k in keys}
    t0 = time.time()
    for i, (img_path, gt_path) in enumerate(items):
        h_star = h_bo_A if img_path.stem not in sa else h_bo_B
        img_t, ori_hw = preprocess(img_path); img_t = img_t.cuda()
        h_img, w_img = img_t.shape[2], img_t.shape[3]
        Ac = torch.zeros((C, h_img, w_img), dtype=torch.float32, device="cuda")
        Bc = torch.zeros((C, h_img, w_img), dtype=torch.float32, device="cuda")
        count_mat = torch.zeros((1, 1, h_img, w_img), dtype=torch.float32, device="cuda")
        gt = load_gt(gt_path, cfg["reduce_zero_label"],
                         label_map=cfg.get("label_map"))
        cc = torch.arange(C, device="cuda")
        with torch.no_grad():
            for (y1, x1, y2, x2) in slide_crops(h_img, w_img):
                crop = img_t[:, :, y1:y2, x1:x2]
                Hr, Wr = crop.shape[2:]
                pad = compute_padsize(Hr, Wr, 16)
                if any(pad):
                    crop = nn.functional.pad(crop, pad)
                A, B = gacr_crop_logits(model, tf32, crop.half(), h_anchor, h_star)
                cg, cgd = crop.shape[2] // 16, crop.shape[3] // 16
                for field, canvas in [(A, Ac), (B, Bc)]:
                    up = F.interpolate(field.reshape(1, C, cg, cgd),
                                       size=crop.shape[-2:], mode="bilinear")[0]
                    l, r, t, b = pad
                    canvas += nn.functional.pad(up[:, t:t + Hr, l:l + Wr].unsqueeze(0),
                                                (int(x1), int(w_img - x2), int(y1), int(h_img - y2)))[0]
                count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0
            A = Ac / count_mat[0]
            B = Bc / count_mat[0]
            P0 = (A * LOGIT_SCALE).softmax(dim=0)
            u = -(P0 * (P0 + EPS).log()).sum(0) / logC
            maps = {"anchor-only": A, "double-ungated": B, "gacr-combined": A + u * (B - A)}
            for k, mp in maps.items():
                best_val = torch.full((ori_hw[0], ori_hw[1]), -float("inf"), device="cuda")
                best_idx = torch.zeros((ori_hw[0], ori_hw[1]), dtype=torch.int64, device="cuda")
                for c0 in range(0, C, 25):
                    lg = F.interpolate(mp[c0:c0 + 25].unsqueeze(0), size=ori_hw,
                                       mode="bilinear")[0] * LOGIT_SCALE
                    v, idx = lg.max(0)
                    upd = v > best_val
                    best_val[upd] = v[upd]
                    best_idx[upd] = idx[upd] + c0
                pred = best_idx.cpu().numpy().astype(np.uint8)
                conf[k] += confusion_from_pred(pred, gt, C)
            del Ac, Bc, A, B, P0, maps
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(items)}] {((i+1)/(time.time()-t0)):.1f} img/s", flush=True)

    raw = {"voc20": 80.93, "context59": 35.84, "ade20k": 16.64}[ds]
    print(f"\n=== {ds} GACR combination (anchor=v{v_anchor} boost h{h_anchor}, "
          f"{len(items)} imgs, {time.time()-t0:.0f}s) ===")
    print(f"{'Raw (Gate-1)':20s} {raw:.2f}")
    out = {"dataset": ds, "anchor_variant": int(v_anchor), "raw": raw}
    for k in keys:
        m = miou(iou_from_confusion(conf[k])) * 100
        out[k] = m
        print(f"{k:20s} {m:.2f}  ({m-raw:+.2f} vs Raw)")
    json.dump(out, open(f"outputs/results/gacr_combo_{ds}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
