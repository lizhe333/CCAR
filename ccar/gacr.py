"""Practical GACR logit construction. Replicates scripts/16_gacr_combo.py
gacr_crop_logits EXACTLY (same fp32 op order), minus the trunk forward —
operating on a stashed (H, L, 768) head_pre_ln tensor (fp16, CLS row included,
as returned by ccar.clip_model._trunk_last_block). Frozen: do not modify op
order; scripts/17 asserts equality against a verbatim copy of script 16.

Boost convention (identical to script 16): feats = (full + F_ha) + F_hc; when
h_c == h_a this is implicitly full + 2*F_ha (double boost). Every derived
variant below uses the same convention.
"""
import torch

from .clip_model import _manual_ln_fp32

EPS = 1e-6
LOGIT_SCALE = 50.0


def _cos_logits(feats, ln_post, proj32, tf32):
    """(L,768) fp32 -> official post (manual LN fp32 -> drop CLS -> proj ->
    L2norm) -> (C, N) class-major cosine logits."""
    X = _manual_ln_fp32(feats.unsqueeze(0), ln_post)[0]
    toks = X[1:] @ proj32
    toks = toks / toks.norm(dim=-1, keepdim=True)
    return (toks @ tf32.T).T


def _grouped_boost(h32, full, ln_post, proj32, tf32, head_of_class, extra_anchor):
    """Per-class boosted logits, classes grouped by boost head (script 16
    pattern). feats_c = (full + F_anchor if extra_anchor is not None else full)
    + F_hc — anchor added FIRST to keep script 16's exact op order."""
    C = tf32.shape[0]
    out = torch.empty(C, full.shape[0] - 1, device=full.device, dtype=torch.float32)
    oc = torch.as_tensor(head_of_class, device=full.device, dtype=torch.long)
    for h in torch.unique(oc):
        feats = (full + h32[int(extra_anchor)]) if extra_anchor is not None else full
        feats = feats + h32[int(h)]
        logits = _cos_logits(feats, ln_post, proj32, tf32)
        out[oc == h] = logits[oc == h]
    return out


def ab_logits_from_heads(head_pre_ln, out_proj_bias, ln_post, proj32, tf32,
                         h_anchor, h_class):
    """Returns dict of (C, N) fp32 class-major logits:
      A               full + F_ha
      B               full + F_ha + F_hc   (h_c == h_a -> full + 2 F_ha)
      B_class_only    full + F_hc
      B_double_anchor full + F_ha + F_ha   (global-only residual control)
    """
    h32 = head_pre_ln.float()
    full = h32.sum(0) + out_proj_bias.float().unsqueeze(0)
    hc = torch.as_tensor(h_class, device=h32.device, dtype=torch.long)
    A = _cos_logits(full + h32[int(h_anchor)], ln_post, proj32, tf32)
    B = _grouped_boost(h32, full, ln_post, proj32, tf32, hc,
                       extra_anchor=int(h_anchor))
    B_class_only = _grouped_boost(h32, full, ln_post, proj32, tf32, hc,
                                  extra_anchor=None)
    B_double_anchor = _grouped_boost(
        h32, full, ln_post, proj32, tf32,
        torch.full_like(hc, int(h_anchor)), extra_anchor=int(h_anchor))
    return dict(A=A, B=B, B_class_only=B_class_only, B_double_anchor=B_double_anchor)


def uncertainty_gate(A, logC, eps=EPS, logit_scale=LOGIT_SCALE):
    """u_p = H(softmax_c(scale*A))/log C per pixel.

    The default remains the frozen script-16 scale of 50. Protocol runners can
    pass their frozen scale explicitly without changing the gate definition.
    """
    P0 = (A * logit_scale).softmax(dim=0)
    return -(P0 * (P0 + eps).log()).sum(0) / logC
