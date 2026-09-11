"""CLIP model loading, text features, and per-head decomposition of the
ClearCLIP final block.

Vendored open_clip comes from ClearCLIP repo @ ad68a404d55d48d27330b93554eb64a234ff717f
(see PROVENANCE.md). All official code paths are used verbatim wherever possible;
the decomposed forward replicates open_clip/transformer.py VisionTransformer.forward
(lines 500-534) + custom_attn (lines 589-622) exactly, split per attention head.

Per-head feature definition (fixed for this project):
    F_h_preLN = (A_h V_h) @ W_out[64h:64(h+1), :]^T        (no out_proj bias; bias is head-independent)
    feat_h    = L2norm( ln_post(F_h_preLN[:, 1:, :]) @ visual.proj )
    L_{p,c,h} = <feat_h[p], t_c>
Recomposition identity: sum_h F_h_preLN + out_proj.bias == custom_attn output (pre-ln_post).
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CCAR_PKG = Path(__file__).resolve().parent          # <work>/ccar/ccar
if str(CCAR_PKG) not in sys.path:
    sys.path.insert(0, str(CCAR_PKG))

from open_clip import load_openai_model  # noqa: E402  (vendored)
from open_clip.transformer import _expand_token  # noqa: E402
from prompts.imagenet_template import openai_imagenet_template  # noqa: E402
from open_clip import tokenizer as _tokenizer  # noqa: E402

DEFAULT_CHECKPOINT_DIR = CCAR_PKG.parent / "checkpoints"
CKPT = os.environ.get(
    "CCAR_CHECKPOINT",
    str(DEFAULT_CHECKPOINT_DIR / "ViT-B-16.pt"),
)
CKPT_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
CKPT_VITL = os.environ.get(
    "CCAR_CHECKPOINT_VITL",
    str(DEFAULT_CHECKPOINT_DIR / "ViT-L-14.pt"),
)
# 16 heads x 64 head_dim (head_dim invariant vs B/16); slide_inference call
# sites must pass num_heads=16, patch=14 for ViT-L/14 (config v1.7 vitl).

def build_model(precision="fp16", device="cuda", vit_type="ViT-B/16"):
    """Official load path: load_openai_model applies QuickGELU for OpenAI weights.
    vit_type="ViT-L/14" loads CKPT_VITL (config v1.7 vitl); the default path is
    unchanged (same checkpoint, same call)."""
    ckpt = Path(CKPT_VITL if vit_type == "ViT-L/14" else CKPT).expanduser()
    if not ckpt.is_file():
        env_name = "CCAR_CHECKPOINT_VITL" if vit_type == "ViT-L/14" else "CCAR_CHECKPOINT"
        raise FileNotFoundError(
            f"CLIP checkpoint not found: {ckpt}. Set {env_name} to the checkpoint path."
        )
    model = load_openai_model(str(ckpt), precision=precision, device=device)
    model.eval()
    return model


def get_cls_idx(path):
    """Verbatim from clearclip_segmentor.py get_cls_idx."""
    with open(path, "r") as f:
        name_sets = f.readlines()
    class_names, class_indices = [], []
    for idx in range(len(name_sets)):
        names_i = name_sets[idx].split(",")
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace("\n", "").strip() for item in class_names]
    return class_names, class_indices


@torch.no_grad()
def build_text_features(model, cls_file, device="cuda"):
    """Verbatim replication of ClearCLIPSegmentation.__init__ text pipeline
    (80-template OpenAI ImageNet ensemble, fp16 like the official segmentor)."""
    query_words, query_idx = get_cls_idx(cls_file)
    query_features = []
    for qw in query_words:
        query = _tokenizer.tokenize([temp(qw) for temp in openai_imagenet_template]).to(device)
        feature = model.encode_text(query)
        feature /= feature.norm(dim=-1, keepdim=True)
        feature = feature.mean(dim=0)
        feature /= feature.norm()
        query_features.append(feature.unsqueeze(0))
    return torch.cat(query_features, dim=0), torch.tensor(query_idx, dtype=torch.int64, device=device)


def aggregate_query_logits(logits, query_idx, num_classes):
    """Max-pool synonym query logits into class logits.

    This preserves the official SCLIP argmax rule for no-background datasets
    while exposing one score field per semantic class for CCAR selection.
    ``logits`` may be (Q,N) or (V,Q,N).
    """
    if query_idx is None or logits.shape[-2] == num_classes:
        return logits
    idx = torch.as_tensor(query_idx, device=logits.device, dtype=torch.long)
    out_shape = list(logits.shape)
    out_shape[-2] = num_classes
    out = torch.full(
        out_shape, -float("inf"), device=logits.device, dtype=logits.dtype
    )
    for c in range(num_classes):
        mask = idx == c
        if not torch.any(mask):
            raise ValueError(f"class {c} has no text query")
        out[..., c, :] = logits[..., mask, :].amax(dim=-2)
    return out


@torch.no_grad()
def _trunk_last_block(
    model, img, return_attn_weights=False, model_type="ClearCLIP",
    return_trunk_input=False,
):
    """Replicates VisionTransformer.forward trunk + custom_attn branch
    (open_clip/transformer.py @ ad68a40, lines 500-534 / 589-622) up to the final
    block's per-head pre-LN contributions. Returns (head_pre_ln (H,L,768) fp32-ish
    in model dtype, attn module). With return_attn_weights=True, additionally
    returns the final-layer attention weights (H, L, L, model dtype) — used only
    by the SHA-style entropy selection control (config v1.6); the math is
    unchanged and default callers are unaffected (identity-gated by scripts/25).
    model_type: "ClearCLIP" (softmax(qq), default, unchanged) or "SCLIP"
    (softmax(qq) + softmax(kk), transformer.py:610-613 verbatim — both softmaxes
    are within-head, so the per-head decomposition identity is unchanged).
    """
    visual = model.visual
    x = visual.conv1(img)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    x = torch.cat([_expand_token(visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
    if x.shape[1] != visual.positional_embedding.shape[0]:
        x = x + visual.interpolate_pos_encoding(x, img.shape[2], img.shape[3]).to(x.dtype)
    else:
        x = x + visual.positional_embedding.to(x.dtype)
    x = visual.patch_dropout(x)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)                                   # NLD -> LND
    for blk in visual.transformer.resblocks[:-1]:
        x = blk(x)
    blk = visual.transformer.resblocks[-1]
    attn = blk.attn
    num_heads = attn.num_heads
    xln = blk.ln_1(x)
    _, bsz, embed_dim = xln.size()
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5
    q, k, v = F.linear(xln, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    qq_attn = torch.bmm(q, q.transpose(1, 2)) * scale
    if model_type == "SCLIP":
        kk_attn = torch.bmm(k, k.transpose(1, 2)) * scale
        attn_weights = F.softmax(qq_attn, dim=-1) + F.softmax(kk_attn, dim=-1)
    else:
        attn_weights = F.softmax(qq_attn, dim=-1)            # ClearCLIP self-self attention
    attn_output = torch.bmm(attn_weights, v)                 # (bsz*H, L, d)
    attn_output_ld = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
    W = attn.out_proj.weight
    head_pre_ln = []
    for h in range(num_heads):
        head_ctx = attn_output_ld[:, :, h * head_dim:(h + 1) * head_dim]   # (L, 1, d)
        W_h = W[:, h * head_dim:(h + 1) * head_dim]                        # (768_out, d_in) column block
        head_pre_ln.append((head_ctx @ W_h.t()).squeeze(1))                # (L, 768)
    heads = torch.stack(head_pre_ln, dim=0)
    if return_attn_weights and return_trunk_input:
        return heads, attn, attn_weights, x
    if return_attn_weights:
        return heads, attn, attn_weights
    if return_trunk_input:
        return heads, attn, x
    return heads, attn             # (H, L, 768), attn module


def _manual_ln_fp32(x, ln):
    """LayerNorm in fp32 with module parameters (eps preserved)."""
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + ln.eps) * ln.weight.float() + ln.bias.float()


@torch.no_grad()
def forward_perturbation_variants(model, img, text_features, model_type="ClearCLIP"):
    """Full-MHA perturbation variants in the FEATURE domain (pre-ln_post 768-d).

    variant 0        : x_full = sum_j F_j + b           (== custom_attn output)
    variants 1..H    : REMOVE_h: x_full - F_h
    variants H+1..2H : BOOST_h : x_full + F_h           (lambda = 1)

    All variants go through the OFFICIAL post: ln_post -> drop CLS -> @visual.proj
    -> L2norm (fp32 math; fp16 backbone). Returns (2H+1, Q, N) cosine logits.
    model_type: "ClearCLIP" (default) or "SCLIP" — plumbed to _trunk_last_block.
    """
    if model_type == "SCLIP":
        head_pre_ln, attn, trunk_x = _trunk_last_block(
            model, img, model_type=model_type, return_trunk_input=True
        )
    else:
        head_pre_ln, attn = _trunk_last_block(
            model, img, model_type=model_type
        )
        trunk_x = None
    Hn = head_pre_ln.shape[0]
    h32 = head_pre_ln.float()
    full = h32.sum(dim=0) + attn.out_proj.bias.float().unsqueeze(0)   # (L, 768)
    variants = [full]
    for h in range(Hn):
        variants.append(full - h32[h])
    for h in range(Hn):
        variants.append(full + h32[h])
    if model_type == "SCLIP":
        # Official SCLIP retains the last block residual and MLP. Apply every
        # attention-head perturbation before that frozen nonlinear tail.
        blk = model.visual.transformer.resblocks[-1]
        trunk = trunk_x.squeeze(1)
        completed = []
        for variant in variants:
            y = trunk + variant.to(trunk.dtype)
            y_lnd = y.unsqueeze(1)
            y_lnd = y_lnd + blk.mlp(blk.ln_2(y_lnd))
            completed.append(y_lnd.squeeze(1).float())
        X = torch.stack(completed, dim=0)
    else:
        X = torch.stack(variants, dim=0)                      # (25, L, 768)
    X = _manual_ln_fp32(X, model.visual.ln_post)
    tokens = X[:, 1:, :]                                      # drop CLS
    if model.visual.proj is not None:
        tokens = tokens @ model.visual.proj.float()
    feats = tokens / tokens.norm(dim=-1, keepdim=True)        # (25, N, 512)
    logits = torch.einsum("vnd,qd->vqn", feats, text_features.float())
    return logits


@torch.no_grad()
def forward_head_decomposed(model, img, return_official=True, model_type="ClearCLIP"):
    """Single forward through ViT trunk; returns per-head dense features.

    Args:
        img: (1, 3, H, W) on model device/dtype. H,W multiples of 16 (padded crops).
        model_type: "ClearCLIP" (default, unchanged) or "SCLIP".
    Returns dict:
        feats:   (num_heads, N, D)  L2-normalized per-head patch features (CLS dropped)
        pre_ln_sum_biasless: (L, 1, 768) sum_h F_h_preLN (LND layout, no out_proj bias)
        head_pre_ln: (num_heads, L, 768) per-head pre-LN contributions, LND layout (bsz=1)
        official: (N, D) official encode_image output (if return_official)
    Exact replication notes (vs open_clip/transformer.py @ ad68a40):
      - lines 501-521: conv1 / class emb / pos emb (interp) / ln_pre / blocks[:-1]
      - custom_attn ClearCLIP branch: A = softmax((q q^T) * d^-0.5); out = (A v)
      - out_proj slicing: full out = concat_h(head_h) @ W^T + b, so per-head rows of W
        give head contributions whose SUM (+ b) equals the official output.
      - post: output.permute(1,0,2) -> ln_post -> tokens = x[:, 1:] -> @ proj
    """
    visual = model.visual
    B = img.shape[0]
    assert B == 1, "decomposed path assumes batch size 1 (official protocol)"
    x = visual.conv1(img)                                   # [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)   # [*, grid^2, width]

    x = torch.cat([_expand_token(visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
    if x.shape[1] != visual.positional_embedding.shape[0]:
        x = x + visual.interpolate_pos_encoding(x, img.shape[2], img.shape[3]).to(x.dtype)
    else:
        x = x + visual.positional_embedding.to(x.dtype)

    x = visual.patch_dropout(x)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)                                  # NLD -> LND

    for blk in visual.transformer.resblocks[:-1]:
        x = blk(x)

    blk = visual.transformer.resblocks[-1]
    attn = blk.attn
    num_heads = attn.num_heads
    xln = blk.ln_1(x)
    _, bsz, embed_dim = xln.size()
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5

    q, k, v = F.linear(xln, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    qq_attn = torch.bmm(q, q.transpose(1, 2)) * scale
    if model_type == "SCLIP":
        kk_attn = torch.bmm(k, k.transpose(1, 2)) * scale
        attn_weights = F.softmax(qq_attn, dim=-1) + F.softmax(kk_attn, dim=-1)
    else:
        attn_weights = F.softmax(qq_attn, dim=-1)           # ClearCLIP self-self attention
    attn_output = torch.bmm(attn_weights, v)                # (bsz*H, L, d)

    # official layout: transpose -> view(-1, bsz, embed) interleaves heads (bsz=1 => concat)
    attn_output_ld = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
    W = attn.out_proj.weight                                # (768, 768): out = inp @ W.T + b

    head_pre_ln = []
    for h in range(num_heads):
        head_ctx = attn_output_ld[:, :, h * head_dim:(h + 1) * head_dim]   # (L, 1, d)
        W_h = W[:, h * head_dim:(h + 1) * head_dim]                        # (768_out, d_in) column block
        head_pre_ln.append(head_ctx @ W_h.t())                             # (L, 1, 768)
    head_pre_ln = torch.stack(head_pre_ln, dim=0)           # (H, L, 1, 768) -> squeeze below
    head_pre_ln = head_pre_ln.squeeze(2)                    # (H, L, 768)
    pre_ln_sum_biasless = head_pre_ln.sum(dim=0, keepdim=True).transpose(0, 1)  # (L, 1, 768)

    feats = []
    for h in range(num_heads):
        # plan definition (§3): F_h = (A_h V_h) @ W_out[:, head slice] @ visual.proj
        # — strictly linear per head, NO per-head ln_post (LayerNorm belongs to the
        # official full-feature path where it is applied to the SUM; the recomposition
        # sanity check is defined pre-LN, see config.yaml decomposition.fp32_tolerance)
        tokens = head_pre_ln[h][1:, :]                      # (N, 768), drop CLS row (LND)
        if visual.proj is not None:
            tokens = tokens @ visual.proj                   # (N, 512)
        feats.append(tokens)
    feats = torch.stack(feats, dim=0)                       # (H, N, D)
    feats = feats / feats.norm(dim=-1, keepdim=True)

    # EXACT LN-aware per-head features (mode 'ln_affine'):
    # ln_post is affine per patch given the FULL vector's statistics:
    #   LN(x) = sum_h [gamma * (F_h - mu_h) / sigma_p] + gamma*(b - mu_b)/sigma_p + beta
    # where mu_h = channel-mean of head contribution, sigma_p/mu_b from the full
    # attn output. Recombination reproduces the official feature EXACTLY, while
    # keeping every head in the text-aligned normalized space (the plan's raw-linear
    # version measured noise-level per-head mIoU because it bypassed ln_post).
    full = head_pre_ln.sum(dim=0) + attn.out_proj.bias      # (L, 768) == custom_attn out
    # fp32 for the affine decomposition (small per-head contributions lose precision
    # in fp16; official-path equivalence is untouched — this is our analysis math)
    full = full.float()
    head_pre_ln32 = head_pre_ln.float()
    mu_full = full.mean(dim=-1, keepdim=True)               # (L, 1)
    sigma_full = full.std(dim=-1, unbiased=False, keepdim=True)  # (L, 1)
    gamma = visual.ln_post.weight.float()                   # (768,)
    beta = visual.ln_post.bias.float()
    mu_h = head_pre_ln32.mean(dim=-1, keepdim=True)         # (H, L, 1)
    aff = (head_pre_ln32 - mu_h) / sigma_full               # (H, L, 768)
    mu_bias = attn.out_proj.bias.float().mean()             # scalar mean of out_proj bias
    const_ln = gamma * (attn.out_proj.bias.float().unsqueeze(0) - mu_bias) / sigma_full + beta  # (L, 768)
    head_ln_affine = aff * gamma                            # (H, L, 768) = per-head LN contributions
    ln_feats = []
    for h in range(num_heads):
        tokens = head_ln_affine[h][1:, :]                   # drop CLS (fp32)
        if visual.proj is not None:
            tokens = tokens @ visual.proj.float()           # (N, 512)
        ln_feats.append(tokens)
    ln_feats = torch.stack(ln_feats, dim=0)                 # (H, N, D) RAW (recomposable)
    ln_feats_norm = ln_feats / ln_feats.norm(dim=-1, keepdim=True)
    const_feat = const_ln[1:, :]                            # (N, 768) fp32
    if visual.proj is not None:
        const_feat = const_feat @ visual.proj.float()       # (N, D) not normalized

    # official-path features reconstructed from the per-head decomposition:
    # recombination MUST happen pre-ln_post (LayerNorm is nonlinear);
    # sum_h F_h + b == custom_attn output exactly (see Gate-2 check [1])
    feats_sum = (head_pre_ln.sum(dim=0, keepdim=True).transpose(0, 1)
                 + attn.out_proj.bias)                    # (L, 1, 768)
    feats_sum = model.visual.ln_post(feats_sum.permute(1, 0, 2))
    tokens_sum = feats_sum[:, 1:]
    if visual.proj is not None:
        tokens_sum = tokens_sum @ visual.proj
    tokens_sum = tokens_sum / tokens_sum.norm(dim=-1, keepdim=True)

    out = dict(feats=feats, ln_feats=ln_feats_norm, ln_feats_raw=ln_feats,
               const_feat=const_feat, head_pre_ln=head_pre_ln,
               pre_ln_sum_biasless=pre_ln_sum_biasless,
               feats_recomposed=tokens_sum,
               official_attn_out=attn.out_proj(attn_output_ld))  # (L, 1, 768) LND, exact custom_attn output
    if return_official:
        out["official"] = model.encode_image(img, model_type, True)
    return out
