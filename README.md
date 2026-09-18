# CCAR

Official PyTorch implementation of **CCAR: Class-Conditioned Attention Residuals with Competition-Aware Head Selection for Frozen CLIP Segmentation**.

CCAR is a training-free plug-in for CLIP-based open-vocabulary semantic segmentation. A head that performs well for a category when shared across all classes may become less effective once each class uses its own head, because the competing class scores change. CCAR therefore assigns class-specific residual heads through **competition-aware head selection**: each candidate replaces only the target-class score while competing scores stay fixed at an initial reference prediction, and the resulting mask's calibration-set IoU determines the selected head. Entropy gating fuses shared and class-specific scores for calibration and inference. All network weights remain frozen; one offline calibration round uses pixel annotations from 10% of the target training images.

## Highlights

- Preserves the complete frozen multi-head CLIP representation and adds class-specific residual heads.
- Competition-aware head selection: candidate heads are scored by replacement IoU against fixed competing scores, matching the inter-class competition faced at inference.
- Entropy gating fuses shared and class-specific scores, applying the class-specific adjustment mainly in uncertain regions.
- One-round, weight-free offline calibration on 10% of the target training split.
- Includes the audited eight-benchmark runner used for the reported experiments.

## Method

Let $X$ denote frozen CLIP's final visual attention block output and $F_h$ the projected contribution of head $h=0,\ldots,H-1$. The score $\Phi_c(U)$ measures cosine similarity between the frozen readout of $U$ and the class-$c$ text embedding. Attention follows the base method.

### Class-specific assignment

CCAR retains all heads in $X$ and adds residuals before LayerNorm. A shared anchor head $h_a$ serves as a visual anchor, and a mapping $g$ assigns each class its residual head. These offline-selected heads define shared and candidate class-specific scores:

$$
A_c=\Phi_c(X+F_{h_a}),\qquad
Z_{h,c}=\Phi_c(X+F_{h_a}+F_h).
$$

Class $c$ uses $h=g(c)$ for text matching.

### Entropy gating

The shared prediction's uncertainty controls the class-specific contribution. After window aggregation, class probabilities and the normalized entropy at location $p$ are

$$
P_{p,c}=\operatorname{softmax}_c(\tau A_{p,c}),\qquad
u_p=-\frac{\sum_c P_{p,c}\log(P_{p,c}+10^{-6})}{\log C},
$$

where $C$ is the number of foreground text classes and $\tau$ is the base method's logit scale. The final score is

$$
L_{p,c}(g)=A_{p,c}+u_p\bigl(Z_{g(c),p,c}-A_{p,c}\bigr).
$$

The gate $u_p$ is shared across classes: low entropy favors shared scores, while higher entropy increases class-specific weights.

### Fixed competing scores

Head selection is performed offline on a pixel-labeled calibration set, with all network weights frozen. For initialization, each head predicts all classes through $\Phi(X+F_h)$; on the calibration set, the highest-mIoU head becomes $h_a$, and each class's highest-IoU head defines the initial mapping $g_0(c)$.

Reference scores $L(g_0)$ are computed with $h_a$ and $g_0$. Each candidate head $h$ replaces only class $c$'s score, keeping other-class scores fixed. The highest competing score $R_{p,c}$ and candidate score $Q_{h,p,c}$ are

$$
R_{p,c}=\max_{j\ne c}L_{p,j}(g_0),\qquad
Q_{h,p,c}=A_{p,c}+u_p\bigl(Z_{h,p,c}-A_{p,c}\bigr).
$$

The competition margin is $D_{h,p,c}=Q_{h,p,c}-R_{p,c}$; positive margins form class $c$'s candidate mask. For calibration image $i$, let $\Omega_i$ contain valid annotated pixels and $Y_{i,c}\subseteq\Omega_i$ denote class $c$'s ground-truth mask. With image-indexed scores, the replacement mask and selected head are

$$
M_{i,c,h}=\{p\in\Omega_i:Q_{i,h,p,c}>R_{i,p,c}\},\qquad
g_1(c)=\arg\max_h
\frac{\sum_i |M_{i,c,h}\cap Y_{i,c}|}
{\sum_i |M_{i,c,h}\cup Y_{i,c}|}.
$$

The assignments $g_1$ are fixed after one calibration round, and both $h_a$ and $g_1$ are fixed at inference.

## Results

Open-vocabulary segmentation (mIoU, %) with frozen OpenAI CLIP ViT-B/16. Baseline rows use published numbers; `+ CCAR` rows are our confirmed runs, and arrows compare with the corresponding published baseline. ResCLIP uses its NACLIP-based variant. Each reproduced baseline/CCAR pair shares preprocessing and postprocessing; CCAR selects heads on 10% of the target training split and performs no weight updates.

| Method | VOC21 | C60 | Object | VOC20 | C59 | Stuff | City | ADE | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ClearCLIP | 51.8 | 32.6 | 33.0 | 80.9 | 35.9 | 23.9 | 30.0 | 16.7 | 38.1 |
| ClearCLIP + CCAR | 52.5 | 36.3 | 33.0 | 82.3 | 38.6 | 25.8 | 34.6 | 19.0 | **40.2** (↑2.1) |
| SCLIP | 59.1 | 30.4 | 30.5 | 80.4 | 34.2 | 22.4 | 32.2 | 16.1 | 38.2 |
| SCLIP + CCAR | 62.9 | 36.5 | 37.3 | 81.4 | 39.1 | 26.0 | 39.5 | 20.5 | **42.9** (↑4.7) |
| ResCLIP | 61.1 | 33.5 | 35.0 | 86.0 | 36.8 | 24.7 | 35.9 | 18.0 | 41.4 |
| ResCLIP + CCAR | 63.2 | 37.1 | 38.3 | 86.5 | 40.0 | 27.0 | 40.6 | 20.6 | **44.2** (↑2.8) |

In paired reproduction, mean mIoU over the eight settings rises from 38.24 to 40.24 (ClearCLIP), 39.12 to 42.88 (SCLIP), and 41.51 to 44.15 (ResCLIP); ResCLIP + CCAR attains the highest listed mean (44.2).

### Ablations

Full-validation mIoU (%) with ClearCLIP, averaged over calibration seeds 42–44:

| Configuration | C59 | ADE |
|---|---:|---:|
| w/o class-specific assignment | 37.1 | 17.5 |
| w/o entropy gating | 38.1 | 18.8 |
| w/o fixed competing scores | 37.7 | 18.1 |
| Full CCAR | **38.6** | **19.1** |

### Computational cost

ClearCLIP accuracy and cost on an RTX 4080 SUPER in FP16:

| Data | Method | mIoU (%) | Calibration (min) | Inference (ms/image) | Peak memory (GiB) |
|---|---|---:|---:|---:|---:|
| C59 | ClearCLIP | 35.8 | -- | 28.1 | 0.78 |
| C59 | + CCAR | 38.6 | 1.8 | 42.8 | 1.35 |
| ADE | ClearCLIP | 16.6 | -- | 27.5 | 1.68 |
| ADE | + CCAR | 19.2 | 14.8 | 44.4 | 1.70 |

Offline calibration is performed only once; CCAR adds 14.7/16.9 ms per image on C59/ADE with peak-memory increases of 0.57/0.02 GiB.

## Installation

The reference environment uses Python 3.12, PyTorch 2.8, CUDA 12.8, and an NVIDIA GPU. Install a PyTorch build compatible with your CUDA driver, then install the remaining dependencies:

```bash
git clone https://github.com/lizhe333/CCAR.git
cd CCAR
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Download the OpenAI CLIP ViT-B/16 checkpoint:

```bash
mkdir -p checkpoints
curl -L \
  https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt \
  -o checkpoints/ViT-B-16.pt
sha256sum checkpoints/ViT-B-16.pt
```

Expected SHA-256:

```text
5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f
```

The default checkpoint location is `checkpoints/ViT-B-16.pt`. Set `CCAR_CHECKPOINT` to use another location.

## Data

Datasets and model weights are not distributed in this repository. Set the dataset root before running an experiment:

```bash
export CCAR_DATA_ROOT=/path/to/data
export CCAR_CHECKPOINT=/path/to/ViT-B-16.pt
```

The runner supports these dataset keys:

| Key | Dataset | Main paths under `CCAR_DATA_ROOT` |
|---|---|---|
| `voc20`, `voc21` | PASCAL VOC 2012 | `VOCdevkit/VOC2012/{JPEGImages,SegmentationClass,ImageSets/Segmentation}` |
| `context59`, `context60` | PASCAL Context | `VOCdevkit/VOC2010/{JPEGImages,SegmentationClassContext,ImageSets/SegmentationContext}` |
| `ade20k` | ADE20K | `ADEChallengeData2016/{images,annotations}/{training,validation}` |
| `cocostuff` | COCO-Stuff | `coco/{train2017,val2017,stuffthingmaps}` |
| `cocoobject` | COCO-Object | `coco/{train2017,val2017}` and `coco_object/annotations` |
| `cityscapes` | Cityscapes | `cityscapes/{leftImg8bit,gtFine}/{train,val}` |

COCO-Object masks can be prepared from COCO-Stuff raw masks with:

```bash
python scripts/prepare_coco_object.py
```

## Reproduction

Run all commands from the repository root. Start with a read-only data audit:

```bash
python scripts/eight_benchmark.py \
  --phase audit \
  --datasets voc20 context59 ade20k
```

Run a one-image smoke test:

```bash
python scripts/eight_benchmark.py \
  --phase smoke \
  --protocol clearclip448 \
  --methods clearclip \
  --datasets voc20
```

Calibrate on 10% of the training split and evaluate ClearCLIP + CCAR:

```bash
python scripts/eight_benchmark.py \
  --phase run \
  --protocol clearclip448 \
  --calibration-frac 0.1 \
  --seed 42 \
  --methods clearclip \
  --datasets voc21 context60 cocoobject voc20 context59 cocostuff cityscapes ade20k
```

Evaluate the native SCLIP protocol:

```bash
python scripts/eight_benchmark.py \
  --phase run \
  --protocol sclip_official_native \
  --calibration-frac 0.1 \
  --seed 42 \
  --methods sclip \
  --datasets voc21 context60 cocoobject voc20 context59 cocostuff cityscapes ade20k
```

Useful phases are `audit`, `smoke`, `baseline`, `selection`, `eval`, and `run`. The runner writes caches, sampling manifests, selected head tables, checkpoints, and evaluation results below `outputs/`. Interrupted selection and evaluation jobs resume from their saved checkpoints.

## Repository layout

```text
ccar/
  clip_model.py       CLIP loading and exact per-head decomposition
  gacr.py             class-conditioned residual readout and entropy gate
  pipeline.py         dataset, sliding-window, and evaluation protocol
  sclip_native.py     native SCLIP postprocessing contracts
  analysis_utils.py   shared analysis helpers
  open_clip/          vendored ClearCLIP/OpenCLIP model code
  prompts/            ImageNet prompt templates
  cls/                class-name lists for the eight benchmarks
scripts/
  eight_benchmark.py  calibration and evaluation entry point
  11_perturb_cache.py perturbation-cache helper
  16_gacr_combo.py    audited residual-combination reference
  prepare_coco_object.py  COCO-Object mask preparation
repos/                reference configs from ClearCLIP and SCLIP-official
third_party/          bundled third-party license files
```

## Acknowledgements

This implementation builds on [CLIP](https://github.com/openai/CLIP), [OpenCLIP](https://github.com/mlfoundations/open_clip), [ClearCLIP](https://github.com/mc-lan/ClearCLIP), and [SCLIP](https://github.com/wangf3014/SCLIP). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for source provenance and bundled license files.

## License

A project-level license has not yet been selected. Third-party files retain their original licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
