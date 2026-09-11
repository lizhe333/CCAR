# CCAR-CLIP

Official PyTorch implementation of **CCAR: Class-Conditioned Attention Residuals for Frozen CLIP Segmentation**.

CCAR is a training-free plug-in for CLIP-based semantic segmentation: it keeps the complete multi-head representation, adds a shared attention-head residual and a class-specific residual, and applies the class-specific adjustment mainly in uncertain regions. A one-round, competition-aware calibration step selects the head table from a small labeled subset without updating model weights.

## Highlights

- Preserves the complete frozen CLIP representation.
- Assigns residual attention heads by semantic class.
- Selects heads with frozen-rival replacement IoU, matching the inter-class competition used at inference.
- Supports ClearCLIP and native SCLIP attention rules.
- Includes the audited eight-benchmark runner used for the reported experiments.

## Method

For the final visual attention block, CCAR decomposes the output into head contributions

\[
F_h=(S_hV_h)W_h^\top, \qquad X=\sum_h F_h+b.
\]

The frozen readout produces an anchor score and a class-specific score:

\[
A_c=\Phi_c(X+F_{h_a}), \qquad
Z_c=\Phi_c(X+F_{h_a}+F_{g(c)}).
\]

An entropy gate derived from the anchor prediction combines them:

\[
L_c=A_c+u\,(Z_c-A_c), \qquad
u=\frac{H(\operatorname{softmax}(sA))}{\log C}.
\]

Calibration first obtains an initial class-head table, freezes its strongest rival score for each class, and evaluates each candidate head by the pooled IoU of pixels where the candidate beats that rival. All selected heads are installed together in one update.

## Results

Validation mIoU (%) with OpenAI CLIP ViT-B/16. Each baseline/CCAR pair uses the same preprocessing and postprocessing. CCAR selects heads on 10% of the target training split and performs no weight updates.

| Method | VOC21 | C60 | Object | VOC20 | C59 | Stuff | City | ADE | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ClearCLIP | 51.79 | 33.78 | 33.03 | 80.93 | 35.84 | 23.90 | 30.04 | 16.64 | 38.24 |
| ClearCLIP + CCAR | 52.45 | 36.25 | 32.96 | 82.28 | 38.58 | 25.76 | 34.59 | 19.00 | 40.24 |
| SCLIP | 59.62 | 32.59 | 33.52 | 81.54 | 34.14 | 22.77 | 32.34 | 16.45 | 39.12 |
| SCLIP + CCAR | 62.89 | 36.48 | 37.29 | 81.35 | 39.05 | 25.99 | 39.46 | 20.51 | 42.88 |

The exact values are stored in [`results/main_results.csv`](results/main_results.csv).

## Installation

The reference environment uses Python 3.12, PyTorch 2.8, CUDA 12.8, and an NVIDIA GPU. Install a PyTorch build compatible with your CUDA driver, then install the remaining dependencies:

```bash
git clone https://github.com/lizhe333/CCAR-CLIP.git
cd CCAR-CLIP
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
  open_clip/          vendored ClearCLIP/OpenCLIP model code
scripts/
  eight_benchmark.py  calibration and evaluation entry point
  11_perturb_cache.py perturbation-cache helper
  16_gacr_combo.py    audited residual-combination reference
results/
  main_results.csv    reported eight-benchmark results
```

## Acknowledgements

This implementation builds on [CLIP](https://github.com/openai/CLIP), [OpenCLIP](https://github.com/mlfoundations/open_clip), [ClearCLIP](https://github.com/mc-lan/ClearCLIP), and [SCLIP](https://github.com/wangf3014/SCLIP). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for source provenance and bundled license files.

## License

A project-level license has not yet been selected. Third-party files retain their original licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
