"""Protocol-aware CCAR benchmark runner.

The legacy ClearCLIP 448/448/224 path remains available as ``clearclip448``.
The isolated ``lht_dih336`` path freezes 336/224/112, scale 40, fresh caches
and fresh frozen-rival selections. The ``sclip_official_native`` path follows
the official SCLIP per-dataset inference contracts, including query-level
post-processing. Every native artifact is checked by protocol and run hashes;
validation labels are read only after selection is frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccar import gacr  # noqa: E402
from ccar.clip_model import (  # noqa: E402
    _manual_ln_fp32,
    _trunk_last_block,
    build_model,
    build_text_features,
    CKPT_SHA256,
    aggregate_query_logits,
)
from ccar.pipeline import (  # noqa: E402
    DATASETS,
    confusion_from_pred,
    compute_padsize,
    dataset_items,
    iou_from_confusion,
    load_gt,
    miou,
    preprocess,
    slide_crops,
)
from ccar.sclip_native import (  # noqa: E402
    NATIVE_DATASET_CONFIG,
    native_perturbation_confusion,
    official_query_prediction,
)

SPEC16 = importlib.util.spec_from_file_location(
    "ccar_p16_eight", ROOT / "scripts/16_gacr_combo.py"
)
P16 = importlib.util.module_from_spec(SPEC16)
assert SPEC16.loader is not None
SPEC16.loader.exec_module(P16)

SPEC11 = importlib.util.spec_from_file_location(
    "ccar_p11_eight", ROOT / "scripts/11_perturb_cache.py"
)
P11 = importlib.util.module_from_spec(SPEC11)
assert SPEC11.loader is not None
SPEC11.loader.exec_module(P11)

DATASET_ORDER = [
    "voc21",
    "context60",
    "cocoobject",
    "voc20",
    "cityscapes",
    "context59",
    "ade20k",
    "cocostuff",
]
METHODS = {"clearclip": "ClearCLIP", "sclip": "SCLIP"}
BACKGROUND_THRESHOLD = {
    "voc21": 0.5,
    "context60": 0.15,
    "cocoobject": 0.4,
}

DEFAULT_PROTOCOL = "clearclip448"
DEFAULT_SEED = 42
DEFAULT_CALIBRATION_FRAC = 0.1
GPU_GUARD_OVERRIDE_MIB: int | None = None


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    short_side: int
    crop: int
    stride: int
    logit_scale: float
    output_dir: str
    dataset_slide_overrides: bool
    allow_legacy_cache_reuse: bool
    allow_confirm_v3_reuse: bool
    official_sclip_postprocess: bool = False


PROTOCOLS = {
    "clearclip448": ProtocolSpec(
        name="clearclip448", short_side=448, crop=448, stride=224,
        logit_scale=50.0, output_dir="eight_benchmark_v1",
        dataset_slide_overrides=True, allow_legacy_cache_reuse=True,
        allow_confirm_v3_reuse=True,
    ),
    "lht_dih336": ProtocolSpec(
        name="lht_dih336", short_side=336, crop=224, stride=112,
        logit_scale=40.0, output_dir="eight_benchmark_336_v1",
        dataset_slide_overrides=False, allow_legacy_cache_reuse=False,
        allow_confirm_v3_reuse=False,
    ),
    "sclip_native560": ProtocolSpec(
        name="sclip_native560", short_side=560, crop=224, stride=112,
        logit_scale=40.0, output_dir="cityscapes_sclip_native560_v1",
        dataset_slide_overrides=False, allow_legacy_cache_reuse=False,
        allow_confirm_v3_reuse=False,
    ),
    "sclip_official_native": ProtocolSpec(
        name="sclip_official_native", short_side=336, crop=224, stride=112,
        logit_scale=40.0, output_dir="sclip_official_native_ccar_v1",
        dataset_slide_overrides=False, allow_legacy_cache_reuse=False,
        allow_confirm_v3_reuse=False, official_sclip_postprocess=True,
    ),
}

# Backward-compatible module default for callers that import this runner.
OUT = ROOT / "outputs" / PROTOCOLS[DEFAULT_PROTOCOL].output_dir


class ProtocolMismatchError(RuntimeError):
    """An artifact belongs to a different or unidentified protocol."""


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def json_dump(path: Path, obj) -> None:
    """Atomically publish JSON so interrupted jobs never leave valid-looking files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def npz_dump(path: Path, **arrays) -> None:
    """Atomically publish a compressed NumPy artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


IMPLEMENTATION_FILES = (
    "ccar/clip_model.py",
    "ccar/pipeline.py",
    "ccar/gacr.py",
    "scripts/11_perturb_cache.py",
    "scripts/16_gacr_combo.py",
)
IMPLEMENTATION_SHA256 = {
    rel: sha256_file(ROOT / rel) for rel in IMPLEMENTATION_FILES
}


def source_method_key(method_type: str) -> str:
    return "clearclip" if method_type == "ClearCLIP" else "sclip"


SCLIP_CLASS_KEYS = {
    "voc21": "voc21",
    "context60": "context60",
    "cocoobject": "coco_object",
    "voc20": "voc20",
    "context59": "context59",
    "ade20k": "ade20k",
    "cocostuff": "coco_stuff",
}


def class_file_for(ds: str, method_type: str) -> Path:
    if method_type == "SCLIP":
        key = SCLIP_CLASS_KEYS.get(ds)
        candidate = ROOT / "repos/SCLIP-official/configs" / f"cls_{key}.txt"
        if key is not None and candidate.exists():
            return candidate
    return Path(DATASETS[ds]["cls_file"])


def class_max_logits(logits, query_idx, num_classes):
    """Vectorized synonym max-pooling, equivalent to aggregate_query_logits."""
    if query_idx is None or logits.shape[-2] == num_classes:
        return logits
    idx = torch.as_tensor(query_idx, device=logits.device, dtype=torch.long)
    shape = [1] * logits.ndim
    shape[-2] = idx.numel()
    shape[-1] = 1
    scatter_idx = idx.reshape(shape).expand_as(logits)
    out_shape = list(logits.shape)
    out_shape[-2] = num_classes
    out = torch.full(
        out_shape, -float("inf"), device=logits.device, dtype=logits.dtype
    )
    return out.scatter_reduce_(
        -2, scatter_idx, logits, reduce="amax", include_self=True
    )


NATIVE_IMPLEMENTATION_FILES = (
    "scripts/eight_benchmark.py",
    "ccar/sclip_native.py",
)
NATIVE_IMPLEMENTATION_SHA256 = {
    rel: sha256_file(ROOT / rel) for rel in NATIVE_IMPLEMENTATION_FILES
}


def protocol_is_native_sclip(protocol: ProtocolSpec) -> bool:
    return protocol.official_sclip_postprocess


def protocol_settings(protocol: ProtocolSpec, ds: str) -> dict:
    if protocol_is_native_sclip(protocol):
        if ds not in NATIVE_DATASET_CONFIG:
            raise KeyError(f"official SCLIP native settings missing for {ds}")
        return dict(NATIVE_DATASET_CONFIG[ds])
    return {
        "short_side": protocol.short_side,
        "crop": slide_spec(ds, protocol)[0],
        "stride": slide_spec(ds, protocol)[1],
        "logit_scale": protocol.logit_scale,
        "prob_thd": float(BACKGROUND_THRESHOLD.get(ds, 0.0)),
        "area_thd": None,
    }


def protocol_short_side(protocol: ProtocolSpec, ds: str) -> int:
    return int(protocol_settings(protocol, ds)["short_side"])


def protocol_logit_scale(protocol: ProtocolSpec, ds: str) -> float:
    return float(protocol_settings(protocol, ds)["logit_scale"])


def protocol_prob_thd(protocol: ProtocolSpec, ds: str) -> float:
    return float(protocol_settings(protocol, ds)["prob_thd"])


def protocol_area_thd(protocol: ProtocolSpec, ds: str) -> float | None:
    return protocol_settings(protocol, ds)["area_thd"]


def gpu_free_mib() -> int | None:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(p.stdout.strip().splitlines()[0])
    except Exception:
        return None


def gpu_guard(min_free_mib: int = 3072) -> None:
    """Allow shared GPUs; only defer when the projected job may OOM."""
    if GPU_GUARD_OVERRIDE_MIB is not None:
        min_free_mib = GPU_GUARD_OVERRIDE_MIB
    if min_free_mib <= 0:
        return
    free = gpu_free_mib()
    if free is not None and free < min_free_mib:
        raise RuntimeError(
            f"GPU free memory is {free} MiB, below the {min_free_mib} MiB "
            "safety margin; do not start this job yet."
        )


def slide_spec(
    ds: str,
    protocol: ProtocolSpec = PROTOCOLS[DEFAULT_PROTOCOL],
) -> tuple[int, int]:
    if not protocol.dataset_slide_overrides:
        return protocol.crop, protocol.stride
    cfg = DATASETS[ds]
    return (
        int(cfg.get("slide_crop", protocol.crop)),
        int(cfg.get("slide_stride", protocol.stride)),
    )


def default_output_root(protocol: ProtocolSpec) -> Path:
    return ROOT / "outputs" / protocol.output_dir


def protocol_record(
    protocol: ProtocolSpec, ds: str, method_type: str = "ClearCLIP"
) -> dict:
    """Canonical, dataset-specific inference contract used for artifact checks."""
    cfg = DATASETS[ds]
    settings = protocol_settings(protocol, ds)
    crop, stride = int(settings["crop"]), int(settings["stride"])
    cls_file = class_file_for(ds, method_type)
    label_map = cfg.get("label_map")
    record = {
        "name": protocol.name,
        "model": "OpenAI CLIP ViT-B/16",
        "base_method": method_type,
        "sclip_official_commit": (
            "3608360267b6130c1ef18090d7289f17c771cb90"
            if method_type == "SCLIP" else None
        ),
        "checkpoint_sha256": CKPT_SHA256,
        "implementation_sha256": dict(IMPLEMENTATION_SHA256),
        "short_side": int(settings["short_side"]),
        "max_long_side": 2048,
        "input_interpolation": "opencv.INTER_LINEAR",
        "crop": crop,
        "stride": stride,
        "patch": 16,
        "logit_scale": float(settings["logit_scale"]),
        "output_resolution": "original",
        "output_interpolation": "torch.bilinear",
        "align_corners": False,
        "text_templates": "OpenAI 80-template ImageNet ensemble",
        "vocabulary_file": cls_file.name,
        "vocabulary_sha256": sha256_file(cls_file),
        "num_classes": int(cfg["num_classes"]),
        "background_prob_threshold": (
            0.0 if protocol_is_native_sclip(protocol)
            else float(BACKGROUND_THRESHOLD.get(ds, 0.0))
        ),
        "reduce_zero_label": bool(cfg["reduce_zero_label"]),
        "label_map_sha256": sha256_json(label_map) if label_map is not None else None,
        "ignore_index": 255,
        "pamr": False,
        "densecrf": False,
    }
    if protocol_is_native_sclip(protocol):
        record.update({
            "official_sclip_postprocess": True,
            "probability_threshold": protocol_prob_thd(protocol, ds),
            "area_threshold": protocol_area_thd(protocol, ds),
            "synonym_aggregation": "query_softmax_then_class_max",
            "native_implementation_sha256": dict(NATIVE_IMPLEMENTATION_SHA256),
        })
    return record


def protocol_sha256(
    protocol: ProtocolSpec, ds: str, method_type: str = "ClearCLIP"
) -> str:
    return sha256_json(protocol_record(protocol, ds, method_type))


def run_sha256(
    protocol_hash: str,
    seed: int,
    calibration_frac: float,
    method_type: str,
) -> str:
    return sha256_json({
        "protocol_sha256": protocol_hash,
        "method_type": method_type,
        "seed": int(seed),
        "calibration_frac": float(calibration_frac),
    })


def _frac_tag(frac: float) -> str:
    return format(float(frac), ".12g").replace("-", "m").replace(".", "p")


def selection_manifest_path(
    out: Path,
    method_type: str,
    ds: str,
    seed: int,
    calibration_frac: float,
) -> Path:
    stem = f"{source_method_key(method_type)}_{ds}_seed{seed}"
    if not np.isclose(calibration_frac, DEFAULT_CALIBRATION_FRAC):
        stem += f"_cal{_frac_tag(calibration_frac)}"
    return out / "manifests" / "selection" / f"{stem}.json"


def result_run_suffix(seed: int, calibration_frac: float) -> str:
    if seed == DEFAULT_SEED and np.isclose(
        calibration_frac, DEFAULT_CALIBRATION_FRAC
    ):
        return ""
    return f"_seed{seed}_cal{_frac_tag(calibration_frac)}"


def _semantic_protocol(record: dict) -> dict:
    """Protocol identity excluding source-file hashes used only for provenance."""
    clean = dict(record)
    clean.pop("implementation_sha256", None)
    return clean


def _artifact_semantically_current(artifact: dict) -> bool:
    record = artifact.get("protocol")
    ds = artifact.get("dataset")
    method_type = artifact.get("method_type")
    if not isinstance(record, dict) or ds not in DATASETS or method_type is None:
        return False
    name = record.get("name")
    if name not in PROTOCOLS:
        return False
    current = protocol_record(PROTOCOLS[name], ds, method_type)
    return _semantic_protocol(record) == _semantic_protocol(current)


def require_artifact_protocol(
    artifact: dict,
    expected_protocol_sha256: str,
    expected_run_sha256: str | None,
    *,
    kind: str,
    require_hash: bool,
) -> None:
    actual_protocol = artifact.get("protocol_sha256")
    if actual_protocol is None:
        if require_hash:
            raise ProtocolMismatchError(
                f"{kind} has no protocol_sha256; refusing unidentified/448 reuse"
            )
        return
    semantic_compat = False
    if actual_protocol != expected_protocol_sha256:
        semantic_compat = _artifact_semantically_current(artifact)
        if not semantic_compat:
            raise ProtocolMismatchError(
                f"{kind} protocol mismatch: expected {expected_protocol_sha256}, "
                f"found {actual_protocol}"
            )
    if expected_run_sha256 is not None:
        actual_run = artifact.get("run_sha256")
        if actual_run is None and require_hash:
            raise ProtocolMismatchError(
                f"{kind} has no run_sha256; refusing table reuse"
            )
        if actual_run is not None and actual_run != expected_run_sha256:
            compatible_run = False
            if semantic_compat:
                try:
                    compatible_run = expected_run_sha256 == run_sha256(
                        expected_protocol_sha256,
                        int(artifact["seed"]),
                        float(artifact["calibration_frac"]),
                        artifact["method_type"],
                    )
                except (KeyError, TypeError, ValueError):
                    compatible_run = False
            if not compatible_run:
                raise ProtocolMismatchError(
                    f"{kind} run mismatch: expected {expected_run_sha256}, "
                    f"found {actual_run}"
                )


def train_items(ds: str) -> list[tuple[Path, Path]]:
    cfg = DATASETS[ds]
    if ds in {"voc20", "voc21"}:
        split = Path(cfg["train_split"])
        ids = [x.strip() for x in split.read_text().splitlines() if x.strip()]
        return [(cfg["img_dir"] / (x + ".jpg"), cfg["gt_dir"] / (x + ".png")) for x in ids]
    if ds in {"context59", "context60"}:
        split = Path(cfg["train_split"])
        ids = [x.strip() for x in split.read_text().splitlines() if x.strip()]
        return [(cfg["img_dir"] / (x + ".jpg"), cfg["gt_dir"] / (x + ".png")) for x in ids]
    if ds == "ade20k":
        img_dir = Path(cfg["train_img_dir"])
        gt_dir = Path(cfg["train_gt_dir"])
        return sorted((p, gt_dir / f"{p.stem}.png") for p in img_dir.glob("*.jpg"))

    if cfg.get("train_split") is not None:
        ids = [
            line.strip()
            for line in Path(cfg["train_split"]).read_text().splitlines()
            if line.strip()
        ]
        img_dir = cfg.get("train_img_dir", cfg["img_dir"])
        gt_dir = cfg.get("train_gt_dir", cfg["gt_dir"])
        gt_suffix = cfg.get("gt_suffix", ".png")
        return [
            (img_dir / (item + cfg["img_suffix"]),
             gt_dir / (item + gt_suffix))
            for item in ids
        ]

    if cfg.get("train_img_dir") is None:
        raise KeyError(f"no training split configured for {ds}")

    gt_suffix = cfg.get("gt_suffix", ".png")
    globber = cfg["train_img_dir"].rglob if cfg.get("recursive", False) else cfg["train_img_dir"].glob
    items = []
    for image in globber("*" + cfg["img_suffix"]):
        rel = image.relative_to(cfg["train_img_dir"])
        gt_name = rel.name[:-len(cfg["img_suffix"])] + gt_suffix
        items.append((image, cfg["train_gt_dir"] / rel.parent / gt_name))
    return sorted(items)


def calibration_items(
    ds: str,
    seed: int = DEFAULT_SEED,
    frac: float = DEFAULT_CALIBRATION_FRAC,
):
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"calibration fraction must be in (0, 1], got {frac}")
    full = train_items(ds)
    if not full:
        return full, [], {"n_train": 0, "n_calibration": 0, "seed": seed, "frac": frac}
    missing = [(str(i), str(g)) for i, g in full if not i.exists() or not g.exists()]
    if missing:
        raise FileNotFoundError(f"{ds}: {len(missing)} missing training pairs, e.g. {missing[:2]}")
    rng = np.random.default_rng(seed)
    n = max(1, int(round(frac * len(full))))
    indices = np.sort(rng.choice(len(full), size=n, replace=False))
    selected = [full[int(i)] for i in indices]
    return full, selected, {
        "n_train": len(full),
        "n_calibration": len(selected),
        "seed": seed,
        "frac": frac,
        "sample_rule": "numpy.default_rng(seed).choice(replace=False), sorted indices",
        "all_train_stems_sha256": sha256_bytes(
            "\n".join(p.stem for p, _ in full).encode()
        ),
        "calibration_stems_sha256": sha256_bytes(
            "\n".join(p.stem for p, _ in selected).encode()
        ),
        "calibration_stems": [p.stem for p, _ in selected],
    }


def _cache_candidates(
    ds: str,
    method_type: str,
    protocol: ProtocolSpec = PROTOCOLS[DEFAULT_PROTOCOL],
    out: Path = OUT,
) -> list[Path]:
    suite_cache = out / "caches" / source_method_key(method_type) / ds / "train"
    if not protocol.allow_legacy_cache_reuse:
        return [suite_cache]
    if method_type == "ClearCLIP":
        candidates = [
            ROOT / f"outputs/cache_pert_train_{ds}",
            out / "caches" / "clearclip" / ds / "train",
        ]
    else:
        candidates = [
            ROOT / f"outputs/cache_pert_sclip_train_{ds}",
            out / "caches" / "sclip" / ds / "train",
        ]
    if suite_cache not in candidates:
        candidates.append(suite_cache)
    return candidates


def _cache_file_map(
    path: Path,
    *,
    expected_protocol_sha256: str | None = None,
    require_hash: bool = False,
) -> dict[str, Path]:
    if not path.is_dir():
        return {}
    files = {p.stem: p for p in path.glob("*.npz")}
    if not require_hash:
        return files
    for stem, cache_file in sorted(files.items()):
        with np.load(cache_file, allow_pickle=False) as data:
            actual = (
                str(data["protocol_sha256"].item())
                if "protocol_sha256" in data.files
                else None
            )
        if actual != expected_protocol_sha256:
            found = actual if actual is not None else "missing"
            raise ProtocolMismatchError(
                f"cache {cache_file} protocol mismatch: expected "
                f"{expected_protocol_sha256}, found {found}; refusing 448 reuse"
            )
    return files


def ensure_perturb_cache(
    ds: str,
    method_type: str,
    selected: list[tuple[Path, Path]],
    *,
    force: bool = False,
    protocol: ProtocolSpec = PROTOCOLS[DEFAULT_PROTOCOL],
    out: Path = OUT,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, dict]:
    """Reuse only per-image perturbations with compatible inference identity."""
    expected = {p.stem for p, _ in selected}
    p_hash = protocol_sha256(protocol, ds, method_type)
    strict_hash = not protocol.allow_legacy_cache_reuse
    source = None
    source_map = {}
    for candidate in _cache_candidates(ds, method_type, protocol, out):
        cmap = _cache_file_map(
            candidate,
            expected_protocol_sha256=p_hash,
            require_hash=strict_hash,
        )
        if expected.issubset(cmap):
            source = candidate
            source_map = cmap
            break

    if source is None or (
        force
        and protocol.allow_legacy_cache_reuse
        and str(source).startswith(str(ROOT / "outputs/cache_pert"))
    ):
        source = out / "caches" / source_method_key(method_type) / ds / "train"
        source.mkdir(parents=True, exist_ok=True)
        source_map = _cache_file_map(
            source,
            expected_protocol_sha256=p_hash,
            require_hash=strict_hash,
        )

    missing_items = [(p, g) for p, g in selected if p.stem not in source_map]
    if force:
        missing_items = selected

    if missing_items:
        gpu_guard(12288)
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        np.random.seed(seed)
        P11.CHUNK = 25
        model = build_model(precision="fp16", device="cuda", vit_type="ViT-B/16")
        tf, query_idx = build_text_features(
            model, str(class_file_for(ds, method_type)), device="cuda"
        )
        settings = protocol_settings(protocol, ds)
        crop, stride = int(settings["crop"]), int(settings["stride"])
        t0 = time.perf_counter()
        peak = 0.0
        for n, (image_path, gt_path) in enumerate(missing_items, 1):
            x, ori_hw = preprocess(
                image_path, short_side=int(settings["short_side"])
            )
            gt = load_gt(
                gt_path,
                DATASETS[ds]["reduce_zero_label"],
                label_map=DATASETS[ds].get("label_map"),
            )
            torch.cuda.reset_peak_memory_stats()
            if protocol_is_native_sclip(protocol):
                conf = native_perturbation_confusion(
                    model,
                    tf.float(),
                    x.to("cuda"),
                    ori_hw,
                    gt,
                    DATASETS[ds]["num_classes"],
                    model_type=method_type,
                    crop=crop,
                    stride=stride,
                    logit_scale=float(settings["logit_scale"]),
                    prob_thd=float(settings["prob_thd"]),
                    area_thd=settings["area_thd"],
                    query_idx=query_idx,
                )
            else:
                P11.aggregate_query_logits = class_max_logits
                conf = P11.image_confusion(
                    model,
                    tf.float(),
                    x.to("cuda"),
                    ori_hw,
                    gt,
                    DATASETS[ds]["num_classes"],
                    model_type=method_type,
                    crop=crop,
                    stride=stride,
                    logit_scale=float(settings["logit_scale"]),
                    query_idx=query_idx,
                )
            cache_path = source / f"{image_path.stem}.npz"
            npz_dump(
                cache_path,
                conf=conf,
                protocol_sha256=np.array(p_hash),
                method_type=np.array(method_type),
            )
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**20)
            if n == 1 or n % 100 == 0 or n == len(missing_items):
                print(
                    f"[cache {method_type} {ds}] {n}/{len(missing_items)} "
                    f"peak={peak:.0f}MiB",
                    flush=True,
                )
        del model, tf
        torch.cuda.empty_cache()
        source_map = _cache_file_map(
            source,
            expected_protocol_sha256=p_hash,
            require_hash=strict_hash,
        )
        json_dump(
            source.parent / "cache_meta.json",
            {
                "dataset": ds,
                "method_type": method_type,
                "n_new": len(missing_items),
                "protocol": protocol_record(protocol, ds, method_type),
                "protocol_sha256": p_hash,
                "crop": crop,
                "stride": stride,
                "short_side": int(settings["short_side"]),
                "logit_scale": float(settings["logit_scale"]),
                "prob_thd": float(settings["prob_thd"]),
                "area_thd": settings["area_thd"],
                "seed": seed,
                "calibration_stems_sha256": sha256_bytes(
                    "\n".join(sorted(expected)).encode()
                ),
                "seconds": time.perf_counter() - t0,
                "peak_vram_mib": peak,
            },
        )

    missing_after = sorted(expected - set(source_map))
    if missing_after:
        raise RuntimeError(f"{ds} {method_type}: cache still missing {missing_after[:3]}")
    record = {
        "cache_dir": str(source),
        "protocol": protocol_record(protocol, ds, method_type),
        "protocol_sha256": p_hash,
        "n_calibration_files": len(expected),
        "cache_files_sha256": sha256_bytes(
            "\n".join(
                sha256_file(source_map[s])
                for s in sorted(expected)
            ).encode()
        ),
        "reuse": str(source).startswith(str(ROOT / "outputs/cache_pert")),
        "files": {s: str(source_map[s]) for s in sorted(expected)},
    }
    return source, record


def aggregate_A(cache_dir: Path, selected: list[tuple[Path, Path]], C: int):
    files = _cache_file_map(cache_dir)
    conf = np.zeros((25, C, C), dtype=np.int64)
    for image_path, _ in selected:
        with np.load(files[image_path.stem]) as data:
            conf += data["conf"].astype(np.int64)
    inter = np.einsum("vcc->vc", conf).astype(np.float64)
    union = conf.sum(2) + conf.sum(1) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.divide(inter, union, out=np.full_like(inter, np.nan), where=union > 0)
    v_miou = np.nanmean(iou, axis=1)
    h_anchor = int(np.nanargmax(v_miou[13:25]))
    h_class = np.argmax(np.nan_to_num(iou[13:25], nan=-1.0), axis=0).astype(np.int64)
    support = conf[0].sum(axis=1) > 0
    h_class[~support] = h_anchor
    return h_anchor, h_class, conf, v_miou, int((~support).sum())


def place(field, canvas, y1, x1, y2, x2, crop, pad):
    C, N = field.shape
    gh, gw = crop.shape[-2] // 16, crop.shape[-1] // 16
    up = F.interpolate(
        field.reshape(1, C, gh, gw),
        size=crop.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0]
    left, right, top, bottom = pad
    canvas[:, y1:y2, x1:x2] += up[:, top:top + (y2 - y1), left:left + (x2 - x1)]


@torch.no_grad()
def all_head_canvases(
    model,
    tf32,
    image_path: Path,
    h_anchor: int,
    model_type: str,
    crop_size: int,
    stride: int,
    short_side: int = 448,
    query_idx=None,
    num_classes: int | None = None,
):
    x, ori_hw = preprocess(image_path, short_side=short_side)
    x = x.to("cuda")
    _, _, height, width = x.shape
    C = int(num_classes if num_classes is not None else tf32.shape[0])
    fields = {"A": torch.zeros((C, height, width), device="cuda", dtype=torch.float32)}
    fields.update({
        f"Z{h}": torch.zeros((C, height, width), device="cuda", dtype=torch.float32)
        for h in range(12)
    })
    count = torch.zeros((1, height, width), device="cuda", dtype=torch.float32)
    for y1, x1, y2, x2 in slide_crops(height, width, crop_size, stride):
        crop_img = x[:, :, y1:y2, x1:x2]
        hr, wr = crop_img.shape[-2:]
        pad = compute_padsize(hr, wr, 16)
        if any(pad):
            crop_img = F.pad(crop_img, pad)
        if model_type == "SCLIP":
            hp, attn, trunk_x = _trunk_last_block(
                model, crop_img.half(), model_type=model_type,
                return_trunk_input=True,
            )
        else:
            hp, attn = _trunk_last_block(
                model, crop_img.half(), model_type=model_type
            )
            trunk_x = None
        h32 = hp.float()
        full = h32.sum(0) + attn.out_proj.bias.float().unsqueeze(0)
        af = full + h32[h_anchor]
        variants = [af] + [af + h32[h] for h in range(12)]
        if model_type == "SCLIP":
            blk = model.visual.transformer.resblocks[-1]
            trunk = trunk_x.squeeze(1)
            completed = []
            for variant in variants:
                y = trunk + variant.to(trunk.dtype)
                y = y.unsqueeze(1)
                y = y + blk.mlp(blk.ln_2(y))
                completed.append(y.squeeze(1).float())
            stacked = torch.stack(completed, dim=0)
        else:
            stacked = torch.stack(variants, dim=0)
        tokens = _manual_ln_fp32(stacked, model.visual.ln_post)[:, 1:]
        tokens = tokens @ model.visual.proj.float()
        tokens = tokens / tokens.norm(dim=-1, keepdim=True)
        logits = torch.einsum("hnd,qd->hqn", tokens, tf32)
        logits = class_max_logits(logits, query_idx, C)
        common = {"A": logits[0]}
        common.update({f"Z{h}": logits[h + 1] for h in range(12)})
        for key, field in common.items():
            place(field, fields[key], y1, x1, y2, x2, crop_img, pad)
        count[:, y1:y2, x1:x2] += 1
    if torch.any(count == 0):
        raise AssertionError(f"uncovered pixels for {image_path}")
    for key in fields:
        fields[key] /= count
    return fields, ori_hw


def top_two(
    field: torch.Tensor,
    ori_hw: tuple[int, int],
    logit_scale: float = 50.0,
):
    C = field.shape[0]
    best = torch.full(ori_hw, -float("inf"), device=field.device)
    second = torch.full_like(best, -float("inf"))
    pred = torch.zeros(ori_hw, dtype=torch.long, device=field.device)
    for start in range(0, C, 10):
        lg = F.interpolate(
            field[start:start + 10].unsqueeze(0),
            size=ori_hw,
            mode="bilinear",
            align_corners=False,
        )[0] * logit_scale
        vals, idx = lg.max(0)
        candidates = torch.topk(
            lg, min(2, lg.shape[0]), dim=0
        ).values
        candidate_second = (
            candidates[1] if candidates.shape[0] > 1
            else torch.full_like(candidates[0], -float("inf"))
        )
        second = torch.topk(
            torch.stack((best, second, vals, candidate_second)), 2, dim=0
        ).values[1]
        changed = vals > best
        pred[changed] = idx[changed] + start
        best = torch.maximum(best, vals)
    return best, second, pred



def _selection_checkpoint_path(path: Path) -> Path:
    return path.with_name(path.stem + "_checkpoint.npz")


def _assert_selection_uses_train_only(
    ds: str,
    full: list[tuple[Path, Path]],
    selected: list[tuple[Path, Path]],
) -> dict:
    validation = dataset_items(ds)
    train_images = {str(p.resolve()) for p, _ in full}
    selected_images = {str(p.resolve()) for p, _ in selected}
    validation_images = {str(p.resolve()) for p, _ in validation}
    if not selected_images.issubset(train_images):
        raise AssertionError(f"{ds}: calibration list contains a non-training image")
    exact_overlap = selected_images & validation_images
    stem_overlap = {p.stem for p, _ in selected} & {p.stem for p, _ in validation}
    if exact_overlap or stem_overlap:
        raise AssertionError(
            f"{ds}: calibration/validation overlap: "
            f"{len(exact_overlap)} paths, {len(stem_overlap)} stems"
        )
    return {
        "train_validation_exact_path_overlap": 0,
        "calibration_validation_stem_overlap": 0,
        "validation_count": len(validation),
        "validation_stems_sha256": sha256_bytes(
            "\n".join(p.stem for p, _ in validation).encode()
        ),
    }


def _write_sampling_manifest(
    out: Path,
    ds: str,
    seed: int,
    calibration_frac: float,
    sampling: dict,
    split_audit: dict,
) -> Path:
    path = (
        out / "manifests" / "sampling"
        / f"{ds}_seed{seed}_cal{_frac_tag(calibration_frac)}.json"
    )
    record = {
        "dataset": ds,
        "seed": seed,
        "calibration_frac": calibration_frac,
        "sampling": sampling,
        "split_audit": split_audit,
    }
    if path.exists():
        old = json.loads(path.read_text())
        if old != record:
            raise RuntimeError(f"sampling manifest changed unexpectedly: {path}")
    else:
        json_dump(path, record)
    return path


def select_competition(
    ds: str,
    method_type: str,
    selected: list[tuple[Path, Path]],
    h_anchor: int,
    h_class_A: np.ndarray,
    *,
    force: bool = False,
    protocol: ProtocolSpec = PROTOCOLS[DEFAULT_PROTOCOL],
    out: Path = OUT,
    seed: int = DEFAULT_SEED,
    calibration_frac: float = DEFAULT_CALIBRATION_FRAC,
):
    C = DATASETS[ds]["num_classes"]
    crop, stride = slide_spec(ds, protocol)
    p_hash = protocol_sha256(protocol, ds, method_type)
    r_hash = run_sha256(p_hash, seed, calibration_frac, method_type)
    path = selection_manifest_path(out, method_type, ds, seed, calibration_frac)

    if path.exists() and not force:
        existing = json.loads(path.read_text())
        require_artifact_protocol(
            existing, p_hash, r_hash, kind=f"selection {path}",
            require_hash=not protocol.allow_legacy_cache_reuse,
        )
        if existing.get("h_class_C") is not None:
            return existing

    # Legacy confirmation tables are valid only for the original frozen 448 run.
    reuse = (
        protocol.allow_confirm_v3_reuse
        and method_type == "ClearCLIP"
        and ds in {"context59", "ade20k"}
        and seed == DEFAULT_SEED
        and np.isclose(calibration_frac, DEFAULT_CALIBRATION_FRAC)
        and not force
        and (ROOT / f"innovation_night/confirm_v3/results/sel3_{ds}_s42.json").exists()
    )
    if reuse:
        src = ROOT / f"innovation_night/confirm_v3/results/sel3_{ds}_s42.json"
        data = json.loads(src.read_text())
        result = {
            "dataset": ds,
            "method_type": method_type,
            "selection": "current frozen-rival C table reused from confirm_v3",
            "h_anchor": int(data["h_anchor"]),
            "h_class_A": data["h_class_A_arm"],
            "h_class_C": data["h_class_C"],
            "n_calibration": int(data["n_cal"]),
            "calibration_source": "confirm_v3/sel3 seed42; exact current table",
            "table_sha256": data["table_sha256"]["C"],
            "crop": crop,
            "stride": stride,
            "weights_updated": False,
            "validation_labels_read_before_eval": False,
            "source_path": str(src),
            "protocol": protocol_record(protocol, ds, method_type),
            "protocol_sha256": p_hash,
            "run_sha256": r_hash,
            "seed": seed,
            "calibration_frac": calibration_frac,
        }
        json_dump(path, result)
        return result

    h_class_A = np.asarray(h_class_A, dtype=np.int64)
    gpu_guard(12288)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(precision="fp16", device="cuda", vit_type="ViT-B/16")
    tf, query_idx = build_text_features(
        model, str(class_file_for(ds, method_type)), device="cuda"
    )
    tf32 = tf.float()
    stats = np.zeros((12, 3, C), dtype=np.int64)
    ties = 0
    support = np.zeros(C, dtype=bool)
    start_index = 0
    checkpoint = _selection_checkpoint_path(path)
    selected_sha = sha256_bytes("\n".join(p.stem for p, _ in selected).encode())
    if checkpoint.exists() and not force:
        with np.load(checkpoint, allow_pickle=False) as data:
            cp_protocol = str(data["protocol_sha256"].item())
            cp_run = str(data["run_sha256"].item())
            cp_selected = str(data["calibration_stems_sha256"].item())
            if (cp_protocol, cp_run, cp_selected) != (p_hash, r_hash, selected_sha):
                raise ProtocolMismatchError(f"selection checkpoint mismatch: {checkpoint}")
            stats = data["stats"].astype(np.int64)
            support = data["support"].astype(bool)
            ties = int(data["ties"].item())
            start_index = int(data["next_index"].item())
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def save_checkpoint(next_index: int) -> None:
        npz_dump(
            checkpoint,
            protocol_sha256=np.array(p_hash),
            run_sha256=np.array(r_hash),
            calibration_stems_sha256=np.array(selected_sha),
            next_index=np.array(next_index, dtype=np.int64),
            stats=stats,
            support=support,
            ties=np.array(ties, dtype=np.int64),
        )

    try:
        for n0 in range(start_index, len(selected)):
            image_path, gt_path = selected[n0]
            fields, ori_hw = all_head_canvases(
                model, tf32, image_path, h_anchor, method_type, crop, stride,
                short_side=protocol_short_side(protocol, ds),
                query_idx=query_idx,
                num_classes=C,
            )
            A = fields["A"]
            gate = gacr.uncertainty_gate(
                A, float(np.log(C)), logit_scale=protocol_logit_scale(protocol, ds)
            )
            old = A + gate * (
                torch.stack([
                    fields[f"Z{int(h_class_A[c])}"][c] for c in range(C)
                ]) - A
            )
            best, second, oldpred = top_two(
                old, ori_hw, logit_scale=protocol_logit_scale(protocol, ds)
            )
            gt = load_gt(
                gt_path,
                DATASETS[ds]["reduce_zero_label"],
                label_map=DATASETS[ds].get("label_map"),
            )
            gt_t = torch.as_tensor(gt, device="cuda", dtype=torch.long)
            valid = gt_t != 255
            support |= np.bincount(gt[gt != 255], minlength=C).astype(bool)
            for h in range(12):
                score = A + gate * (fields[f"Z{h}"] - A)
                for c0 in range(0, C, 10):
                    c1 = min(C, c0 + 10)
                    ar = torch.arange(c0, c1, device="cuda")[:, None, None]
                    sh = F.interpolate(
                        score[c0:c1].unsqueeze(0),
                        size=ori_hw,
                        mode="bilinear",
                        align_corners=False,
                    )[0] * protocol_logit_scale(protocol, ds)
                    rival = torch.where(
                        oldpred[None] == ar, second[None], best[None]
                    )
                    beat = (sh > rival) & valid[None]
                    gtc = gt_t[None] == ar
                    stats[h, 0, c0:c1] += (
                        (beat & gtc).sum((1, 2)).cpu().numpy()
                    )
                    stats[h, 1, c0:c1] += (
                        (beat & ~gtc).sum((1, 2)).cpu().numpy()
                    )
                    stats[h, 2, c0:c1] += (
                        ((~beat) & gtc).sum((1, 2)).cpu().numpy()
                    )
                    ties += int(((sh == rival) & valid[None]).sum().item())
                del score
            del fields, A, old, gate, best, second, oldpred
            n = n0 + 1
            if n == 1 or n % 50 == 0 or n == len(selected):
                save_checkpoint(n)
            if n == 1 or n % 100 == 0 or n == len(selected):
                print(f"[select {method_type} {ds}] {n}/{len(selected)}", flush=True)
    except Exception:
        save_checkpoint(max(start_index, n0))
        raise

    union = stats.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        class_iou = np.divide(
            stats[:, 0],
            union,
            out=np.full((12, C), np.nan, dtype=np.float64),
            where=union > 0,
        )
    h_class_C = np.argmax(
        np.nan_to_num(class_iou, nan=-1.0), axis=0
    ).astype(np.int64)
    h_class_C[~support] = h_anchor
    result = {
        "dataset": ds,
        "method_type": method_type,
        "selection": "one-round frozen-rival replacement IoU",
        "h_anchor": int(h_anchor),
        "h_class_A": h_class_A.tolist(),
        "h_class_C": h_class_C.tolist(),
        "n_calibration": len(selected),
        "calibration_stems_sha256": selected_sha,
        "gt_supported_classes": int(support.sum()),
        "fallback_to_shared_anchor": int((~support).sum()),
        "strict_rival_ties": int(ties),
        "selection_seconds": time.perf_counter() - t0,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "crop": crop,
        "stride": stride,
        "short_side": protocol_short_side(protocol, ds),
        "logit_scale": protocol_logit_scale(protocol, ds),
        "prob_thd": protocol_prob_thd(protocol, ds),
        "area_thd": protocol_area_thd(protocol, ds),
        "weights_updated": False,
        "validation_labels_read_before_eval": False,
        "table_sha256": sha256_json({
            "h_anchor": int(h_anchor), "h_class": h_class_C.tolist()
        }),
        "protocol": protocol_record(protocol, ds, method_type),
        "protocol_sha256": p_hash,
        "run_sha256": r_hash,
        "seed": seed,
        "calibration_frac": calibration_frac,
        "checkpoint_resumed_at": start_index,
    }
    json_dump(path, result)
    del model, tf
    torch.cuda.empty_cache()
    return result


def ensure_selection(
    ds: str,
    method_type: str,
    *,
    force: bool = False,
    protocol: ProtocolSpec = PROTOCOLS[DEFAULT_PROTOCOL],
    out: Path = OUT,
    seed: int = DEFAULT_SEED,
    calibration_frac: float = DEFAULT_CALIBRATION_FRAC,
):
    p_hash = protocol_sha256(protocol, ds, method_type)
    r_hash = run_sha256(p_hash, seed, calibration_frac, method_type)
    path = selection_manifest_path(out, method_type, ds, seed, calibration_frac)
    if path.exists() and not force:
        selection = json.loads(path.read_text())
        require_artifact_protocol(
            selection, p_hash, r_hash, kind=f"selection {path}",
            require_hash=not protocol.allow_legacy_cache_reuse,
        )
        if selection.get("h_class_C") is not None:
            return selection

    full, selected, sampling = calibration_items(ds, seed, calibration_frac)
    if not selected:
        raise RuntimeError(f"{ds}: no calibration images")
    split_audit = _assert_selection_uses_train_only(ds, full, selected)
    sampling_path = _write_sampling_manifest(
        out, ds, seed, calibration_frac, sampling, split_audit
    )
    cache_dir, cache_record = ensure_perturb_cache(
        ds, method_type, selected, force=force, protocol=protocol, out=out,
        seed=seed,
    )
    h_anchor, h_class_A, _conf, v_miou, fallback = aggregate_A(
        cache_dir, selected, DATASETS[ds]["num_classes"]
    )
    selection = select_competition(
        ds, method_type, selected, h_anchor, h_class_A,
        force=force, protocol=protocol, out=out, seed=seed,
        calibration_frac=calibration_frac,
    )
    selection.update({
        "sampling": sampling,
        "sampling_manifest": str(sampling_path),
        "split_audit": split_audit,
        "cache": cache_record,
        "h_anchor_A": int(h_anchor),
        "h_class_A_computed": h_class_A.tolist(),
        "train_variant_miou": [
            None if not np.isfinite(v) else float(v * 100)
            for v in v_miou.tolist()
        ],
        "train_fallback_classes_A": fallback,
        "protocol": protocol_record(protocol, ds, method_type),
        "protocol_sha256": p_hash,
        "run_sha256": r_hash,
        "seed": seed,
        "calibration_frac": calibration_frac,
    })
    json_dump(path, selection)
    return selection


def prediction_from_field(
    field, ori_hw, threshold: float, logit_scale: float
):
    probs = F.interpolate(
        field.unsqueeze(0), size=ori_hw, mode="bilinear", align_corners=False
    )[0].float()
    probs = (probs * logit_scale).softmax(dim=0)
    pred = probs.argmax(dim=0)
    if threshold > 0:
        pred = pred.clone()
        pred[probs.max(dim=0).values < threshold] = 0
    return pred


def prediction_for_protocol(
    field,
    ori_hw,
    ds: str,
    protocol: ProtocolSpec,
    query_idx,
    num_classes: int,
):
    settings = protocol_settings(protocol, ds)
    if protocol_is_native_sclip(protocol):
        return official_query_prediction(
            field,
            ori_hw,
            query_idx,
            num_classes,
            logit_scale=float(settings["logit_scale"]),
            prob_thd=float(settings["prob_thd"]),
            area_thd=settings["area_thd"],
        )
    return prediction_from_field(
        field,
        ori_hw,
        float(settings["prob_thd"]),
        float(settings["logit_scale"]),
    )


def _result_paths(
    out: Path,
    method_type: str,
    ds: str,
    limit: int | None,
    seed: int,
    calibration_frac: float,
) -> tuple[Path, Path, Path]:
    suffix = result_run_suffix(seed, calibration_frac)
    if limit is not None:
        suffix += f"_smoke{limit}"
    base = out / "results" / source_method_key(method_type)
    result_path = base / f"{ds}{suffix}.json"
    perimg_path = base / f"{ds}{suffix}_perimg.npz"
    checkpoint_path = base / f"{ds}{suffix}_checkpoint.npz"
    return result_path, perimg_path, checkpoint_path


@torch.no_grad()
def combined_crop_logits(
    model, tf32, crop_half, h_anchor, h_star, method_type, query_idx,
    return_query: bool = False,
):
    """Compute baseline, anchor, and class-residual logits from one trunk pass."""
    if method_type == "SCLIP":
        heads, attn, trunk_x = _trunk_last_block(
            model, crop_half, model_type=method_type,
            return_trunk_input=True,
        )
    else:
        heads, attn = _trunk_last_block(
            model, crop_half, model_type=method_type
        )
        trunk_x = None
    h32 = heads.float()
    full = h32.sum(0) + attn.out_proj.bias.float().unsqueeze(0)

    def complete(attn_variant):
        if method_type != "SCLIP":
            return attn_variant
        blk = model.visual.transformer.resblocks[-1]
        y = trunk_x.squeeze(1) + attn_variant.to(trunk_x.dtype)
        y = y.unsqueeze(1)
        y = y + blk.mlp(blk.ln_2(y))
        return y.squeeze(1).float()

    def cosine(attn_variant):
        x = _manual_ln_fp32(
            complete(attn_variant).unsqueeze(0), model.visual.ln_post
        )[0]
        tokens = x[1:] @ model.visual.proj.float()
        tokens = tokens / tokens.norm(dim=-1, keepdim=True)
        return (tokens @ tf32.T).T

    C = int(h_star.numel())
    raw_q = cosine(full)
    A_q = cosine(full + h32[h_anchor])
    query_heads = h_star[torch.as_tensor(
        query_idx, device=h_star.device, dtype=torch.long
    )]
    B_q = torch.empty_like(A_q)
    for h in torch.unique(query_heads):
        logits = cosine(full + h32[h_anchor] + h32[h])
        mask = query_heads == h
        B_q[mask] = logits[mask]
    if return_query:
        return raw_q, A_q, B_q
    raw = class_max_logits(raw_q, query_idx, C)
    A = class_max_logits(A_q, query_idx, C)
    B = class_max_logits(B_q, query_idx, C)
    return raw, A, B


@torch.no_grad()
def baseline_crop_logits(
    model, tf32, crop_half, method_type, query_idx, C,
    return_query: bool = False,
):
    """Return unmodified logits, optionally retaining official query channels."""
    if method_type == "SCLIP":
        heads, attn, trunk_x = _trunk_last_block(
            model, crop_half, model_type=method_type,
            return_trunk_input=True,
        )
    else:
        heads, attn = _trunk_last_block(
            model, crop_half, model_type=method_type,
        )
        trunk_x = None
    full = heads.float().sum(0) + attn.out_proj.bias.float().unsqueeze(0)
    if method_type == "SCLIP":
        blk = model.visual.transformer.resblocks[-1]
        x = trunk_x.squeeze(1) + full.to(trunk_x.dtype)
        x = x.unsqueeze(1)
        x = x + blk.mlp(blk.ln_2(x))
        full = x.squeeze(1).float()
    x = _manual_ln_fp32(full.unsqueeze(0), model.visual.ln_post)[0]
    tokens = x[1:] @ model.visual.proj.float()
    tokens = tokens / tokens.norm(dim=-1, keepdim=True)
    logits = (tokens @ tf32.T).T
    if return_query:
        return logits
    return class_max_logits(logits, query_idx, C)


@torch.no_grad()
def evaluate_baseline(
    ds: str,
    method_type: str,
    *,
    force: bool = False,
    protocol: ProtocolSpec = PROTOCOLS[DEFAULT_PROTOCOL],
    out: Path = OUT,
    seed: int = DEFAULT_SEED,
):
    """Evaluate a base method without building or reading a CCAR head table."""
    p_hash = protocol_sha256(protocol, ds, method_type)
    base = out / "results" / source_method_key(method_type)
    result_path = base / f"{ds}_baseline.json"
    perimg_path = base / f"{ds}_baseline_perimg.npz"
    checkpoint_path = base / f"{ds}_baseline_checkpoint.npz"
    if result_path.exists() and perimg_path.exists() and not force:
        result = json.loads(result_path.read_text())
        require_artifact_protocol(
            result, p_hash, None, kind=f"baseline result {result_path}",
            require_hash=True,
        )
        return result

    items = dataset_items(ds)
    if not items:
        raise RuntimeError(f"{ds}: validation split is empty")
    gpu_guard(8192)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(precision="fp16", device="cuda", vit_type="ViT-B/16")
    tf, query_idx = build_text_features(
        model, str(class_file_for(ds, method_type)), device="cuda"
    )
    tf32 = tf.float()
    C = DATASETS[ds]["num_classes"]
    settings = protocol_settings(protocol, ds)
    crop_size, stride = int(settings["crop"]), int(settings["stride"])
    query_level = protocol_is_native_sclip(protocol)
    Q = int(tf.shape[0]) if query_level else C
    conf = np.zeros((C, C), dtype=np.int64)
    perimg = np.zeros((len(items), 3, C), dtype=np.uint64)
    pred_pixels = np.zeros(C, dtype=np.uint64)
    gt_pixels = np.zeros(C, dtype=np.uint64)
    items_hash = sha256_bytes("\n".join(p.stem for p, _ in items).encode())
    start_index = 0
    compute_s = 0.0
    if checkpoint_path.exists() and not force:
        with np.load(checkpoint_path, allow_pickle=False) as data:
            identity = (
                str(data["protocol_sha256"].item()),
                str(data["items_stems_sha256"].item()),
            )
            if identity != (p_hash, items_hash):
                raise ProtocolMismatchError(
                    f"baseline checkpoint mismatch: {checkpoint_path}"
                )
            start_index = int(data["next_index"].item())
            conf = data["conf"].astype(np.int64)
            perimg = data["tp_fp_fn"].astype(np.uint64)
            pred_pixels = data["pred_pixels"].astype(np.uint64)
            gt_pixels = data["gt_pixels"].astype(np.uint64)
            compute_s = float(data["compute_s"].item())

    def save_checkpoint(next_index: int) -> None:
        npz_dump(
            checkpoint_path,
            protocol_sha256=np.array(p_hash),
            items_stems_sha256=np.array(items_hash),
            next_index=np.array(next_index, dtype=np.int64),
            conf=conf,
            tp_fp_fn=perimg,
            pred_pixels=pred_pixels,
            gt_pixels=gt_pixels,
            compute_s=np.array(compute_s),
        )

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for n0 in range(start_index, len(items)):
            image_path, gt_path = items[n0]
            x, ori_hw = preprocess(image_path, short_side=int(settings["short_side"]))
            x = x.to("cuda")
            _, _, height, width = x.shape
            canvas = torch.zeros(
                (Q, height, width), device="cuda", dtype=torch.float32
            )
            count = torch.zeros(
                (1, height, width), device="cuda", dtype=torch.float32
            )
            torch.cuda.synchronize()
            stage_t = time.perf_counter()
            for y1, x1, y2, x2 in slide_crops(
                height, width, crop_size, stride
            ):
                crop_img = x[:, :, y1:y2, x1:x2]
                hr, wr = crop_img.shape[-2:]
                pad = compute_padsize(hr, wr, 16)
                if any(pad):
                    crop_img = F.pad(crop_img, pad)
                logits = baseline_crop_logits(
                    model, tf32, crop_img.half(), method_type, query_idx, C,
                    return_query=query_level,
                )
                place(logits, canvas, y1, x1, y2, x2, crop_img, pad)
                count[:, y1:y2, x1:x2] += 1
            if torch.any(count == 0):
                raise AssertionError(f"uncovered validation image {image_path}")
            canvas /= count
            pred = prediction_for_protocol(
                canvas, ori_hw, ds, protocol, query_idx, C
            )
            torch.cuda.synchronize()
            compute_s += time.perf_counter() - stage_t
            gt = load_gt(
                gt_path,
                DATASETS[ds]["reduce_zero_label"],
                label_map=DATASETS[ds].get("label_map"),
            )
            pred_np = pred.cpu().numpy().astype(np.int64)
            c = confusion_from_pred(pred_np, gt, C)
            conf += c
            tp = np.diag(c).astype(np.uint64)
            perimg[n0, 0] = tp
            perimg[n0, 1] = c.sum(0) - tp
            perimg[n0, 2] = c.sum(1) - tp
            valid = gt != 255
            pred_pixels += np.bincount(pred_np[valid], minlength=C).astype(np.uint64)
            gt_pixels += np.bincount(gt[valid], minlength=C).astype(np.uint64)
            n = n0 + 1
            if n == 1 or n % 100 == 0 or n == len(items):
                save_checkpoint(n)
            if n == 1 or n % 50 == 0 or n == len(items):
                elapsed = time.perf_counter() - t0
                print(
                    f"[baseline {method_type} {ds}] {n}/{len(items)} "
                    f"{(n - start_index) / max(elapsed, 1e-6):.2f} img/s",
                    flush=True,
                )
    except Exception:
        save_checkpoint(max(start_index, n0))
        raise

    iou = iou_from_confusion(conf)
    result = {
        "dataset": ds,
        "method_type": method_type,
        "result_type": "reproduced_baseline",
        "n_validation": len(items),
        "items_stems_sha256": items_hash,
        "protocol": protocol_record(protocol, ds, method_type),
        "protocol_sha256": p_hash,
        "seed": seed,
        "values_miou": {"baseline": float(miou(iou) * 100)},
        "per_class_iou": {"baseline": iou.tolist()},
        "prediction_pixels": pred_pixels.tolist(),
        "ground_truth_pixels": gt_pixels.tolist(),
        "runtime_s": time.perf_counter() - t0,
        "compute_s": compute_s,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "weights_updated": False,
        "checkpoint_resumed_at": start_index,
    }
    json_dump(result_path, result)
    npz_dump(
        perimg_path,
        stems=np.array([p.stem for p, _ in items]),
        tp_fp_fn=perimg,
        protocol_sha256=np.array(p_hash),
    )
    del model, tf, tf32, x
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def evaluate(
    ds: str,
    method_type: str,
    selection: dict,
    *,
    limit: int | None = None,
    force: bool = False,
    protocol: ProtocolSpec = PROTOCOLS[DEFAULT_PROTOCOL],
    out: Path = OUT,
    seed: int = DEFAULT_SEED,
    calibration_frac: float = DEFAULT_CALIBRATION_FRAC,
):
    p_hash = protocol_sha256(protocol, ds, method_type)
    r_hash = run_sha256(p_hash, seed, calibration_frac, method_type)
    require_artifact_protocol(
        selection, p_hash, r_hash, kind=f"{method_type} {ds} selection",
        require_hash=not protocol.allow_legacy_cache_reuse,
    )
    result_path, perimg_path, checkpoint_path = _result_paths(
        out, method_type, ds, limit, seed, calibration_frac
    )
    if result_path.exists() and perimg_path.exists() and not force and limit is None:
        result = json.loads(result_path.read_text())
        require_artifact_protocol(
            result, p_hash, r_hash, kind=f"result {result_path}",
            require_hash=not protocol.allow_legacy_cache_reuse,
        )
        with np.load(perimg_path, allow_pickle=False) as data:
            if (
                "protocol_sha256" not in data.files
                or str(data["protocol_sha256"].item()) != p_hash
                or "run_sha256" not in data.files
                or str(data["run_sha256"].item()) != r_hash
            ):
                raise ProtocolMismatchError(f"per-image result mismatch: {perimg_path}")
        return result

    items = dataset_items(ds, limit=limit, seed=seed)
    if not items:
        raise RuntimeError(f"{ds}: validation split is empty")
    gpu_guard(8192)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(precision="fp16", device="cuda", vit_type="ViT-B/16")
    tf, query_idx = build_text_features(
        model, str(class_file_for(ds, method_type)), device="cuda"
    )
    tf32 = tf.float()
    C = DATASETS[ds]["num_classes"]
    settings = protocol_settings(protocol, ds)
    crop_size, stride = int(settings["crop"]), int(settings["stride"])
    query_level = protocol_is_native_sclip(protocol)
    Q = int(tf.shape[0]) if query_level else C
    h_anchor = int(selection["h_anchor"])
    h_class = torch.tensor(
        selection["h_class_C"], dtype=torch.long, device="cuda"
    )
    conf_base = np.zeros((C, C), dtype=np.int64)
    conf_ccar = np.zeros((C, C), dtype=np.int64)
    perimg = np.zeros((len(items), 2, 3, C), dtype=np.uint64)
    items_hash = sha256_bytes("\n".join(p.stem for p, _ in items).encode())
    start_index = 0
    base_compute_s = 0.0
    ccar_compute_s = 0.0
    if checkpoint_path.exists() and not force:
        with np.load(checkpoint_path, allow_pickle=False) as data:
            identity = (
                str(data["protocol_sha256"].item()),
                str(data["run_sha256"].item()),
                str(data["items_stems_sha256"].item()),
            )
            if identity != (p_hash, r_hash, items_hash):
                raise ProtocolMismatchError(
                    f"evaluation checkpoint mismatch: {checkpoint_path}"
                )
            start_index = int(data["next_index"].item())
            conf_base = data["conf_base"].astype(np.int64)
            conf_ccar = data["conf_ccar"].astype(np.int64)
            perimg = data["tp_fp_fn"].astype(np.uint64)
            base_compute_s = float(data["baseline_compute_s"].item())
            ccar_compute_s = float(data["ccar_compute_s"].item())

    def save_checkpoint(next_index: int) -> None:
        npz_dump(
            checkpoint_path,
            protocol_sha256=np.array(p_hash),
            run_sha256=np.array(r_hash),
            items_stems_sha256=np.array(items_hash),
            next_index=np.array(next_index, dtype=np.int64),
            conf_base=conf_base,
            conf_ccar=conf_ccar,
            tp_fp_fn=perimg,
            baseline_compute_s=np.array(base_compute_s),
            ccar_compute_s=np.array(ccar_compute_s),
        )

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for n0 in range(start_index, len(items)):
            image_path, gt_path = items[n0]
            x, ori_hw = preprocess(image_path, short_side=int(settings["short_side"]))
            x = x.to("cuda")
            _, _, height, width = x.shape
            raw_canvas = torch.zeros(
                (Q, height, width), device="cuda", dtype=torch.float32
            )
            a_canvas = torch.zeros_like(raw_canvas)
            b_canvas = torch.zeros_like(raw_canvas)
            count = torch.zeros(
                (1, height, width), device="cuda", dtype=torch.float32
            )
            for y1, x1, y2, x2 in slide_crops(
                height, width, crop_size, stride
            ):
                crop_img = x[:, :, y1:y2, x1:x2]
                hr, wr = crop_img.shape[-2:]
                pad = compute_padsize(hr, wr, 16)
                if any(pad):
                    crop_img = F.pad(crop_img, pad)
                crop_half = crop_img.half()

                torch.cuda.synchronize()
                stage_t = time.perf_counter()
                raw, A, B = combined_crop_logits(
                    model, tf32, crop_half, h_anchor, h_class,
                    method_type, query_idx, return_query=query_level,
                )
                place(raw, raw_canvas, y1, x1, y2, x2, crop_img, pad)
                place(A, a_canvas, y1, x1, y2, x2, crop_img, pad)
                place(B, b_canvas, y1, x1, y2, x2, crop_img, pad)
                torch.cuda.synchronize()
                ccar_compute_s += time.perf_counter() - stage_t
                count[:, y1:y2, x1:x2] += 1
            if torch.any(count == 0):
                raise AssertionError(f"uncovered validation image {image_path}")
            raw_canvas /= count
            a_canvas /= count
            b_canvas /= count

            stage_t = time.perf_counter()
            if query_level:
                a_class = class_max_logits(
                    a_canvas.permute(1, 0, 2), query_idx, C
                ).permute(1, 0, 2)
                gate = gacr.uncertainty_gate(
                    a_class, float(np.log(C)),
                    logit_scale=float(settings["logit_scale"]),
                )
                gate_q = gate.unsqueeze(0)
                ccar_field = a_canvas + gate_q * (b_canvas - a_canvas)
            else:
                gate = gacr.uncertainty_gate(
                    a_canvas, float(np.log(C)),
                    logit_scale=float(settings["logit_scale"]),
                )
                ccar_field = a_canvas + gate * (b_canvas - a_canvas)
            pred1 = prediction_for_protocol(
                ccar_field, ori_hw, ds, protocol, query_idx, C
            )
            torch.cuda.synchronize()
            ccar_compute_s += time.perf_counter() - stage_t
            pred0 = prediction_for_protocol(
                raw_canvas, ori_hw, ds, protocol, query_idx, C
            )
            gt = load_gt(
                gt_path,
                DATASETS[ds]["reduce_zero_label"],
                label_map=DATASETS[ds].get("label_map"),
            )
            c0 = confusion_from_pred(
                pred0.cpu().numpy().astype(np.int64), gt, C
            )
            c1 = confusion_from_pred(
                pred1.cpu().numpy().astype(np.int64), gt, C
            )
            conf_base += c0
            conf_ccar += c1
            for j, c in enumerate((c0, c1)):
                tp = np.diag(c).astype(np.uint64)
                perimg[n0, j, 0] = tp
                perimg[n0, j, 1] = c.sum(0) - tp
                perimg[n0, j, 2] = c.sum(1) - tp
            n = n0 + 1
            if n == 1 or n % 100 == 0 or n == len(items):
                save_checkpoint(n)
            if n == 1 or n % max(1, len(items) // 10) == 0 or n == len(items):
                elapsed = time.perf_counter() - t0
                print(
                    f"[eval {method_type} {ds}] {n}/{len(items)} "
                    f"{(n - start_index) / max(elapsed, 1e-6):.2f} img/s",
                    flush=True,
                )
    except Exception:
        save_checkpoint(max(start_index, n0))
        raise

    names = ["baseline", "ccar"]
    ious = {
        "baseline": iou_from_confusion(conf_base).tolist(),
        "ccar": iou_from_confusion(conf_ccar).tolist(),
    }
    values = {
        "baseline": float(miou(iou_from_confusion(conf_base)) * 100),
        "ccar": float(miou(iou_from_confusion(conf_ccar)) * 100),
    }
    result = {
        "dataset": ds,
        "method_type": method_type,
        "n_validation": len(items),
        "items_stems_sha256": items_hash,
        "protocol": protocol_record(protocol, ds, method_type),
        "protocol_sha256": p_hash,
        "run_sha256": r_hash,
        "seed": seed,
        "calibration_frac": calibration_frac,
        "h_anchor": h_anchor,
        "h_class_sha256": sha256_json(selection["h_class_C"]),
        "selection_table_sha256": selection["table_sha256"],
        "values_miou": values,
        "per_class_iou": ious,
        "runtime_s": time.perf_counter() - t0,
        "baseline_compute_s": None,
        "ccar_compute_s": None,
        "paired_shared_trunk_compute_s": ccar_compute_s,
        "timing_note": "baseline and CCAR evaluated from one shared trunk pass",
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "weights_updated": False,
        "selection_path": str(
            selection_manifest_path(
                out, method_type, ds, seed, calibration_frac
            )
        ),
        "checkpoint_resumed_at": start_index,
    }
    json_dump(result_path, result)
    npz_dump(
        perimg_path,
        stems=np.array([p.stem for p, _ in items]),
        names=np.array(names),
        tp_fp_fn=perimg,
        protocol_sha256=np.array(p_hash),
        run_sha256=np.array(r_hash),
    )
    del model, tf, tf32, x
    torch.cuda.empty_cache()
    return result


def smoke(
    ds: str,
    method_type: str,
    protocol: ProtocolSpec,
    out: Path,
    seed: int,
    calibration_frac: float,
):
    items = dataset_items(ds)
    if not items:
        return {
            "dataset": ds, "method_type": method_type,
            "status": "blocked_empty_validation",
        }
    p_hash = protocol_sha256(protocol, ds, method_type)
    r_hash = run_sha256(p_hash, seed, calibration_frac, method_type)
    fake = {
        "h_anchor": 0,
        "h_class_C": [0] * DATASETS[ds]["num_classes"],
        "table_sha256": sha256_json({
            "h_anchor": 0,
            "h_class": [0] * DATASETS[ds]["num_classes"],
        }),
        "protocol_sha256": p_hash,
        "run_sha256": r_hash,
    }
    result = evaluate(
        ds, method_type, fake, limit=1, force=True, protocol=protocol,
        out=out, seed=seed, calibration_frac=calibration_frac,
    )
    result["status"] = "pass"
    result["smoke_only"] = True
    path = out / "logs" / f"smoke_{source_method_key(method_type)}_{ds}.json"
    json_dump(path, result)
    return result


def data_audit(ds: str) -> dict:
    full = train_items(ds)
    validation = dataset_items(ds)
    missing_train = sum(
        1 for image, gt in full if not image.exists() or not gt.exists()
    )
    missing_val = sum(
        1 for image, gt in validation if not image.exists() or not gt.exists()
    )
    train_stems = {p.stem for p, _ in full}
    val_stems = {p.stem for p, _ in validation}
    cfg = DATASETS[ds]
    return {
        "dataset": ds,
        "n_train": len(full),
        "n_validation": len(validation),
        "missing_train_pairs": missing_train,
        "missing_validation_pairs": missing_val,
        "train_validation_stem_overlap": len(train_stems & val_stems),
        "num_classes": cfg["num_classes"],
        "class_file": str(cfg["cls_file"]),
        "class_file_sha256": sha256_file(Path(cfg["cls_file"])),
        "reduce_zero_label": cfg["reduce_zero_label"],
        "ignore_index": 255,
        "background_threshold": BACKGROUND_THRESHOLD.get(ds, 0.0),
        "label_map_sha256": (
            sha256_json(cfg["label_map"])
            if cfg.get("label_map") is not None else None
        ),
    }


def update_state(
    out: Path,
    protocol: ProtocolSpec,
    ds: str,
    method_key: str,
    status: str,
    detail=None,
):
    path = out / "states" / f"{ds}.json"
    state = json.loads(path.read_text()) if path.exists() else {
        "version": "eight_benchmark_protocol_runner_v2",
        "protocol": protocol.name,
        "dataset": ds,
        "state": {},
    }
    if state.get("protocol") != protocol.name:
        raise ProtocolMismatchError(f"state file protocol mismatch: {path}")
    state.setdefault("state", {})[method_key] = status
    if detail is not None:
        state["state"][f"{method_key}_detail"] = detail
    json_dump(path, state)


@contextmanager
def output_lock(out: Path, scope: str):
    out.mkdir(parents=True, exist_ok=True)
    safe_scope = "".join(c if c.isalnum() or c in "_-" else "_" for c in scope)
    lock_path = out / f".runner.{safe_scope}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode())
        yield
    finally:
        os.close(fd)
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def main():
    global GPU_GUARD_OVERRIDE_MIB
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default=DEFAULT_PROTOCOL)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--calibration-frac", type=float, default=DEFAULT_CALIBRATION_FRAC
    )
    ap.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=None)
    ap.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    ap.add_argument(
        "--phase", choices=("audit", "smoke", "baseline", "selection", "eval", "run"),
        default="run",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--min-free-mib", type=int, default=None,
        help=("override all GPU free-memory guards; use 0 to disable the guard "
              "while retaining checkpoint/OOM recovery"),
    )
    args = ap.parse_args()

    if args.min_free_mib is not None and args.min_free_mib < 0:
        ap.error("--min-free-mib must be non-negative")
    GPU_GUARD_OVERRIDE_MIB = args.min_free_mib

    protocol = PROTOCOLS[args.protocol]
    out = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None else default_output_root(protocol)
    )
    old_root = default_output_root(PROTOCOLS["clearclip448"]).resolve()
    if protocol.name in {"lht_dih336", "sclip_official_native"} and out == old_root:
        ap.error("lht_dih336 may not write to the legacy eight_benchmark_v1 root")
    if not 0.0 < args.calibration_frac <= 1.0:
        ap.error("--calibration-frac must be in (0, 1]")
    datasets = args.datasets
    if datasets is None:
        datasets = (
            [d for d in DATASET_ORDER if d != "cityscapes"]
            if protocol.name == "lht_dih336" else DATASET_ORDER
        )

    out.mkdir(parents=True, exist_ok=True)
    invocation = {
        "protocol": protocol.name,
        "output_root": str(out),
        "seed": args.seed,
        "calibration_frac": args.calibration_frac,
        "datasets": datasets,
        "methods": args.methods,
        "phase": args.phase,
        "force": args.force,
        "min_free_mib": args.min_free_mib,
        "started_unix": time.time(),
    }
    json_dump(out / "LAST_INVOCATION.json", invocation)

    lock_scope = "_".join(datasets)
    with output_lock(out, lock_scope):
        if args.phase == "audit":
            records = [data_audit(ds) for ds in datasets]
            json_dump(out / "data_audit.json", {"datasets": records})
            print(json.dumps(records, ensure_ascii=False, indent=2), flush=True)
            return

        for ds in datasets:
            if not dataset_items(ds):
                for method_key in args.methods:
                    update_state(
                        out, protocol, ds, method_key, "blocked_missing_data",
                        {"reason": "validation split absent"},
                    )
                print(f"[blocked] {ds}: validation data absent", flush=True)
                continue
            for method_key in args.methods:
                method_type = METHODS[method_key]
                if args.phase == "baseline":
                    result = evaluate_baseline(
                        ds, method_type, force=args.force, protocol=protocol,
                        out=out, seed=args.seed,
                    )
                    update_state(
                        out, protocol, ds, method_key,
                        "baseline_complete", result,
                    )
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    continue
                if args.phase == "smoke":
                    result = smoke(
                        ds, method_type, protocol, out, args.seed,
                        args.calibration_frac,
                    )
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    continue
                if args.phase in ("selection", "run"):
                    try:
                        selection = ensure_selection(
                            ds, method_type, force=args.force,
                            protocol=protocol, out=out, seed=args.seed,
                            calibration_frac=args.calibration_frac,
                        )
                        update_state(
                            out, protocol, ds, method_key,
                            "selection_complete", selection,
                        )
                    except Exception as exc:
                        update_state(
                            out, protocol, ds, method_key,
                            "blocked_selection", {"error": repr(exc)},
                        )
                        raise
                else:
                    selection_path = selection_manifest_path(
                        out, method_type, ds, args.seed,
                        args.calibration_frac,
                    )
                    selection = json.loads(selection_path.read_text())
                if args.phase in ("eval", "run"):
                    result = evaluate(
                        ds, method_type, selection, force=args.force,
                        protocol=protocol, out=out, seed=args.seed,
                        calibration_frac=args.calibration_frac,
                    )
                    update_state(
                        out, protocol, ds, method_key, "complete", result
                    )
                    print(
                        json.dumps(result["values_miou"], ensure_ascii=False),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
