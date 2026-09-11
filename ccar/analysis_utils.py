"""Shared offline-analysis helpers (used by scripts 06 and 08)."""
import numpy as np
from pathlib import Path

CACHE = {"voc20": "outputs/cache_voc20", "context59": "outputs/cache_context59",
         "ade20k": "outputs/cache_ade20k"}
N_HEADS = 12


def load_confusions(ds):
    out_dir = Path(CACHE[ds])
    files = sorted(out_dir.glob("*.npz"))
    from ccar.pipeline import DATASETS
    C = DATASETS[ds]["num_classes"]
    total = np.zeros((N_HEADS, C, C), dtype=np.int64)
    stems = []
    for f in files:
        total += np.load(f)["head_conf"].astype(np.int64)
        stems.append(f.stem)
    return total, stems


def aggregate(ds, stems_subset):
    out_dir = Path(CACHE[ds])
    from ccar.pipeline import DATASETS
    C = DATASETS[ds]["num_classes"]
    total = np.zeros((N_HEADS, C, C), dtype=np.int64)
    for s in stems_subset:
        total += np.load(out_dir / f"{s}.npz")["head_conf"].astype(np.int64)
    return total


def iou_per_head_class(conf_hcc):
    """(H,C,C) -> (C,H) IoU (nan where class absent)."""
    Hn, C, _ = conf_hcc.shape
    inter = np.einsum("hcc->hc", conf_hcc).astype(np.float64)
    gt = conf_hcc.sum(axis=2).astype(np.float64)
    pred = conf_hcc.sum(axis=1).astype(np.float64)
    union = gt + pred - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return (inter / union).T


def split_stems(ds, stems, seed=42):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(stems))
    half = len(stems) // 2
    return [stems[i] for i in order[:half]], [stems[i] for i in order[half:]]


def load_pert_conf(path):
    """Load one per-image perturbation confusion npz as int64 (picklable worker)."""
    return np.load(path)["conf"].astype(np.int64)


def sum_pert_confs(paths, C):
    """Aggregate a contiguous chunk of perturbation-confusion npz files.

    Integer addition is exact/associative, so chunked partial sums are
    bit-identical to any sequential order (worker for ProcessPool use;
    returns one (2H+1, C, C) int64 array per chunk -> minimal IPC)."""
    out = None
    for p in paths:
        conf = np.load(p)["conf"].astype(np.int64)
        if out is None:
            out = np.zeros_like(conf, dtype=np.int64)
        out += conf
    if out is None:
        raise ValueError("cannot aggregate an empty perturbation-cache chunk")
    return out
