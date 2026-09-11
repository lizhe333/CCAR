#!/usr/bin/env python3
"""Convert existing COCO-Stuff raw masks to ClearCLIP COCO-Object masks.

The class mapping is extracted from the vendored ClearCLIP converter at runtime,
so this script does not silently duplicate or alter the official mapping. Images
are reused in-place; only annotation PNGs are written under data/coco_object.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("CCAR_DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser().resolve()
SRC = DATA_ROOT / "coco/stuffthingmaps"
DST = DATA_ROOT / "coco_object/annotations"
OFFICIAL = PROJECT_ROOT / "repos/ClearCLIP/datasets/cvt_coco_object.py"


def official_lut():
    tree = ast.parse(OFFICIAL.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "clsID_to_trID" for t in n.targets))
    raw = ast.literal_eval(node.value)
    lut = np.zeros(256, dtype=np.uint8)
    for key, value in raw.items():
        lut[int(key)] = int(value) + 1 if int(key) <= 90 else 0
    return lut, raw


def convert_split(split: str, lut: np.ndarray, limit: int | None = None):
    src = SRC / split
    dst = DST / split
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.png"))
    if limit is not None:
        files = files[:limit]
    missing = 0
    for i, path in enumerate(files, 1):
        out = dst / f"{path.stem}_instanceTrainIds.png"
        if out.exists():
            continue
        mask = np.asarray(Image.open(path), dtype=np.uint8)
        Image.fromarray(lut[mask], mode="L").save(out, format="PNG")
        if i % 5000 == 0:
            print(f"{split}: {i}/{len(files)}", flush=True)
    return len(files), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    lut, raw = official_lut()
    if not OFFICIAL.exists():
        raise FileNotFoundError(OFFICIAL)
    records = {}
    for split in ("train2017", "val2017"):
        n, missing = convert_split(split, lut, args.limit)
        files = sorted((DST / split).glob("*_instanceTrainIds.png"))
        bad = []
        for f in files[:20]:
            vals = np.unique(np.asarray(Image.open(f)))
            if len(vals) and (vals.min() < 0 or vals.max() > 80):
                bad.append(str(f))
        records[split] = {"source_count_used": n, "output_count": len(files), "missing": missing, "bad_sample_files": bad}
    meta = {"source": str(SRC), "destination": str(DST), "official_converter": str(OFFICIAL), "raw_mapping_entries": len(raw), "records": records, "mapping_sha256": __import__("hashlib").sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()}
    (DST.parent / "conversion_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
