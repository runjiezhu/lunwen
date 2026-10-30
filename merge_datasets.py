"""Merge every task into one big multi-task dataset.

Reads all MedMNIST + external .npz files and writes a single
``data/merged/merged.npz`` holding

    {split}_images   uint8 (N, S, S, 3)   unified-resolution RGB
    {split}_task     int16 (N,)           task id
    {split}_cls      int32 (N,)           class label (-1 if not a cls task)
    {split}_ml       uint8 (N, 14)        multi-label targets (chestmnist)
    {split}_reg      float32 (N,)         regression target (oasis-age-reg)

plus ``merged_meta.json`` describing each task, its head type and class count.

All 2D inputs are resampled to a common square resolution (default 32x32);
grayscale is replicated to three channels, RGB keeps its colour, and 3D
volumes are reduced to a maximum-intensity projection so one 2D backbone can
serve every task.

Usage:
    python merge_datasets.py                 # 32x32, default
    python merge_datasets.py --size 64       # higher resolution
    python merge_datasets.py --dry-run       # report sizes only
"""

import argparse
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
INVENTORY = os.path.join(ROOT, "data", "inventory.json")
OUT_DIR = os.path.join(ROOT, "data", "merged")
SPLITS = ("train", "val", "test")
ML_DIM = 14


def resize_hw(arr, size):
    """Nearest-neighbour resample of the two spatial axes of (N,H,W[,C])."""
    h, w = arr.shape[1], arr.shape[2]
    if h == size and w == size:
        return arr
    ri = (np.arange(size) * h // size).clip(0, h - 1)
    rj = (np.arange(size) * w // size).clip(0, w - 1)
    return arr[:, ri][:, :, rj]


def to_rgb(arr, size):
    """(N,...) uint8 image array -> (N, size, size, 3) uint8."""
    a = arr
    if a.ndim == 4:
        if a.shape[-1] in (1, 3):
            # (N, H, W, C) colour images
            if a.shape[-1] == 1:
                a = a[..., 0]
        else:
            # (N, D, H, W) volume -> maximum-intensity projection
            a = a.max(axis=1)
    if a.ndim == 4:  # RGB
        return np.ascontiguousarray(resize_hw(a, size).astype(np.uint8))
    g = resize_hw(a, size)
    return np.ascontiguousarray(
        np.repeat(g[:, :, :, None], 3, axis=3).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32, help="unified image size")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(INVENTORY, encoding="utf-8") as f:
        inv = json.load(f)
    tasks = inv["tasks"]
    size = args.size

    print("merging %d tasks -> %dx%d RGB" % (len(tasks), size, size))
    print("%-30s %-10s %8s %8s %8s  %s" %
          ("task", "head", "train", "val", "test", "note"))
    print("-" * 88)

    metas = []
    per_task_counts = {}
    for t in tasks:
        name = t["task"]
        src = t["npz"]
        path = src if os.path.isabs(src) else os.path.join(ROOT, src)
        with np.load(path, allow_pickle=True) as z:
            counts = {}
            for sp in SPLITS:
                counts[sp] = int(z[f"{sp}_images"].shape[0])
        per_task_counts[name] = counts

        head = "cls"
        n_out = t["n_classes"] if t["task_type"] == "cls" else 0
        if t["task_type"] == "multilabel":
            head, n_out = "multilabel", ML_DIM
        elif t["task_type"] == "reg":
            head, n_out = "reg", 1
        metas.append({
            "task": name, "head": head, "num_outputs": n_out,
            "task_type": t["task_type"], "group": t["group"],
            "source": t.get("source", ""), "modality": t.get("modality", ""),
            "classes": t.get("classes", []),
            "counts": counts, "orig_shape": t["shape"],
            "was_3d": t["ndim"] == 3,
        })
        print("%-30s %-10s %8d %8d %8d  %s" % (
            name, head, counts["train"], counts["val"], counts["test"],
            t["shape"]))

    tot = {sp: sum(per_task_counts[t["task"]][sp] for t in tasks) for sp in SPLITS}
    print("-" * 88)
    print("%-30s %-10s %8d %8d %8d" % ("TOTAL", "", tot["train"], tot["val"],
                                       tot["test"]))
    nbytes = sum(tot.values()) * size * size * 3
    print("merged size on disk: %.2f GB (uncompressed uint8)" % (nbytes / 1e9))
    if args.dry_run:
        return

    os.makedirs(args.out, exist_ok=True)
    out = {}
    for sp in SPLITS:
        out[f"{sp}_images"] = np.zeros((tot[sp], size, size, 3), dtype=np.uint8)
        out[f"{sp}_task"] = np.zeros(tot[sp], dtype=np.int16)
        out[f"{sp}_cls"] = np.full(tot[sp], -1, dtype=np.int32)
        out[f"{sp}_ml"] = np.zeros((tot[sp], ML_DIM), dtype=np.uint8)
        out[f"{sp}_reg"] = np.full(tot[sp], np.nan, dtype=np.float32)

    cursor = {sp: 0 for sp in SPLITS}
    t0 = time.time()
    for tid, meta in enumerate(metas):
        name = meta["task"]
        path = meta.pop("_path", None)
        src = next(t["npz"] for t in tasks if t["task"] == name)
        path = src if os.path.isabs(src) else os.path.join(ROOT, src)
        with np.load(path, allow_pickle=True) as z:
            for sp in SPLITS:
                n = meta["counts"][sp]
                a, b = cursor[sp], cursor[sp] + n
                out[f"{sp}_images"][a:b] = to_rgb(z[f"{sp}_images"], size)
                out[f"{sp}_task"][a:b] = tid
                lab = np.asarray(z[f"{sp}_labels"])
                if meta["head"] == "cls":
                    out[f"{sp}_cls"][a:b] = lab.reshape(n).astype(np.int32)
                elif meta["head"] == "multilabel":
                    out[f"{sp}_ml"][a:b] = lab.reshape(n, ML_DIM).astype(np.uint8)
                else:
                    out[f"{sp}_reg"][a:b] = lab.reshape(n).astype(np.float32)
                cursor[sp] = b
        print("  merged %-30s %5.1fs" % (name, time.time() - t0))

    npz_path = os.path.join(args.out, "merged.npz")
    print("writing %s ..." % npz_path)
    np.savez(npz_path, **out)

    meta = {
        "size": size,
        "n_tasks": len(metas),
        "counts": tot,
        "tasks": metas,
        "ml_dim": ML_DIM,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_files": [t["npz"] for t in tasks],
    }
    with open(os.path.join(args.out, "merged_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("wrote %s (%.2f GB)" %
          (npz_path, os.path.getsize(npz_path) / 1e9))


if __name__ == "__main__":
    main()
