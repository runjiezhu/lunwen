"""Inventory every training .npz: split sizes, image shape, label space.

Prints a table plus totals, and writes inventory.json for the merge step.
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DIRS = [os.path.join(ROOT, "data", "medmnist"),
        os.path.join(ROOT, "data", "external")]
OUT = os.path.join(ROOT, "data", "inventory.json")


def shape_str(shape):
    return "x".join(str(x) for x in shape)


def main():
    reg = {}
    rp = os.path.join(ROOT, "datasets_registry.json")
    if os.path.exists(rp):
        reg = json.load(open(rp, encoding="utf-8"))

    rows = []
    for d in DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".npz"):
                continue
            name = fn[:-4]
            path = os.path.join(d, fn)
            with np.load(path, allow_pickle=True) as z:
                keys = list(z.keys())
                tr_i = z["train_images"]
                tr_l = z["train_labels"]
                va_i = z["val_images"]
                te_i = z["test_images"]
                extra = [k for k in keys if k not in (
                    "train_images", "val_images", "test_images",
                    "train_labels", "val_labels", "test_labels")]
                labels = tr_l.reshape(len(tr_l), -1)
                uniq = np.unique(labels, axis=0)
                n_cls = len(uniq)
                dtype = str(tr_l.dtype)
                rows.append({
                    "task": name,
                    "npz": os.path.join(d, fn),
                    "group": "medmnist" if "medmnist" in d else "external",
                    "source": reg.get(name, {}).get("source", ""),
                    "modality": reg.get(name, {}).get("modality", ""),
                    "classes": reg.get(name, {}).get("classes", []),
                    "task_type": "reg" if tr_l.dtype.kind == "f" else
                                 ("multilabel" if labels.shape[1] > 1 else "cls"),
                    "n_train": int(len(tr_i)),
                    "n_val": int(len(va_i)),
                    "n_test": int(len(te_i)),
                    "shape": shape_str(tr_i.shape[1:]),
                    "ndim": int(tr_i.ndim - 1),
                    "label_dim": int(labels.shape[1]),
                    "n_classes": int(n_cls),
                    "label_dtype": dtype,
                    "bytes_mb": round(os.path.getsize(os.path.join(d, fn)) / 1e6, 1),
                    "extra_keys": extra,
                })

    rows.sort(key=lambda r: r["task"])
    hdr = "%-30s %-9s %7s %7s %7s %-16s %6s %5s" % (
        "task", "group", "train", "val", "test", "shape", "ncls", "MB")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-30s %-9s %7d %7d %7d %-16s %6d %5.1f" % (
            r["task"], r["group"], r["n_train"], r["n_val"], r["n_test"],
            r["shape"], r["n_classes"], r["bytes_mb"]))

    tot_tr = sum(r["n_train"] for r in rows)
    tot_va = sum(r["n_val"] for r in rows)
    tot_te = sum(r["n_test"] for r in rows)
    print("-" * len(hdr))
    print("%-30s %-9s %7d %7d %7d" % ("TOTAL", str(len(rows)) + " tasks",
                                      tot_tr, tot_va, tot_te))

    shapes = sorted({r["shape"] for r in rows})
    print("\ndistinct image shapes:")
    for s in shapes:
        members = [r["task"] for r in rows if r["shape"] == s]
        print("  %-16s %d tasks" % (s, len(members)))

    tot_bytes = sum(r["bytes_mb"] for r in rows)
    print("\ntotal on-disk: %.1f MB" % tot_bytes)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"tasks": rows, "totals": {
            "n_train": tot_tr, "n_val": tot_va, "n_test": tot_te,
            "n_tasks": len(rows), "bytes_mb": tot_bytes}}, f, indent=2)
    print("wrote " + OUT)


if __name__ == "__main__":
    main()
