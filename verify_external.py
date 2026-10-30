"""Verify converted external datasets and the registry."""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "external")

for name in sorted(os.listdir(OUT)):
    if not name.endswith(".npz"):
        continue
    d = np.load(os.path.join(OUT, name))
    print("=" * 66)
    print(name)
    for split in ("train", "val", "test"):
        imgs = d[split + "_images"]
        labs = d[split + "_labels"]
        u = np.unique(labs)
        rng = "%d..%d" % (labs.min(), labs.max())
        print("  %-5s img=%-16s %-7s lbl=%-12s %-8s nuniq=%-4d range=%s"
              % (split, str(imgs.shape), imgs.dtype, str(labs.shape),
                 labs.dtype, len(u), rng))
        if len(u) <= 8:
            print("        values: %s" % u.tolist())

reg = os.path.join(ROOT, "datasets_registry.json")
if os.path.exists(reg):
    print("=" * 66)
    print("registry:")
    with open(reg, encoding="utf-8") as f:
        r = json.load(f)
    for k, v in r.items():
        print("  %-28s %-8s out=%-3s %s" % (
            k, v.get("task_type"), v.get("num_outputs"), v.get("npz")))
