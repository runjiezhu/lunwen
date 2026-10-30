"""Unit-test the canonical label mapping against real parquet metadata."""

import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_datasets import (DATASETS, canonical_label_map,  # noqa: E402
                               read_class_names)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "parquet_cache")

for name, spec in DATASETS.items():
    if spec.get("custom"):
        continue
    files = [f for f in sorted(os.listdir(CACHE))
             if f.startswith(spec["repo"].replace("/", "__"))]
    if not files:
        print("%-28s (no cache)" % name)
        continue
    names = None
    raw_all = set()
    per_split = {}
    for f in files:
        path = os.path.join(CACHE, f)
        split = f.split("__")[2]
        if names is None:
            names = read_class_names(path)
        try:
            t = pq.read_table(path, columns=["label"])
        except Exception:
            continue
        vals = t.column("label").to_pylist()
        raw_all.update(vals)
        per_split.setdefault(split, {}).update(
            {v: vals.count(v) for v in set(vals)})
    lut = canonical_label_map(raw_all, names, spec.get("classes", []))
    print("=" * 70)
    print("%-28s classes=%s" % (name, spec.get("classes")))
    print("  metadata names: %s" % (names if names else None))
    print("  raw->canonical: %s" % lut)
    for split, dist in per_split.items():
        canon = {}
        for raw, cnt in dist.items():
            c = (lut or {}).get(raw, raw)
            canon[c] = canon.get(c, 0) + cnt
        print("    %-6s raw=%s -> canonical=%s" % (split, dist, canon))
