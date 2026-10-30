"""Check raw label encodings and class-name metadata for Chest-CT."""

import json
import os

import pyarrow.parquet as pq

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "parquet_cache")

for f in sorted(os.listdir(CACHE)):
    if "Chest_CT" not in f:
        continue
    t = pq.read_table(os.path.join(CACHE, f))
    dist = {}
    for v in t.column("label").to_pylist():
        dist[v] = dist.get(v, 0) + 1
    meta = t.schema.metadata or {}
    hf = meta.get(b"huggingface", b"{}").decode("utf-8", "ignore")
    print("=" * 70)
    print(f)
    print("  rows=%d label_dist=%s" % (t.num_rows, dist))
    try:
        info = json.loads(hf)["info"]["features"]
        print("  features:", json.dumps(info)[:400])
    except Exception as e:
        print("  meta ERR", e)
