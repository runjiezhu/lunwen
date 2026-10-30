"""Dump the ClassLabel names for every cached parquet shard."""

import json
import os

import pyarrow.parquet as pq

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "parquet_cache")
seen = {}

for f in sorted(os.listdir(CACHE)):
    if not f.endswith(".parquet"):
        continue
    try:
        t = pq.read_schema(os.path.join(CACHE, f))
    except Exception as e:
        print(f, "ERR", str(e)[:60])
        continue
    raw = (t.metadata or {}).get(b"huggingface")
    if not raw:
        print(f, "-> no huggingface metadata")
        continue
    try:
        feats = json.loads(raw.decode("utf-8", "ignore"))["info"]["features"]
    except Exception as e:
        print(f, "-> meta err", e)
        continue
    names = feats.get("label", {}).get("names")
    repo = f.split("__")[0]
    if names and repo not in seen:
        seen[repo] = (f, names)
        print("=" * 70)
        print(repo)
        print("  from:", f)
        print("  names:", names)
    elif not names:
        print("  %-58s -> NO LABEL COLUMN" % f)

if not seen:
    print("no class label metadata found")
