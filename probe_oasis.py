"""Inspect the OASIS Alzheimer dataset for continuous clinical labels."""

import json
import os
import urllib.request

import pyarrow.parquet as pq

HEADERS = {"User-Agent": "python-requests/2.31"}
REPO = "multabench/extended-img-cls-oasis-alzheimers"
BASE = "https://hf-mirror.com/datasets/%s/resolve/main/" % REPO


def fetch(url, dest):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


os.makedirs("_probe_tmp", exist_ok=True)
for name in ["metadata.json", "data.parquet", "README.md"]:
    try:
        dest = fetch(BASE + name, os.path.join("_probe_tmp", name))
        print("=== %s (%d bytes)" % (name, os.path.getsize(dest)))
        if name.endswith(".json") or name.endswith(".md"):
            print(open(dest, encoding="utf-8", errors="ignore").read()[:2500])
        else:
            t = pq.read_table(dest)
            print("cols:", t.column_names, "rows:", t.num_rows)
            for h in t.slice(0, 5).to_pylist():
                print({k: str(v)[:60] for k, v in h.items()})
    except Exception as e:
        print("=== %s ERR %s" % (name, str(e)[:120]))
