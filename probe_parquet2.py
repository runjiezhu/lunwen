"""Inspect the auto-converted parquet branch of a dataset.

Uses the HF `/api/datasets/<repo>/parquet` endpoint (which returns ready-made
parquet URLs per config/split) and rewrites the host to the mirror so that it
stays reachable.
"""

import json
import os
import sys
import urllib.parse
import urllib.request

import pyarrow.parquet as pq

HEADERS = {"User-Agent": "python-requests/2.31"}
MIRROR = "https://hf-mirror.com"


def get_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def parquet_plan(repo):
    """-> {config: {split: [urls]}}"""
    data = get_json("https://hf-mirror.com/api/datasets/%s/parquet" % repo)
    plan = {}
    for config, splits in data.items():
        for split, urls in splits.items():
            plan.setdefault(config, {})[split] = [
                u.replace("https://huggingface.co", MIRROR) for u in urls
            ]
    return plan


def fetch(url, dest):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


if __name__ == "__main__":
    repo = sys.argv[1]
    plan = parquet_plan(repo)
    for config, splits in plan.items():
        for split, urls in splits.items():
            print("config=%s split=%s shards=%d" % (config, split, len(urls)))
    urls = [u for s in plan.values() for u in s["train"]][:1] or \
           [u for s in plan.values() for u in list(s.values())[0]][:1]
    if not urls:
        sys.exit("no parquet")
    print("downloading", urls[0])
    fetch(urls[0], "_tmp_probe.parquet")
    t = pq.read_table("_tmp_probe.parquet")
    print("rows:", t.num_rows, "size_MB:", round(os.path.getsize("_tmp_probe.parquet")/1e6, 1))
    print("schema:", t.schema)
    for h in t.slice(0, 2).to_pylist():
        print({k: str(v)[:70] for k, v in h.items()})
    os.remove("_tmp_probe.parquet")
