"""Probe parquet schemas of regression-oriented medical datasets."""

import concurrent.futures as cf
import json
import os
import urllib.request

import pyarrow.parquet as pq

HEADERS = {"User-Agent": "python-requests/2.61"}
MIRROR = "https://hf-mirror.com"
TMP = "_probe_tmp"
os.makedirs(TMP, exist_ok=True)


def get_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def parquet_plan(repo):
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
    with urllib.request.urlopen(req, timeout=900) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def probe(repo):
    tag = repo.replace("/", "__")
    try:
        plan = parquet_plan(repo)
    except Exception as e:
        return repo, "PLAN ERR %s" % str(e)[:110]
    lines = []
    for config, splits in plan.items():
        for split, urls in splits.items():
            lines.append("  cfg=%s split=%s shards=%d" % (config, split, len(urls)))
    first = None
    for config, splits in plan.items():
        for split, urls in splits.items():
            if urls:
                first = (config, split, urls[0])
                break
        if first:
            break
    if not first:
        return repo, "\n".join(lines) + "\n  NO PARQUET"
    config, split, url = first
    dest = os.path.join(TMP, tag + ".parquet")
    try:
        fetch(url, dest)
        t = pq.read_table(dest)
        lines.append("  shard rows=%d size=%.1fMB" % (
            t.num_rows, os.path.getsize(dest) / 1e6))
        lines.append("  cols: %s" % t.column_names)
        for h in t.slice(0, 2).to_pylist():
            lines.append("  row: %s" % {k: str(v)[:45] for k, v in h.items()})
        for col in t.column_names:
            if col == "image":
                continue
            vals = t.column(col).to_pylist()[:1500]
            try:
                nums = [float(v) for v in vals if v is not None]
            except (TypeError, ValueError):
                nums = []
            if nums:
                lines.append("  %s: min=%.3f max=%.3f mean=%.3f nuniq=%d" % (
                    col, min(nums), max(nums), sum(nums) / len(nums),
                    len(set(nums))))
        os.remove(dest)
    except Exception as e:
        lines.append("  FETCH ERR %s" % str(e)[:120])
    return repo, "\n".join(lines)


REPOS = [
    "Falah/Alzheimer_MRI",
    "multabench/extended-img-cls-oasis-alzheimers",
    "engabduallah/alzheimer-mri-preprocessed-v01",
    "heig-vd-geo/FLAIR1_mmseg",
    "XavierJiezou/rsseg-mmseg-datasets",
    "BTX24/busbra-birads-processed",
]

if __name__ == "__main__":
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for repo, out in ex.map(probe, REPOS):
            print("=" * 72)
            print(repo)
            print(out)
