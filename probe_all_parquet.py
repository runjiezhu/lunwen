"""Probe parquet schemas of several candidate datasets in parallel."""

import concurrent.futures as cf
import json
import os
import urllib.request

import pyarrow.parquet as pq

HEADERS = {"User-Agent": "python-requests/2.31"}
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
        return repo, "PLAN ERR %s" % str(e)[:100]
    lines = []
    for config, splits in plan.items():
        for split, urls in splits.items():
            lines.append("  cfg=%s split=%s shards=%d" % (config, split, len(urls)))
    first = None
    for config, splits in plan.items():
        for split, urls in splits.items():
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
        lines.append("  first shard rows=%d size=%.1fMB" % (
            t.num_rows, os.path.getsize(dest) / 1e6))
        lines.append("  schema: %s" % str(t.schema).replace("\n", " | "))
        head = t.slice(0, 2).to_pylist()
        for h in head:
            lines.append("  row: %s" % {k: str(v)[:50] for k, v in h.items()})
        # distinct labels if present
        for col in ("label", "labels", "Label"):
            if col in t.column_names:
                vals = t.column(col).to_pylist()[:2000]
                uniq = {}
                for v in vals:
                    uniq[str(v)] = uniq.get(str(v), 0) + 1
                lines.append("  %s dist(first2000): %s" % (col, uniq))
        os.remove(dest)
    except Exception as e:
        lines.append("  FETCH ERR %s" % str(e)[:120])
    return repo, "\n".join(lines)


REPOS = [
    "BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset",
    "hf-vision/chest-xray-pneumonia",
    "Mahadih534/Chest_CT-Scan_images-Dataset",
    "maktub83/BoneAgeTW2-crops",
    "PranomVignesh/MRI-Images-of-Brain-Tumor",
    "mansi-01/skin_lesion_dataset_upload",
]

if __name__ == "__main__":
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for repo, out in ex.map(probe, REPOS):
            print("=" * 72)
            print(repo)
            print(out)
