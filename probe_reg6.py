"""Probe candidate continuous-label datasets found via search."""

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


def fetch(url, dest):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=900) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def probe(repo):
    try:
        plan = get_json("https://hf-mirror.com/api/datasets/%s/parquet" % repo)
    except Exception as e:
        return repo, "no parquet (%s)" % str(e)[:60]
    lines = []
    for config, splits in plan.items():
        for split, urls in splits.items():
            lines.append("cfg=%s split=%s shards=%d" % (config, split, len(urls)))
    urls = [u for s in plan.values() for u in list(s.values())[0]]
    if not urls:
        return repo, "\n".join(lines) + " NOFILES"
    dest = os.path.join(TMP, repo.replace("/", "__") + ".parquet")
    try:
        fetch(urls[0].replace("https://huggingface.co", MIRROR), dest)
        t = pq.read_table(dest)
    except Exception as e:
        return repo, "\n".join(lines) + " FETCH ERR %s" % str(e)[:70]
    lines.append("cols=%s rows=%d" % (t.column_names, t.num_rows))
    for h in t.slice(0, 2).to_pylist():
        lines.append("row: %s" % {k: str(v)[:40] for k, v in h.items()})
    for col in t.column_names:
        if col in ("image", "images"):
            continue
        vals = t.column(col).to_pylist()[:1500]
        try:
            nums = [float(v) for v in vals if v is not None and str(v) != ""]
        except (TypeError, ValueError):
            nums = []
        if len(nums) > len(vals) * 0.8:
            lines.append("  %s: min=%.2f max=%.2f mean=%.2f nuniq=%d" % (
                col, min(nums), max(nums), sum(nums) / len(nums), len(set(nums))))
    os.remove(dest)
    return repo, "\n".join(lines)


REPOS = [
    "VashuTheGreat2/oasis-alzheimers-multi-class",
    "amin-nejad/idrid-disease-grading",
    "MahsaTorki/IDRiD_Dataset",
    "MedOtter/IDRiD",
    "kiennduet/BrainAgeMRI",
    "ERISLab/IDRID",
]

if __name__ == "__main__":
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for repo, out in ex.map(probe, REPOS):
            print("=" * 72)
            print(repo)
            print(out)
