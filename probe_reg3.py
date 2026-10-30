"""Look for a genuine continuous-label medical imaging regression dataset."""

import concurrent.futures as cf
import json
import os
import urllib.request

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


def tree(repo):
    url = "https://hf-mirror.com/api/datasets/%s/tree/main" % repo
    return get_json(url)


def probe(repo):
    lines = []
    try:
        items = tree(repo)
    except Exception as e:
        return repo, "TREE ERR %s" % str(e)[:110]
    for it in items[:25]:
        lines.append("  %-6s %-52s %8.2f MB" % (
            it.get("type"), it.get("path"), (it.get("size") or 0) / 1e6))
    # try parquet
    try:
        plan = get_json("https://hf-mirror.com/api/datasets/%s/parquet" % repo)
    except Exception as e:
        lines.append("  parquet: ERR %s" % str(e)[:70])
        return repo, "\n".join(lines)
    first = None
    for config, splits in plan.items():
        for split, urls in splits.items():
            if urls:
                first = urls[0].replace("https://huggingface.co", MIRROR)
                break
        if first:
            break
    if not first:
        lines.append("  parquet: NONE")
        return repo, "\n".join(lines)
    dest = os.path.join(TMP, repo.replace("/", "__") + ".parquet")
    try:
        fetch(first, dest)
        import pyarrow.parquet as pq
        t = pq.read_table(dest)
        lines.append("  parquet cols=%s rows=%d" % (t.column_names, t.num_rows))
        for col in t.column_names:
            if col == "image":
                continue
            vals = t.column(col).to_pylist()[:1200]
            try:
                nums = [float(v) for v in vals if v is not None]
            except (TypeError, ValueError):
                nums = []
            if nums:
                lines.append("    %s min=%.3f max=%.3f nuniq=%d" % (
                    col, min(nums), max(nums), len(set(nums))))
            else:
                lines.append("    %s sample=%s" % (col, vals[:3]))
        os.remove(dest)
    except Exception as e:
        lines.append("  parquet fetch ERR %s" % str(e)[:100])
    return repo, "\n".join(lines)


REPOS = [
    "tonychen01/lung-RADS2022",
    "multabench/extended-img-cls-oasis-alzheimers",
    "BTX24/ultrasound-breast-classification-combined-birads",
    "Mahadih534/Chest_CT-Scan_images-Dataset",
    "Falah/Alzheimer_MRI",
]

if __name__ == "__main__":
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        for repo, out in ex.map(probe, REPOS):
            print("=" * 72)
            print(repo)
            print(out)
