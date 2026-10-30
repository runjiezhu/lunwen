"""Plan the download: per-dataset parquet shards, sizes, splits, label stats."""

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
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def fetch(url, dest):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=900) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def plan(repo):
    try:
        p = get_json("https://hf-mirror.com/api/datasets/%s/parquet" % repo)
    except Exception as e:
        return repo, {"error": str(e)[:100]}
    out = {"splits": {}}
    for config, splits in p.items():
        for split, urls in splits.items():
            key = "%s/%s" % (config, split)
            out["splits"][key] = {
                "shards": len(urls),
                "urls": [u.replace("https://huggingface.co", MIRROR) for u in urls],
            }
    # inspect one shard of each split for row count and label stats
    stats = {}
    for key, info in out["splits"].items():
        if not info["urls"]:
            continue
        dest = os.path.join(TMP, repo.replace("/", "__") + "_" + key.replace("/", "_") + ".parquet")
        try:
            fetch(info["urls"][0], dest)
            t = pq.read_table(dest)
            s = {"rows_shard0": t.num_rows, "cols": t.column_names,
                 "bytes_shard0": os.path.getsize(dest)}
            for col in t.column_names:
                if col in ("image", "images", "img"):
                    continue
                vals = t.column(col).to_pylist()[:3000]
                try:
                    nums = [float(v) for v in vals if v is not None and str(v) != ""]
                except (TypeError, ValueError):
                    nums = []
                if len(nums) > len(vals) * 0.8:
                    s[col] = "num min=%.2f max=%.2f nuniq=%d" % (
                        min(nums), max(nums), len(set(nums)))
                else:
                    s[col] = "cat uniq=%d sample=%s" % (
                        len(set(map(str, vals))), sorted({str(v) for v in vals})[:6])
            # does the image column carry bytes?
            if "image" in t.column_names:
                cell = t.column("image").to_pylist()[0]
                if isinstance(cell, dict):
                    s["image_has_bytes"] = cell.get("bytes") is not None
            stats[key] = s
            os.remove(dest)
        except Exception as e:
            stats[key] = {"err": str(e)[:100]}
    out["stats"] = stats
    return repo, out


REPOS = [
    "PranomVignesh/MRI-Images-of-Brain-Tumor",
    "Mahadih534/Chest_CT-Scan_images-Dataset",
    "hf-vision/chest-xray-pneumonia",
    "Falah/Alzheimer_MRI",
    "amin-nejad/idrid-disease-grading",
    "BTX24/busbra-birads-processed",
    "Mahadih534/brain-tumor-MRI-dataset",
    "multabench/extended-img-cls-oasis-alzheimers",
]

if __name__ == "__main__":
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        results = dict(ex.map(plan, REPOS))
    with open("_probe_tmp/plan.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    for repo, out in results.items():
        print("=" * 72)
        print(repo)
        if "error" in out:
            print("  ERROR", out["error"])
            continue
        for key, info in out["splits"].items():
            print("  split %-22s shards=%d" % (key, info["shards"]))
        for key, s in out.get("stats", {}).items():
            print("   [%s] %s" % (key, s))
