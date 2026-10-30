"""Check Chest-CT label semantics and OASIS tar.gz image layout."""

import io
import json
import os
import tarfile
import urllib.request

import pyarrow.parquet as pq

HEADERS = {"User-Agent": "python-requests/2.31"}
MIRROR = "https://hf-mirror.com"
os.makedirs("_probe_tmp", exist_ok=True)


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
    return dest


print("=" * 70)
print("Chest-CT: label distribution over the whole train shard")
url = MIRROR + "/api/datasets/Mahadih534/Chest_CT-Scan_images-Dataset/parquet/default/train/0.parquet"
dest = "_probe_tmp/ct_train.parquet"
fetch(url, dest)
t = pq.read_table(dest)
vals = t.column("label").to_pylist()
dist = {}
for v in vals:
    dist[v] = dist.get(v, 0) + 1
print("  label dist:", dist, "rows:", t.num_rows)
for h in t.slice(0, 2).to_pylist():
    print("  row:", {k: str(v)[:50] for k, v in h.items()})
os.remove(dest)

print()
print("=" * 70)
print("Chest-CT README (class names)")
for repo in ["Mahadih534/Chest_CT-Scan_images-Dataset",
             "PranomVignesh/MRI-Images-of-Brain-Tumor",
             "Falah/Alzheimer_MRI",
             "amin-nejad/idrid-disease-grading",
             "BTX24/busbra-birads-processed"]:
    try:
        txt = fetch(MIRROR + "/datasets/%s/resolve/main/README.md" % repo,
                    "_probe_tmp/readme.md").decode("utf-8", "ignore")
        body = txt[:900]
        print("---", repo)
        print(body)
    except Exception as e:
        print("---", repo, "README ERR", str(e)[:90])
