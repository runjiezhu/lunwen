"""Inspect OASIS metadata CSVs and the PranomVignesh MRI dataset labels."""

import io
import json
import os
import urllib.request
import zipfile

import pyarrow.parquet as pq

HEADERS = {"User-Agent": "python-requests/2.31"}
MIRROR = "https://hf-mirror.com"
os.makedirs("_probe_tmp", exist_ok=True)


def fetch(url, dest=None):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    if dest:
        open(dest, "wb").write(data)
    return data


print("=" * 70)
print("OASIS train metadata CSV")
url = MIRROR + "/datasets/VashuTheGreat2/oasis-alzheimers-multi-class/resolve/main/oasis_train_patients_metadata.csv"
txt = fetch(url).decode("utf-8", "ignore")
print(txt[:1500])

print("=" * 70)
print("OASIS parquet (train) full columns")
url = MIRROR + "/api/datasets/VashuTheGreat2/oasis-alzheimers-multi-class/parquet/default/train/0.parquet"
dest = "_probe_tmp/oasis_train.parquet"
fetch(url, dest)
t = pq.read_table(dest)
print("rows:", t.num_rows)
d = t.to_pylist()
mmse = [float(r["MMSE"]) for r in d if r.get("MMSE") not in (None, "")]
print("MMSE n=%d min=%.1f max=%.1f mean=%.2f nuniq=%d" % (
    len(mmse), min(mmse), max(mmse), sum(mmse) / len(mmse), len(set(mmse))))
print("CDR values:", sorted({str(r.get("CDR")) for r in d}))
print("class values:", sorted({str(r.get("class")) for r in d}))
print("sample rows:")
for r in d[:3]:
    print("  ", {k: str(v)[:40] for k, v in r.items()})
