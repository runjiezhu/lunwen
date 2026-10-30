"""Inspect the OASIS MRI zip: image files + continuous clinical columns."""

import io
import urllib.request
import zipfile

import pyarrow.parquet as pq

HEADERS = {"User-Agent": "python-requests/2.31"}
MIRROR = "https://hf-mirror.com"
REPO = "multabench/extended-img-cls-oasis-alzheimers"


def fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.read()


zdata = fetch(MIRROR + "/datasets/%s/resolve/main/images-00000.zip" % REPO)
print("zip bytes: %.2f MB" % (len(zdata) / 1e6))
z = zipfile.ZipFile(io.BytesIO(zdata))
names = z.namelist()
print("entries:", len(names))
for n in names[:6]:
    print("   ", n, z.getinfo(n).file_size)

meta = fetch(MIRROR + "/datasets/%s/resolve/main/data.parquet" % REPO)
t = pq.read_table(io.BytesIO(meta))
print("cols:", t.column_names, "rows:", t.num_rows)
rows = t.to_pylist()
for col in ("Age", "MMSE", "eTIV", "nWBV", "CDR"):
    if col in t.column_names:
        vals = [r[col] for r in rows if r[col] not in (None, "", "None")]
        try:
            nums = [float(v) for v in vals]
            print("  %-6s n=%-4d min=%-8.2f max=%-8.2f nuniq=%d" % (
                col, len(nums), min(nums), max(nums), len(set(nums))))
        except (TypeError, ValueError):
            print("  %-6s non-numeric sample=%s" % (col, vals[:5]))
print("sample ID:", rows[0]["ID"])
