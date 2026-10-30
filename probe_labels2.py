"""Check label semantics (READMEs) and the OASIS zip-based image dataset."""

import concurrent.futures as cf
import io
import json
import os
import urllib.request
import zipfile

HEADERS = {"User-Agent": "python-requests/2.31"}
MIRROR = "https://hf-mirror.com"
os.makedirs("_probe_tmp", exist_ok=True)


def fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.read()


def readme(repo):
    try:
        txt = fetch(MIRROR + "/datasets/%s/resolve/main/README.md" % repo)
        return repo, txt.decode("utf-8", "ignore")[:1200]
    except Exception as e:
        return repo, "ERR %s" % str(e)[:100]


def oasis_zip():
    repo = "multabench/extended-img-cls-oasis-alzheimers"
    lines = []
    try:
        zdata = fetch(MIRROR + "/datasets/%s/resolve/main/images-00000.zip" % repo)
        lines.append("zip bytes: %.1f MB" % (len(zdata) / 1e6))
        z = zipfile.ZipFile(io.BytesIO(zdata))
        names = z.namelist()
        lines.append("entries: %d" % len(names))
        for n in names[:8]:
            lines.append("   %s (%d bytes)" % (n, z.getinfo(n).file_size))
    except Exception as e:
        lines.append("zip ERR %s" % str(e)[:120])
    try:
        meta = fetch(MIRROR + "/datasets/%s/resolve/main/data.parquet" % repo)
        import pyarrow.parquet as pq
        t = pq.read_table(io.BytesIO(meta))
        lines.append("data.parquet cols=%s rows=%d" % (t.column_names, t.num_rows))
        rows = t.to_pylist()
        ages = [float(r["Age"]) for r in rows if r.get("Age") not in (None, "", "None")]
        lines.append("Age: n=%d min=%.0f max=%.0f nuniq=%d" % (
            len(ages), min(ages), max(ages), len(set(ages))))
    except Exception as e:
        lines.append("parquet ERR %s" % str(e)[:120])
    return repo, "\n".join(lines)


if __name__ == "__main__":
    repos = ["Mahadih534/Chest_CT-Scan_images-Dataset",
             "Falah/Alzheimer_MRI",
             "amin-nejad/idrid-disease-grading",
             "BTX24/busbra-birads-processed",
             "PranomVignesh/MRI-Images-of-Brain-Tumor"]
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(readme, r) for r in repos]
        futs.append(ex.submit(oasis_zip))
        for fut in cf.as_completed(futs):
            repo, out = fut.result()
            print("=" * 72)
            print(repo)
            print(out)
