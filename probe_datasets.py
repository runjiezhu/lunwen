"""Probe candidate HF-mirror medical imaging datasets: file tree + total size."""

import json
import sys
import urllib.parse
import urllib.request

API = "https://hf-mirror.com/api/datasets/"
HEADERS = {"User-Agent": "python-requests/2.31"}


def get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def tree(repo, path=""):
    url = API + repo + "/tree/main" + ("/" + path if path else "")
    return get(url)


def probe(repo):
    print("=" * 72)
    print("REPO:", repo)
    try:
        items = tree(repo)
    except Exception as e:
        print("  ERR tree:", type(e).__name__, str(e)[:120])
        return
    total = 0
    for it in items[:40]:
        size = it.get("size") or 0
        total += size
        print("  %-6s %-60s %10.2f MB" % (
            it.get("type"), it.get("path"), size / 1e6))
    if len(items) > 40:
        print("  ... %d more entries" % (len(items) - 40))
        total = sum((i.get("size") or 0) for i in items)
    print("  TOTAL(listed): %.1f MB" % (total / 1e6))


if __name__ == "__main__":
    repos = sys.argv[1:]
    if not repos:
        repos = [
            "BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset",
            "PranomVignesh/MRI-Images-of-Brain-Tumor",
            "Mahadih534/Chest_CT-Scan_images-Dataset",
            "hf-vision/chest-xray-pneumonia",
            "Mahadih534/brain-tumor-MRI-dataset",
            "theshoaibme/panorgan-busi-breast-ultasound",
            "mansi-01/skin_lesion_dataset_upload",
            "mezeidragos-lateral/bach-breast-histopathology-he-staining",
        ]
    for r in repos:
        probe(r)
