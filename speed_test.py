"""Measure mirror download throughput."""

import time
import urllib.request

HEADERS = {"User-Agent": "python-requests/2.31"}
URL = ("https://hf-mirror.com/api/datasets/hf-vision/chest-xray-pneumonia"
       "/parquet/default/test/0.parquet")

req = urllib.request.Request(URL, headers=HEADERS)
t0 = time.time()
total = 0
with urllib.request.urlopen(req, timeout=300) as r:
    while True:
        chunk = r.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > 30 * 1024 * 1024:
            break
dt = time.time() - t0
print("downloaded %.1f MB in %.1fs -> %.2f MB/s" % (total / 1e6, dt, total / 1e6 / dt))
