"""List all entries of a repo tree with sizes (flat)."""

import json
import sys
import urllib.request

HEADERS = {"User-Agent": "python-requests/2.31"}


def get_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


for repo in sys.argv[1:]:
    print("=" * 72)
    print(repo)
    try:
        items = get_json("https://hf-mirror.com/api/datasets/%s/tree/main" % repo)
    except Exception as e:
        print("  ERR", str(e)[:110])
        continue
    for it in items:
        print("  %-6s %-50s %9.2f MB" % (
            it.get("type"), it.get("path"), (it.get("size") or 0) / 1e6))
