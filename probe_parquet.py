"""Check whether the HF mirror exposes the auto-converted parquet branch."""

import json
import sys
import urllib.request

HEADERS = {"User-Agent": "python-requests/2.31"}


def get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def branches(repo):
    url = "https://hf-mirror.com/api/datasets/%s/refs" % repo
    try:
        return [b["name"] for b in get(url)]
    except Exception as e:
        return "ERR %s" % str(e)[:80]


if __name__ == "__main__":
    for repo in sys.argv[1:]:
        print(repo, "->", branches(repo))
        url = "https://hf-mirror.com/api/datasets/%s/parquet" % repo
        try:
            files = get(url)
            tot = 0
            for f in files[:10]:
                print("   ", f.get("config"), f.get("split"), f.get("url"))
            for f in files:
                tot += f.get("size") or 0
            print("    n_parquet=%d total=%.1f MB" % (len(files), tot / 1e6))
        except Exception as e:
            print("    parquet ERR", str(e)[:100])
