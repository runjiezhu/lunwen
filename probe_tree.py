"""Recursively probe a dataset repo tree with per-directory file counts/sizes."""

import json
import sys
import urllib.request

API = "https://hf-mirror.com/api/datasets/"
HEADERS = {"User-Agent": "python-requests/2.31"}


def get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def walk(repo, path="", depth=0, maxdepth=2):
    url = API + repo + "/tree/main" + ("/" + path if path else "")
    try:
        items = get(url)
    except Exception as e:
        print("  %sERR %s %s" % ("  " * depth, path, str(e)[:80]))
        return 0, 0
    n_files = 0
    size = 0
    dirs = []
    for it in items:
        if it.get("type") == "directory":
            dirs.append(it["path"])
        else:
            n_files += 1
            size += it.get("size") or 0
    print("  %s%-45s files=%-6d %.1f MB" % ("  " * depth, path or "/", n_files, size / 1e6))
    if depth < maxdepth:
        for d in dirs[:8]:
            walk(repo, d, depth + 1, maxdepth)
    return n_files, size


if __name__ == "__main__":
    maxdepth = 2
    args = sys.argv[1:]
    if args and args[0].startswith("--depth="):
        maxdepth = int(args[0].split("=")[1])
        args = args[1:]
    for repo in args:
        print("=" * 72)
        print("REPO:", repo)
        walk(repo, maxdepth=maxdepth)
