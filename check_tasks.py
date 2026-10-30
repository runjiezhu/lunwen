"""Cross-check local MedMNIST npz files against the official INFO metadata."""

import json
import os
import re
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "medmnist")

z = zipfile.ZipFile(os.path.join(ROOT, "_pkg", "medmnist-3.0.2-py3-none-any.whl"))
src = z.read("medmnist/info.py").decode("utf-8")


def blocks(text):
    out = {}
    for m in re.finditer(r'^\s*"([a-z0-9]+)":\s*\{', text, re.M):
        name = m.group(1)
        start = m.end() - 1
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    out[name] = text[m.start():i + 1]
                    break
    return out


info = blocks(src)
rows = []
for name, block in info.items():
    tm = re.search(r'"task":\s*"([^"]+)"', block)
    task = tm.group(1) if tm else "?"
    ns = re.search(r'"n_samples":\s*\{([^}]*)\}', block)
    n_samples = dict(re.findall(r'"(\w+)":\s*(\d+)', ns.group(1))) if ns else {}
    npz = os.path.join(DATA_DIR, f"{name}.npz")
    local = {}
    if os.path.exists(npz):
        try:
            d = np.load(npz)
            for sp in ("train", "val", "test"):
                local[sp] = int(d[f"{sp}_labels"].shape[0])
            local["nuniq"] = len(np.unique(d["train_labels"]))
            local["dtype"] = str(d["train_labels"].dtype)
        except Exception as e:
            local["error"] = type(e).__name__
    rows.append((name, task, n_samples, local))

print("%-18s %-22s %-28s %s" % ("subset", "official task", "official n_samples", "local"))
for name, task, ns, local in rows:
    print("%-18s %-22s %-28s %s" % (
        name, task,
        "train=%s val=%s test=%s" % (ns.get("train"), ns.get("val"), ns.get("test")),
        local if local else "MISSING",
    ))
