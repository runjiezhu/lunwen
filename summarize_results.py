import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "checkpoints")

rows = []
for name in sorted(os.listdir(CKPT)):
    d = os.path.join(CKPT, name)
    if not os.path.isdir(d):
        continue
    p = os.path.join(d, "summary.json")
    if not os.path.exists(p):
        continue
    s = json.load(open(p, encoding="utf-8"))
    rows.append({
        "task": name,
        "type": s.get("task_type"),
        "metric": s.get("metric_name"),
        "val": s.get("best_val"),
        "test": s.get("test"),
        "epochs": s.get("epochs"),
    })

rows.sort(key=lambda r: (r["type"] or "", r["task"]))
w = max(len(r["task"]) for r in rows)
for r in rows:
    v = r["val"]
    t = r["test"]
    print("%-*s %-11s %-4s val=%-10s test=%-10s (%d ep)"
          % (w, r["task"], r["type"], r["metric"],
             "%.4f" % v if isinstance(v, (int, float)) else "n/a",
             "%.4f" % t if isinstance(t, (int, float)) else "n/a",
             r["epochs"] or 0))
print("\ntotal tasks with results: %d" % len(rows))
