"""Training console backend for the medical-imaging training suite.

A dependency-free HTTP server (Python standard library only) that exposes:

    GET  /api/overview          tasks, results, GPU status, aggregate stats
    GET  /api/task/<name>       one task's summary + per-epoch history
    POST /api/train             start a training job
    GET  /api/jobs              list jobs and their status
    GET  /api/jobs/<id>/log     new log lines since a cursor
    POST /api/jobs/<id>/stop    terminate a running job

Run with:
    python console/server.py --port 8770
"""

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "train_medmnist.py")
CKPT = os.path.join(ROOT, "checkpoints")
REGISTRY = os.path.join(ROOT, "datasets_registry.json")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

SUITE_DEFAULTS = {}
try:
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "_run_suite", os.path.join(ROOT, "run_suite.py")
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    for _t, _e, _b, _l in _mod.MEDMNIST_SUITE + _mod.EXTERNAL_SUITE:
        SUITE_DEFAULTS[_t] = {"epochs": _e, "batch_size": _b, "lr": _l}
except Exception:  # pragma: no cover - defaults are a convenience only
    pass


# --------------------------------------------------------------------------
# task metadata
# --------------------------------------------------------------------------

def load_registry():
    """External datasets from the registry plus auto-discovered .npz files."""
    reg = {}
    if os.path.exists(REGISTRY):
        with open(REGISTRY, encoding="utf-8") as f:
            reg = json.load(f)
    ext_dir = os.path.join(ROOT, "data", "external")
    if os.path.isdir(ext_dir):
        for fn in sorted(os.listdir(ext_dir)):
            if fn.endswith(".npz"):
                name = fn[:-4]
                reg.setdefault(name, {"npz": "data/external/" + fn,
                                      "source": "auto-discovered"})
    return reg


def builtin_tasks():
    """MedMNIST v2 tasks, read from the trainer without importing torch."""
    out = {}
    src = open(os.path.join(ROOT, "train_medmnist.py"), encoding="utf-8")
    text = src.read()
    src.close()
    m = re.search(r"BUILTIN_TASKS\s*=\s*\{(.*?)\n\}", text, re.S)
    if not m:
        return out
    for name, body in re.findall(r'"([\w\d]+)":\s*\(([^)]*)\)', m.group(1)):
        parts = [p.strip().strip('"') for p in body.split(",")]
        task_type = parts[0]
        num_out = int(parts[1])
        is_3d = parts[2].lower() == "true"
        out[name] = {
            "task_type": task_type,
            "num_outputs": num_out,
            "is_3d": is_3d,
            "npz": "data/medmnist/%s.npz" % name,
            "source": "MedMNIST v2",
        }
    return out


def all_tasks():
    tasks = builtin_tasks()
    for name, spec in load_registry().items():
        spec = dict(spec)
        spec.setdefault("task_type", None)
        spec.setdefault("num_outputs", 0)
        spec.setdefault("is_3d", False)
        spec.setdefault("source", "external")
        tasks[name] = spec
    # The merged multi-task corpus is trainable through the same entry point.
    merged_npz = os.path.join(ROOT, "data", "merged", "merged.npz")
    if os.path.exists(merged_npz):
        tasks["merged"] = {
            "task_type": "multitask",
            "num_outputs": 0,
            "is_3d": False,
            "npz": os.path.join("data", "merged", "merged.npz"),
            "source": "merged corpus (all 25 tasks)",
            "default_epochs": 10,
            "default_batch_size": 128,
            "default_lr": 0.001,
        }
    for name, spec in tasks.items():
        npz = spec.get("npz") or ""
        p = npz if os.path.isabs(npz) else os.path.join(ROOT, npz)
        spec["npz_path"] = p
        spec["has_data"] = os.path.exists(p)
        spec["size_mb"] = round(os.path.getsize(p) / 1e6, 1) if spec["has_data"] else 0
        d = SUITE_DEFAULTS.get(name, {})
        spec["default_epochs"] = d.get("epochs", 30)
        spec["default_batch_size"] = d.get("batch_size", 64)
        spec["default_lr"] = d.get("lr", 0.001)
        spec["has_result"] = os.path.exists(os.path.join(CKPT, name, "summary.json"))
    return tasks


def merged_info():
    """Summary of the merged multi-task corpus, if it has been built."""
    p = os.path.join(ROOT, "data", "merged", "merged_meta.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        return None
    counts = m.get("counts", {})
    return {
        "n_tasks": m.get("n_tasks"),
        "size": m.get("size"),
        "n_train": counts.get("train"),
        "n_val": counts.get("val"),
        "n_test": counts.get("test"),
        "total": sum(v for v in counts.values() if isinstance(v, int)),
        "bytes_gb": round(os.path.getsize(
            os.path.join(ROOT, "data", "merged", "merged.npz")) / 1e9, 2)
        if os.path.exists(os.path.join(ROOT, "data", "merged", "merged.npz")) else 0,
    }


def read_summary(task):
    p = os.path.join(CKPT, task, "summary.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def all_results():
    rows = []
    if not os.path.isdir(CKPT):
        return rows
    for name in sorted(os.listdir(CKPT)):
        d = os.path.join(CKPT, name)
        if not os.path.isdir(d):
            continue
        s = read_summary(name)
        if not s:
            continue
        rows.append({
            "task": name,
            "task_type": s.get("task_type"),
            "metric": s.get("metric_name"),
            "val": s.get("best_val"),
            "test": s.get("test"),
            "epochs": s.get("epochs"),
            "best_epoch": s.get("best_epoch"),
            "params": s.get("params"),
            "source": s.get("source"),
            "test_mae": s.get("test_mae"),
            "num_outputs": s.get("num_outputs"),
        })
    rows.sort(key=lambda r: (r["task_type"] or "", r["task"]))
    return rows


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------

def gpu_status():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used,utilization.gpu,"
                   "temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if not out:
            return {"available": False, "reason": "no GPU reported"}
        name, total, used, util, temp = [x.strip() for x in out.splitlines()[0].split(",")]
        return {
            "available": True,
            "name": name,
            "memory_total": int(total),
            "memory_used": int(used),
            "memory_pct": round(int(used) / max(int(total), 1) * 100, 1),
            "utilization": int(util),
            "temperature": int(temp),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


# --------------------------------------------------------------------------
# job manager
# --------------------------------------------------------------------------

class Job:
    def __init__(self, job_id, task, params):
        self.id = job_id
        self.task = task
        self.params = params
        self.proc = None
        self.status = "queued"
        self.started = time.time()
        self.ended = None
        self.exit_code = None
        self.lines = []
        self.lock = threading.Lock()
        self.done = threading.Event()
        self._stop_requested = False

    def append(self, line):
        with self.lock:
            self.lines.append(line)
            if len(self.lines) > 5000:
                del self.lines[:1000]

    def log_since(self, cursor):
        with self.lock:
            total = len(self.lines)
            start = max(0, min(cursor, total))
            return self.lines[start:], total

    def payload(self):
        return {
            "id": self.id,
            "task": self.task,
            "params": self.params,
            "status": self.status,
            "started": self.started,
            "ended": self.ended,
            "elapsed": round((self.ended or time.time()) - self.started, 1),
            "exit_code": self.exit_code,
        }


class JobManager:
    def __init__(self):
        self.jobs = {}
        self.order = []
        self.lock = threading.Lock()

    def create(self, task, params):
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id, task, params)
        with self.lock:
            self.jobs[job_id] = job
            self.order.append(job_id)
            if len(self.order) > 100:
                old = self.order.pop(0)
                self.jobs.pop(old, None)
        return job

    def get(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def list(self):
        with self.lock:
            return [self.jobs[j].payload() for j in reversed(self.order)]

    def stop(self, job):
        if job.proc and job.proc.poll() is None:
            job._stop_requested = True
            job.status = "stopping"
            try:
                job.proc.terminate()
            except Exception:
                pass
            return True
        return False


JOBS = JobManager()


def backup_existing(task):
    """Snapshot the current artefacts of a task before they are overwritten."""
    d = os.path.join(CKPT, task)
    if not os.path.isdir(d):
        return []
    saved = []
    for fn in ("summary.json", "best.pt", "last.pt"):
        src = os.path.join(d, fn)
        if not os.path.exists(src):
            continue
        dst = os.path.join(d, fn.replace(".json", ".prev.json")
                           .replace(".pt", ".prev.pt"))
        try:
            shutil.copy2(src, dst)
            saved.append(os.path.basename(dst))
        except Exception:
            pass
    return saved


def run_job(job):
    """Launch the trainer for a job and stream its output into the log."""
    merged = job.task == "merged"
    cmd = [sys.executable, "-u",
           os.path.join(ROOT, "train_merged.py") if merged else TRAIN]
    if merged:
        cmd += ["--epochs", str(job.params.get("epochs") or 10),
                "--batch-size", str(job.params.get("batch_size") or 128),
                "--lr", str(job.params.get("lr") or 0.001),
                "--seed", str(job.params.get("seed") or 42)]
    else:
        cmd += ["--task", job.task]
        for key, flag in (("epochs", "--epochs"), ("batch_size", "--batch-size"),
                          ("lr", "--lr"), ("seed", "--seed"), ("width", "--width")):
            if job.params.get(key) is not None:
                cmd += [flag, str(job.params[key])]
    iters = int(job.params.get("iters") or 0)
    if iters > 0:
        cmd += ["--iters", str(iters)]
    if job.params.get("resume"):
        cmd += ["--resume"]
    job.status = "running"
    if job.params.get("overwrite"):
        for name in backup_existing(job.task):
            job.append("[backup] %s" % name)
    job.append("$ " + " ".join(cmd))
    try:
        job.proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=(0x08000000 if os.name == "nt" else 0),  # NO_WINDOW
        )
    except Exception as exc:
        job.status = "failed"
        job.append("failed to launch: %s" % exc)
        job.ended = time.time()
        job.exit_code = -1
        job.done.set()
        return

    for raw in job.proc.stdout:
        for line in raw.rstrip("\n").splitlines():
            job.append(line)
    job.proc.wait()
    job.exit_code = job.proc.returncode
    job.ended = time.time()
    if job._stop_requested:
        job.status = "stopped"
    elif job.exit_code == 0:
        job.status = "done"
    else:
        job.status = "failed"
    job.done.set()


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "TrainConsole/1.0"

    def log_message(self, fmt, *args):
        pass

    # -- helpers ----------------------------------------------------------
    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype):
        if not os.path.exists(path):
            self.send_error(404)
            return
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8-sig"))
        except Exception:
            return {}

    # -- routing ----------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self.send_file(os.path.join(STATIC, "index.html"),
                                  "text/html; charset=utf-8")
        if path == "/api/overview":
            tasks = all_tasks()
            results = all_results()
            by_task = {r["task"]: r for r in results}
            for name, spec in tasks.items():
                spec["result"] = by_task.get(name)
            cls = [r for r in results if r["task_type"] == "cls"]
            return self.send_json({
                "tasks": tasks,
                "results": results,
                "gpu": gpu_status(),
                "merged": merged_info(),
                "stats": {
                    "total": len(results),
                    "available": len([t for t in tasks.values() if t["has_data"]]),
                    "cls": len(cls),
                    "best_cls": max(cls, key=lambda r: r["test"] or 0)["task"] if cls else None,
                    "mean_cls": round(sum(r["test"] for r in cls) / len(cls), 4) if cls else None,
                    "jobs": len(JOBS.list()),
                },
                "running": [j for j in JOBS.list() if j["status"] in ("running", "queued")],
            })
        if path.startswith("/api/task/"):
            name = path[len("/api/task/"):]
            s = read_summary(name)
            if s is None:
                return self.send_json({"error": "no result for %s" % name}, 404)
            spec = all_tasks().get(name, {})
            return self.send_json({"summary": s, "spec": spec})
        if path == "/api/jobs":
            return self.send_json({"jobs": JOBS.list()})
        m = re.match(r"^/api/jobs/([0-9a-f]+)/log$", path)
        if m:
            job = JOBS.get(m.group(1))
            if job is None:
                return self.send_json({"error": "unknown job"}, 404)
            q = self.path.split("?")[1] if "?" in self.path else ""
            cursor = 0
            for kv in q.split("&"):
                if kv.startswith("since="):
                    try:
                        cursor = int(kv[6:])
                    except ValueError:
                        cursor = 0
            lines, total = job.log_since(cursor)
            return self.send_json({"lines": lines, "cursor": total,
                                   "status": job.status, "job": job.payload()})
        if path == "/api/gpu":
            return self.send_json(gpu_status())
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            full = os.path.normpath(os.path.join(STATIC, rel))
            if not full.startswith(STATIC):
                return self.send_error(403)
            ctype = ("text/css; charset=utf-8" if full.endswith(".css")
                     else "application/javascript; charset=utf-8"
                     if full.endswith(".js") else "text/plain; charset=utf-8")
            return self.send_file(full, ctype)
        return self.send_error(404)

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self.read_body()
        if path == "/api/train":
            task = body.get("task")
            if not task:
                return self.send_json({"error": "task required"}, 400)
            tasks = all_tasks()
            if task not in tasks:
                return self.send_json({"error": "unknown task %s" % task}, 400)
            if not tasks[task]["has_data"]:
                return self.send_json({"error": "dataset missing for %s" % task}, 400)
            running = [j for j in JOBS.list() if j["status"] == "running"]
            if running:
                return self.send_json(
                    {"error": "已有训练任务在运行: %s" % running[0]["task"]}, 409)
            spec = tasks[task]
            resume = bool(body.get("resume"))
            existing = os.path.join(CKPT, task, "summary.json")
            if os.path.exists(existing) and not body.get("overwrite") and not resume:
                prev = read_summary(task) or {}
                return self.send_json({
                    "error": "任务 %s 已有结果（%s %s=%.4f），重跑会覆盖原结果。"
                             "确认覆盖请勾选「覆盖已有结果」。"
                             % (task, prev.get("metric_name", "?"),
                                "test", prev.get("test") or 0.0),
                    "needs_overwrite": True,
                }, 409)
            spec = tasks[task]
            params = {
                "epochs": int(body.get("epochs") or spec["default_epochs"]),
                "batch_size": int(body.get("batch_size") or spec["default_batch_size"]),
                "lr": float(body.get("lr") if body.get("lr") is not None
                            else spec["default_lr"]),
                "seed": int(body.get("seed") or 42),
                "width": int(body.get("width") or 32),
                "overwrite": bool(body.get("overwrite")),
                "iters": int(body.get("iters") or 0),
                "resume": resume,
            }
            job = JOBS.create(task, params)
            threading.Thread(target=run_job, args=(job,), daemon=True).start()
            return self.send_json({"job": job.payload()})
        m = re.match(r"^/api/jobs/([0-9a-f]+)/stop$", path)
        if m:
            job = JOBS.get(m.group(1))
            if job is None:
                return self.send_json({"error": "unknown job"}, 404)
            ok = JOBS.stop(job)
            return self.send_json({"stopped": ok, "job": job.payload()})
        return self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("Training console on http://%s:%d" % (args.host, args.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
