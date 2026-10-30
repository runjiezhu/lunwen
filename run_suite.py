"""Run the full medical-imaging training suite on the local GPU.

Trains every task registered in train_medmnist.py (MedMNIST v2 subsets plus
the externally downloaded public datasets) and writes
checkpoints/<task>/{last.pt,best.pt,summary.json}.

Usage:
    python run_suite.py                 # everything
    python run_suite.py brain-tumor-mri # selected tasks only
    python run_suite.py --external      # only the downloaded datasets
    python run_suite.py --medmnist      # only the MedMNIST subsets
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(ROOT, "train_medmnist.py")
CKPT = os.path.join(ROOT, "checkpoints")
REGISTRY = os.path.join(ROOT, "datasets_registry.json")

# MedMNIST v2 subsets: (task, epochs, batch_size, lr)
MEDMNIST_SUITE = [
    ("pneumoniamnist", 30, 64, 0.001),   # chest X-ray, lung/pleural effusion
    ("retinamnist", 40, 32, 0.001),     # retinal fundus, 5 grades
    ("dermamnist", 30, 64, 0.001),      # dermatoscopy, 7 skin lesions
    ("organamnist", 30, 64, 0.001),     # CT organ, 11 classes
    ("organcmnist", 30, 64, 0.001),     # abdominal CT, 11 classes
    ("organsmnist", 30, 64, 0.001),     # abdominal CT (different crop)
    ("bloodmnist", 30, 64, 0.001),      # blood cell microscopy, 8 classes
    ("breastmnist", 30, 64, 0.001),     # breast ultrasound, 2 classes
    ("octmnist", 30, 64, 0.001),        # retinal OCT, 4 classes
    ("pathmnist", 20, 64, 0.001),       # colorectal pathology, 9 classes
    ("tissuemnist", 15, 64, 0.001),     # colorectal histology, 8 classes
    ("chestmnist", 12, 64, 0.001),      # chest X-ray, 14 multi-label
    ("fracturemnist3d", 30, 32, 0.001), # 3D CT fracture, 3 classes
    ("nodulemnist3d", 30, 32, 0.001),   # 3D lung nodules, 2 classes
    ("organmnist3d", 25, 32, 0.001),    # 3D CT organs, 11 classes
    ("synapsemnist3d", 30, 32, 0.001),  # 3D EM synapses, 2 classes
    ("adrenalmnist3d", 30, 32, 0.001),  # 3D adrenal gland, 2 classes
    ("vesselmnist3d", 30, 32, 0.001),   # 3D vessel segmentation, 2 classes
]

# Downloaded public datasets: (task, epochs, batch_size, lr)
EXTERNAL_SUITE = [
    ("brain-tumor-mri", 30, 64, 0.001),           # brain MRI, 4 tumor types
    ("chest-ct-lung-cancer", 40, 64, 0.001),      # chest CT, 4 lung cancer types
    ("chest-xray-pneumonia", 20, 64, 0.001),      # chest X-ray, pneumonia
    ("alzheimer-mri", 30, 64, 0.001),             # brain MRI, 4 dementia grades
    ("idrid-diabetic-retinopathy", 30, 32, 0.001),  # retinal fundus, 5 DR grades
    ("breast-ultrasound-birads", 30, 64, 0.001),  # breast US, BI-RADS 2-5
    ("oasis-age-reg", 40, 64, 0.001),             # brain MRI -> patient age (regression)
]


def dataset_path(task):
    """Locate the .npz backing a task."""
    med = os.path.join(ROOT, "data", "medmnist", f"{task}.npz")
    if os.path.exists(med):
        return med
    ext = os.path.join(ROOT, "data", "external", f"{task}.npz")
    if os.path.exists(ext):
        return ext
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", help="run only these tasks")
    ap.add_argument("--external", action="store_true")
    ap.add_argument("--medmnist", action="store_true")
    args = ap.parse_args()

    suite = []
    if not args.external:
        suite.extend(MEDMNIST_SUITE)
    if not args.medmnist:
        suite.extend(EXTERNAL_SUITE)
    if args.tasks:
        suite = [s for s in suite if s[0] in args.tasks]
        # allow running a task that is not in the predefined lists
        known = {s[0] for s in suite}
        for t in args.tasks:
            if t not in known:
                suite.append((t, 30, 64, 0.001))

    results = []
    for task, epochs, bs, lr in suite:
        npz = dataset_path(task)
        if npz is None:
            print(f"[skip] {task}: dataset missing", flush=True)
            continue
        print(f"\n{'='*66}\n[train] {task}  epochs={epochs} bs={bs}\n{'='*66}",
              flush=True)
        summary_path = os.path.join(CKPT, task, "summary.json")
        # Drop any stale summary BEFORE training: if this run crashes, the
        # previous run's summary would otherwise be reported as if it were
        # fresh.
        if os.path.exists(summary_path):
            os.remove(summary_path)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, TRAIN, "--task", task, "--epochs", str(epochs),
             "--batch-size", str(bs), "--lr", str(lr)],
            cwd=ROOT, capture_output=True, text=True,
        )
        out = proc.stdout + proc.stderr
        print(out[-1800:])
        entry = {"task": task, "minutes": round((time.time() - t0) / 60, 1)}
        if proc.returncode == 0 and os.path.exists(summary_path):
            with open(summary_path, encoding="utf-8") as f:
                s = json.load(f)
            entry.update({
                "task_type": s.get("task_type"),
                "metric": s.get("metric_name"),
                "best_val": s.get("best_val"),
                "test": s.get("test"),
                "params": s.get("params"),
            })
        else:
            entry["error"] = proc.returncode
            entry["stderr"] = out[-800:]
        results.append(entry)
        print(f"[done] {task} in {entry['minutes']} min -> "
              f"{entry.get('test')} ({entry.get('metric')})", flush=True)

    print(f"\n{'='*66}\nSUMMARY\n{'='*66}")
    for r in results:
        print("%-30s %-8s val=%-9s test=%-9s (%s min)"
              % (r["task"], r.get("task_type"), r.get("best_val"),
                 r.get("test"), r["minutes"]))
    os.makedirs(CKPT, exist_ok=True)
    with open(os.path.join(CKPT, "suite_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
