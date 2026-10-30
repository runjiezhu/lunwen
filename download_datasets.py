"""Download public medical-imaging datasets and convert them to MedMNIST-style .npz.

All datasets are pulled from the Hugging Face mirror (https://hf-mirror.com) using
the auto-converted parquet branch, which embeds image bytes directly and is far
faster than fetching thousands of individual image files.

For every dataset we produce:
    data/external/<name>.npz
        train_images / val_images / test_images   uint8 (N, S, S) grayscale
        train_labels / val_labels / test_labels   label array

and register it in datasets_registry.json so that train_medmnist.py can run it
with the same code path as the MedMNIST subsets.

Usage:
    python download_datasets.py --list
    python download_datasets.py --all
    python download_datasets.py brain-tumor-mri oasis-age-reg
"""

import argparse
import concurrent.futures as cf
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, "data", "parquet_cache")
OUT_DIR = os.path.join(ROOT, "data", "external")
REGISTRY = os.path.join(ROOT, "datasets_registry.json")
MIRROR = "https://hf-mirror.com"
HEADERS = {"User-Agent": "python-requests/2.31"}
IMG_SIZE = 64

# name -> spec
DATASETS = {
    "brain-tumor-mri": {
        "repo": "PranomVignesh/MRI-Images-of-Brain-Tumor",
        "task_type": "cls",
        "num_outputs": 4,
        "modality": "MRI",
        "classes": ["glioma", "meningioma", "no-tumor", "pituitary"],
        "homepage": "https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification",
    },
    "chest-ct-lung-cancer": {
        "repo": "Mahadih534/Chest_CT-Scan_images-Dataset",
        "task_type": "cls",
        "num_outputs": 4,
        "modality": "CT",
        # The repo exposes 7 folder variants for 4 diseases (e.g.
        # "adenocarcinoma" and "adenocarcinoma_left.lower.lobe_..."), and the
        # train/test splits use DIFFERENT raw indices for the same disease.
        # Resolution is by class-name prefix match, not by raw index.
        "classes": ["adenocarcinoma", "large.cell.carcinoma", "normal",
                    "squamous.cell.carcinoma"],
        "homepage": "https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images",
    },
    "chest-xray-pneumonia": {
        "repo": "hf-vision/chest-xray-pneumonia",
        "task_type": "cls",
        "num_outputs": 2,
        "modality": "X-ray",
        "classes": ["NORMAL", "PNEUMONIA"],
        "homepage": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
    },
    "alzheimer-mri": {
        "repo": "Falah/Alzheimer_MRI",
        "task_type": "cls",
        "num_outputs": 4,
        "modality": "MRI",
        "classes": ["MildDementia", "ModerateDementia", "NonDementia",
                    "VeryMildDementia"],
        "homepage": "https://www.kaggle.com/datasets/mdzahidmahdi/brain-mri-dataset",
    },
    "idrid-diabetic-retinopathy": {
        "repo": "amin-nejad/idrid-disease-grading",
        "task_type": "cls",
        "num_outputs": 5,
        "modality": "fundus",
        "classes": ["noDR", "mildDR", "moderateDR", "severeDR", "proDR"],
        "homepage": "https://www.kaggle.com/datasets/deathtrooper/diabetic-retinopathy-detection",
    },
    "breast-ultrasound-birads": {
        "repo": "BTX24/busbra-birads-processed",
        "task_type": "cls",
        "num_outputs": 4,
        "modality": "ultrasound",
        "classes": ["2", "3", "4", "5"],
        "homepage": "https://www.kaggle.com/datasets/ahmedmohamedmoustafa/breast-cancer-dataset",
    },
    "oasis-age-reg": {
        "repo": "multabench/extended-img-cls-oasis-alzheimers",
        "task_type": "reg",
        "num_outputs": 1,
        "modality": "MRI",
        "target": "Age",
        "homepage": "https://www.kaggle.com/datasets/shreyanmohanty/oasis-alzheimers-detection-multi-class-dataset",
        "custom": "oasis",
    },
}


# --------------------------------------------------------------------------- io


def _fetch(url, timeout=900):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _get_json(url):
    return json.loads(_fetch(url, timeout=90).decode("utf-8"))


def parquet_plan(repo):
    """{split: [parquet urls]} for the auto-converted branch."""
    data = _get_json("https://hf-mirror.com/api/datasets/%s/parquet" % repo)
    plan = {}
    for _config, splits in data.items():
        for split, urls in splits.items():
            plan.setdefault(split, []).extend(
                u.replace("https://huggingface.co", MIRROR) for u in urls
            )
    return plan


def cached_parquet(repo, split, url):
    """Download a parquet shard (cached on disk) and return its path.

    The mirror occasionally closes the connection early and returns a truncated
    body with a 200 status, which later fails as "Parquet magic bytes not found
    in footer". We therefore compare the received size against Content-Length
    and discard incomplete files instead of caching them.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    name = "%s__%s__%s.parquet" % (
        repo.replace("/", "__"), split, os.path.basename(url))
    dest = os.path.join(CACHE_DIR, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    tmp = dest + ".part"
    for attempt in range(5):
        expected = None
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=1800) as r:
                expected = r.headers.get("Content-Length")
                expected = int(expected) if expected else None
                got = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
            if expected is not None and got != expected:
                raise IOError(
                    "truncated: got %d of %d bytes" % (got, expected))
            os.replace(tmp, dest)
            return dest
        except Exception as e:
            if attempt == 4:
                raise
            print("    retry %d/5 for %s (%s)"
                  % (attempt + 1, 5, os.path.basename(url), str(e)[:70]))
            time.sleep(4 * (attempt + 1))
            if os.path.exists(tmp):
                os.remove(tmp)
    return dest


def decode_image(cell):
    """parquet image cell -> (S, S) uint8 grayscale array."""
    if isinstance(cell, dict):
        if cell.get("bytes"):
            data = cell["bytes"]
        else:
            raise ValueError("image cell has no embedded bytes")
    else:
        raise ValueError("unexpected image cell type: %r" % type(cell))
    img = Image.open(io.BytesIO(data)).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# ------------------------------------------------------------------ converters


def split_by_ratio(items, n_val, n_test, seed=0):
    """Deterministic train/val/test split of an index list."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(items))
    n_test = min(n_test, len(items) // 4)
    n_val = min(n_val, len(items) // 4)
    test = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]
    return train.tolist(), val.tolist(), test.tolist()


def convert_from_parquet(name, spec):
    repo = spec["repo"]
    plan = parquet_plan(repo)
    print("  parquet splits: %s" % {k: len(v) for k, v in plan.items()})

    decoded = {}
    class_names = None
    for split, urls in plan.items():
        images, labels = [], []
        for url in urls:
            path = cached_parquet(repo, split, url)
            if class_names is None:
                class_names = read_class_names(path)
            t = pq.read_table(path)
            img_col = t.column("image") if "image" in t.column_names else None
            lbl_col = t.column("label") if "label" in t.column_names else None
            for i in range(t.num_rows):
                if img_col is None:
                    break
                try:
                    arr = decode_image(img_col[i].as_py())
                except Exception:
                    continue
                images.append(arr)
                if lbl_col is not None:
                    labels.append(lbl_col[i].as_py())
            print("    %-12s %-6s rows=%d" % (split, os.path.basename(url)[:18], t.num_rows))
        if images:
            decoded[split] = (np.stack(images), np.asarray(labels))

    if "train" not in decoded:
        raise RuntimeError("no train split for %s" % name)

    # canonical split mapping
    def take(*names):
        for n in names:
            if n in decoded:
                return decoded[n]
        return None

    tr = take("train")
    va = take("validation", "val")
    te = take("test")

    if va is None:
        # carve a validation slice out of train
        labels_all = tr[1]
        classes = np.unique(labels_all)
        rng = np.random.RandomState(0)
        val_idx = []
        for c in classes:
            ci = np.where(labels_all == c)[0]
            rng.shuffle(ci)
            val_idx.extend(ci[:max(1, int(0.1 * len(ci)))].tolist())
        val_set = set(val_idx)
        tr_idx = [i for i in range(len(tr[0])) if i not in val_set]
        va = (tr[0][val_idx], tr[1][val_idx])
        tr = (tr[0][tr_idx], tr[1][tr_idx])

    if te is None:
        tr_idx, va_idx, te_idx = split_by_ratio(range(len(tr[0])), 0, max(1, len(tr[0]) // 5))
        te = (tr[0][te_idx], tr[1][te_idx])
        va = (tr[0][va_idx], tr[1][va_idx])
        tr = (tr[0][tr_idx], tr[1][tr_idx])

    return {"train": tr, "val": va, "test": te}, class_names


def convert_oasis(name, spec):
    """OASIS: brain MRI slice -> patient Age (continuous regression target)."""
    repo = spec["repo"]
    base = MIRROR + "/datasets/%s/resolve/main/" % repo

    print("  downloading metadata parquet ...")
    meta = pq.read_table(io.BytesIO(_fetch(base + "data.parquet")))
    rows = meta.to_pylist()

    print("  downloading image archive ...")
    zdata = _fetch(base + "images-00000.zip")
    zf = zipfile.ZipFile(io.BytesIO(zdata))

    # index zip entries by the OAS subject id they belong to.
    # Filenames look like:
    #   images/train_NonDemented_OAS1_0002_MR1_1_nii_slice_129_png.rf.<hash>.jpg
    # The subject id is the "OAS1_0002" token pair.
    by_subject = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        stem = os.path.basename(info.filename)
        m = re.search(r"(OAS\d+_\d+)", stem)
        if not m:
            continue
        by_subject.setdefault(m.group(1), []).append(info)

    images, ages = [], []
    for r in rows:
        m = re.search(r"(OAS\d+_\d+)", str(r["ID"]))
        if not m:
            continue
        oas_id = m.group(1)
        age = r.get("Age")
        if age in (None, "", "None"):
            continue
        entries = by_subject.get(oas_id)
        if not entries:
            continue
        for info in entries:
            with zf.open(info) as f:
                img = Image.open(io.BytesIO(f.read())).convert("L")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            images.append(np.asarray(img, dtype=np.uint8))
            ages.append(float(age))

    X = np.stack(images)
    y = np.asarray(ages, dtype=np.float32)
    print("  OASIS slices=%d  age range=[%.0f, %.0f] nuniq=%d"
          % (len(X), y.min(), y.max(), len(np.unique(y))))

    tr_i, va_i, te_i = split_by_ratio(range(len(X)), max(1, len(X) // 10),
                                      max(1, len(X) // 10), seed=1)
    return {
        "train": (X[tr_i], y[tr_i]),
        "val": (X[va_i], y[va_i]),
        "test": (X[te_i], y[te_i]),
    }


# --------------------------------------------------------------------- driver


def read_class_names(parquet_path):
    """ClassLabel names from the parquet's huggingface metadata (may be None)."""
    try:
        t = pq.read_schema(parquet_path)
    except Exception:
        return None
    meta = t.metadata or {}
    raw = meta.get(b"huggingface")
    if not raw:
        return None
    try:
        features = json.loads(raw.decode("utf-8", "ignore"))["info"]["features"]
    except Exception:
        return None
    names = features.get("label", {}).get("names")
    return list(names) if names else None


def _norm(s):
    """Lowercase and drop separators so 'a_noDR' == 'noDR' and
    'no-tumor' == 'notumor'."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def canonical_label_map(raw_values, class_names, canonical_classes):
    """Map raw ClassLabel indices -> canonical class index.

    Some auto-converted repos expose one ClassLabel entry per *folder*, so the
    same semantic class can appear under several indices in different splits
    (e.g. "adenocarcinoma" vs "adenocarcinoma_left.lower.lobe_..."). We resolve
    each raw index to its class name and then to the canonical class by
    longest normalized-prefix match against `canonical_classes`.
    """
    if not class_names or not canonical_classes:
        return None
    norm_canon = [_norm(c) for c in canonical_classes]
    lut = {}
    for raw in sorted({v for v in raw_values}):
        try:
            name = class_names[int(raw)]
        except (IndexError, TypeError, ValueError):
            continue
        nname = _norm(name)
        best, best_score = None, -1
        for ci, nc in enumerate(norm_canon):
            if not nc:
                continue
            if nname == nc:
                score = 10_000 + len(nc)
            elif nname.startswith(nc):
                score = 1_000 + len(nc)
            elif len(nc) >= 3 and nname.endswith(nc):
                # e.g. "a_noDR" -> "noDR" (letter-prefixed folder names)
                score = len(nc)
            else:
                continue
            if score > best_score:
                best, best_score = ci, score
        if best is None:
            continue
        lut[raw] = best
    return lut or None


def finalize_labels(name, spec, splits, class_names=None):
    """Map raw labels to contiguous integers (or keep continuous for reg)."""
    if spec["task_type"] == "reg":
        return splits

    # One global LUT across ALL splits: encoding must be consistent between
    # train/val/test or the saved model would be evaluated on shifted labels.
    all_raw = [v for _imgs, labels in splits.values() for v in labels]
    label_map = spec.get("label_map")
    if label_map is None:
        label_map = canonical_label_map(
            all_raw, class_names, spec.get("classes", [])
        )
    if label_map is None:
        uniq = sorted({v for v in all_raw})
        label_map = {v: i for i, v in enumerate(uniq)}

    out = {}
    for split, (imgs, labels) in splits.items():
        missing = sorted({v for v in labels if v not in label_map})
        if missing:
            print("    WARNING unmapped labels in %s: %s" % (split, missing))
        enc = np.asarray([label_map.get(v, 0) for v in labels], dtype=np.int64)
        out[split] = (imgs, enc)
    return out


def download_one(name):
    spec = DATASETS[name]
    print("\n[%s] %s (%s)" % (name, spec["repo"], spec["task_type"]))
    t0 = time.time()
    class_names = None
    if spec.get("custom") == "oasis":
        splits = convert_oasis(name, spec)
    else:
        splits, class_names = convert_from_parquet(name, spec)
    splits = finalize_labels(name, spec, splits, class_names)

    os.makedirs(OUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUT_DIR, name + ".npz")
    payload = {}
    for split, (imgs, labels) in splits.items():
        payload[split + "_images"] = imgs
        payload[split + "_labels"] = labels
        print("    %-5s images=%-14s labels=%-12s %s" % (
            split, str(imgs.shape), str(labels.shape), labels.dtype))
    np.savez_compressed(npz_path, **payload)
    size_mb = os.path.getsize(npz_path) / 1e6
    print("  wrote %s (%.1f MB) in %.1fs" % (npz_path, size_mb, time.time() - t0))

    entry = {
        "task_type": spec["task_type"],
        "num_outputs": spec["num_outputs"],
        "is_3d": False,
        "npz": os.path.relpath(npz_path, ROOT).replace("\\", "/"),
        "source": spec["repo"],
        "homepage": spec.get("homepage", ""),
        "modality": spec.get("modality", ""),
        "classes": spec.get("classes", []),
        "target": spec.get("target", ""),
    }
    return entry


def write_registry(entries):
    registry = {}
    if os.path.exists(REGISTRY):
        with open(REGISTRY, encoding="utf-8") as f:
            registry = json.load(f)
    registry.update(entries)
    with open(REGISTRY, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    print("registry updated: %s (%d external datasets)"
          % (REGISTRY, len(registry)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="dataset names to download")
    ap.add_argument("--all", action="store_true", help="download every dataset")
    ap.add_argument("--list", action="store_true", help="list known datasets")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.list:
        for name, spec in DATASETS.items():
            print("%-28s %-8s %-10s %s" % (
                name, spec["task_type"], spec["num_outputs"], spec["repo"]))
        return

    names = list(DATASETS) if args.all else args.names
    if not names:
        ap.error("give dataset names, --all, or --list")
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        ap.error("unknown dataset(s): %s" % ", ".join(unknown))

    entries = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, n): n for n in names}
        for fut in cf.as_completed(futs):
            n = futs[fut]
            try:
                entries[n] = fut.result()
            except Exception as e:
                print("\n[%s] FAILED: %s: %s" % (n, type(e).__name__, str(e)[:200]))
    if entries:
        write_registry(entries)
    print("\ndone: %d/%d succeeded" % (len(entries), len(names)))


if __name__ == "__main__":
    main()
