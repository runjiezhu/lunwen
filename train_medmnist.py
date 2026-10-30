"""MedMNIST + external medical-imaging training on local GPU.

Trains a ResNet-style CNN from scratch and writes checkpoints under
checkpoints/<task>/{last.pt,best.pt,summary.json}.

Task types
----------
cls        single-label classification   -> CrossEntropyLoss, metric ACC
multilabel multi-label classification    -> BCEWithLogitsLoss, metric AUC-macro
reg        regression (continuous label) -> MSELoss, metric RMSE

Datasets are stored as MedMNIST-style .npz archives:
    train_images / val_images / test_images  uint8 arrays
    train_labels / val_labels / test_labels  label arrays

A dataset is described in datasets_registry.json so that both the built-in
MedMNIST subsets and externally downloaded datasets can be trained with the
same entry point.

Usage:
    python train_medmnist.py --task pneumoniamnist --epochs 30
    python train_medmnist.py --task synapsemnist3d --epochs 30
    python train_medmnist.py --task brain-tumor-mri --epochs 20 --batch-size 32
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "medmnist")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
REGISTRY_PATH = os.path.join(ROOT, "datasets_registry.json")

# Built-in MedMNIST v2 subsets: task_type -> num_outputs.
#   cls        -> num_outputs = number of classes
#   multilabel -> num_outputs = number of labels
#   reg        -> num_outputs = 1 (continuous target)
BUILTIN_TASKS = {
    "pneumoniamnist": ("cls", 2, False),
    "retinamnist": ("cls", 5, False),
    "dermamnist": ("cls", 7, False),
    "organamnist": ("cls", 11, False),
    "organcmnist": ("cls", 11, False),
    "organsmnist": ("cls", 11, False),
    "bloodmnist": ("cls", 8, False),
    "breastmnist": ("cls", 2, False),
    "octmnist": ("cls", 4, False),
    "pathmnist": ("cls", 9, False),
    "tissuemnist": ("cls", 8, False),
    "chestmnist": ("multilabel", 14, False),
    "adrenalmnist3d": ("cls", 2, True),
    "fracturemnist3d": ("cls", 3, True),
    "nodulemnist3d": ("cls", 2, True),
    "organmnist3d": ("cls", 11, True),
    # NOTE: despite the name, SynapseMNIST3D is officially a *binary-class*
    # task (0 = inhibitory synapse, 1 = excitatory synapse), NOT a regression
    # task. Verified against medmnist INFO["synapsemnist3d"]["task"].
    "synapsemnist3d": ("cls", 2, True),
    "vesselmnist3d": ("cls", 2, True),
}


def load_registry():
    """Merge built-in MedMNIST tasks with the external dataset registry.

    External .npz files under data/external/ are auto-discovered so a dataset
    that has been downloaded but not yet written into datasets_registry.json is
    still trainable. Task metadata (task_type / num_outputs) is taken from the
    registry when available and otherwise inferred from the label array.
    """
    tasks = {}
    for name, (task_type, num_outputs, is_3d) in BUILTIN_TASKS.items():
        tasks[name] = {
            "task_type": task_type,
            "num_outputs": num_outputs,
            "is_3d": is_3d,
            "npz": os.path.join(DATA_DIR, f"{name}.npz"),
            "source": "MedMNIST v2",
        }

    external = {}
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            external = json.load(f)

    ext_dir = os.path.join(ROOT, "data", "external")
    if os.path.isdir(ext_dir):
        for fn in sorted(os.listdir(ext_dir)):
            if not fn.endswith(".npz"):
                continue
            name = fn[:-4]
            if name in external:
                continue
            external[name] = {
                "task_type": None,  # inferred below
                "num_outputs": 0,  # inferred below
                "is_3d": False,
                "npz": os.path.join("data", "external", fn),
                "source": "auto-discovered",
            }

    for name, spec in external.items():
        spec = dict(spec)
        npz = spec.get("npz")
        if npz and not os.path.isabs(npz):
            spec["npz"] = os.path.join(ROOT, npz)
        if spec.get("task_type") is None:
            spec["task_type"] = _infer_task_type(spec["npz"])
        spec.setdefault("is_3d", False)
        if not spec.get("num_outputs"):
            spec["num_outputs"] = _infer_num_outputs(spec["npz"], spec["task_type"])
        tasks[name] = spec
    return tasks


def _infer_task_type(npz_path):
    """A float label array means a continuous (regression) target."""
    try:
        with np.load(npz_path) as d:
            labels = d["train_labels"]
    except Exception:
        return "cls"
    if labels.dtype.kind == "f":
        return "reg"
    return "cls"


def _infer_num_outputs(npz_path, task_type):
    """Derive the output dimension from the label array."""
    if task_type == "reg":
        return 1
    try:
        with np.load(npz_path) as d:
            labels = d["train_labels"]
    except Exception:
        return 2
    if labels.ndim > 1 and labels.shape[1] > 1:
        return int(labels.shape[1])
    return int(np.unique(labels).max()) + 1


def image_channels(shape):
    """Number of channels the model receives for a raw image array shape.

    Mirrors the layout logic in MedDataset.__getitem__ so that the first
    convolution is built with the same channel count the data loader produces.
    """
    if len(shape) == 2:  # (H, W) grayscale
        return 1
    if len(shape) == 3 and shape[-1] in (1, 3):  # (H, W, C) -> permuted, padded to 3
        return 3
    if len(shape) == 3:  # (D, H, W) 3D volume slice
        return 1
    if len(shape) == 4:  # (D, H, W, C) -> permuted, padded to 3
        return 3
    raise ValueError("unsupported image shape %r" % (shape,))


class MedDataset(Dataset):
    def __init__(self, images, labels, task_type, augment, target_norm=None):
        self.images = images
        self.labels = labels
        self.task_type = task_type
        self.augment = augment
        # For regression we standardise the target so MSE is well conditioned;
        # predictions are mapped back to the original units at evaluation time.
        self.target_norm = target_norm

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        arr = np.asarray(self.images[idx])
        if arr.ndim == 2:  # (H, W) grayscale
            x = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0)
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3):  # (H, W, C) RGB
            x = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(2, 0, 1)
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
        elif arr.ndim == 3:  # (D, H, W) 3D volume slice
            x = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0)
        elif arr.ndim == 4:  # (D, H, W, C) 3D volume with channels
            x = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(3, 0, 1, 2)
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1, 1)
        else:
            raise ValueError(f"unsupported image shape {arr.shape}")
        x = x / 255.0
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[-1])
        lab = np.asarray(self.labels[idx])
        if self.task_type == "multilabel":
            y = torch.from_numpy(np.ascontiguousarray(lab)).float().reshape(-1)
        elif self.task_type == "reg":
            # Keep one scalar per sample: np.ascontiguousarray on a 0-d array
            # returns shape (1,), which would collate to (B, 1) and broadcast
            # against the (B,) predictions.
            y = torch.tensor(float(lab), dtype=torch.float32)
            if self.target_norm is not None:
                mean, std = self.target_norm
                y = (y - mean) / std
        else:
            y = torch.from_numpy(np.ascontiguousarray(lab)).long().reshape(-1)[0]
        return x, y


def build_model(num_outputs, is_3d, width=32, in_channels=1):
    """ResNet-style CNN, following the MedMNIST benchmark baseline topology.

    ``in_channels`` must match the channel count produced by MedDataset
    (3 for the RGB MedMNIST subsets, 1 for grayscale and 3D volumes).
    """

    def block2d(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def block3d(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
        )

    if is_3d:
        conv, blk, adaptive = nn.Conv3d, block3d, nn.AdaptiveAvgPool3d(1)
    else:
        conv, blk, adaptive = nn.Conv2d, block2d, nn.AdaptiveAvgPool2d(1)

    layers = [
        conv(in_channels, width, 3, 1, 1, bias=False),
        nn.BatchNorm3d(width) if is_3d else nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
        blk(width, width * 2, 2),
        blk(width * 2, width * 2),
        blk(width * 2, width * 4, 2),
        blk(width * 4, width * 4),
        blk(width * 4, width * 8, 2),
        blk(width * 8, width * 8),
    ]
    feat_dim = width * 8

    head = nn.Sequential(
        adaptive,
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(feat_dim, num_outputs),
    )
    return nn.Sequential(nn.Sequential(*layers), head)


def _auc_macro(y_true, y_score):
    """Macro-averaged AUC computed from rank statistics (no sklearn needed)."""
    y_true = np.asarray(y_true).reshape(len(y_true), -1)
    y_score = np.asarray(y_score).reshape(len(y_score), -1)
    aucs = []
    for c in range(y_true.shape[1]):
        t = y_true[:, c]
        s = y_score[:, c]
        n_pos = float((t > 0.5).sum())
        n_neg = float((t <= 0.5).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(len(s), dtype=np.float64)
        sorted_s = s[order]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            ranks[order[i:j + 1]] = avg_rank
            i = j + 1
        sum_pos = ranks[t > 0.5].sum()
        aucs.append((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    return float(np.mean(aucs)) if aucs else float("nan")


@torch.no_grad()
def evaluate(model, loader, device, task_type, target_norm=None):
    model.eval()
    total_loss, total_n = 0.0, 0
    preds, targets = [], []

    if task_type == "multilabel":
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out, y)
            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)
            preds.append(torch.sigmoid(out).cpu().numpy())
            targets.append(y.cpu().numpy())
        metric = _auc_macro(np.concatenate(targets), np.concatenate(preds))
        return total_loss / max(total_n, 1), metric, "AUC", float("nan")

    if task_type == "reg":
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x).reshape(-1)
            loss = F.mse_loss(out, y)
            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)
            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())
        p = np.concatenate(preds)
        t = np.concatenate(targets)
        if target_norm is not None:
            mean, std = target_norm
            p = p * std + mean
            t = t * std + mean
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae = float(np.mean(np.abs(p - t)))
        return total_loss / max(total_n, 1), rmse, "RMSE", mae

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = F.cross_entropy(out, y)
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        preds.append(out.argmax(1).cpu().numpy())
        targets.append(y.cpu().numpy())
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    return total_loss / max(total_n, 1), float((p == t).mean()), "ACC", float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=0,
                    help="total optimizer steps for this run; overrides --epochs")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="evaluate every N steps (0 = auto)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from checkpoints/<task>/last.pt")
    ap.add_argument("--ckpt-dir", default="",
                    help="override the checkpoint directory (default checkpoints/<task>)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tasks = load_registry()
    if args.task not in tasks:
        raise SystemExit(
            f"unknown task '{args.task}'. available: {', '.join(sorted(tasks))}"
        )
    spec = tasks[args.task]
    task_type = spec["task_type"]
    num_outputs = spec["num_outputs"]
    is_3d = spec["is_3d"]
    npz_path = spec["npz"]
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"dataset not found: {npz_path}")

    data = np.load(npz_path)

    # The first convolution must match the channel count the loader produces.
    # data["train_images"].shape includes the batch axis, so inspect one sample.
    in_channels = image_channels(data["train_images"].shape[1:])

    # Regression: standardise the target using train statistics only.
    target_norm = None
    if task_type == "reg":
        tr_labels = np.asarray(data["train_labels"], dtype=np.float64).reshape(-1)
        mean = float(tr_labels.mean())
        std = float(tr_labels.std())
        if std < 1e-8:
            std = 1.0
        target_norm = (mean, std)
        print(f"target normalisation: mean={mean:.3f} std={std:.3f}")

    train_ds = MedDataset(
        data["train_images"], data["train_labels"], task_type, augment=True,
        target_norm=target_norm,
    )
    val_ds = MedDataset(
        data["val_images"], data["val_labels"], task_type, augment=False,
        target_norm=target_norm,
    )
    test_ds = MedDataset(
        data["test_images"], data["test_labels"], task_type, augment=False,
        target_norm=target_norm,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        drop_last=len(train_ds) > args.batch_size, pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=256, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=0)

    device = torch.device(args.device)
    model = build_model(num_outputs, is_3d, args.width, in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if task_type == "multilabel":
        criterion = nn.BCEWithLogitsLoss()
    elif task_type == "reg":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    out_dir = args.ckpt_dir or os.path.join(CKPT_DIR, args.task)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "last.pt")
    best_path = os.path.join(out_dir, "best.pt")

    print(f"task={args.task} type={task_type} outputs={num_outputs} 3d={is_3d} "
          f"in_ch={in_channels} params={n_params/1e6:.2f}M")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} device={device}")

    # Lower is better for RMSE, higher is better for ACC/AUC.
    lower_is_better = (task_type == "reg")

    best_metric = float("inf") if lower_is_better else -1.0
    history = []
    start_epoch = 1
    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        history = list(ck.get("history") or [])
        start_epoch = int(ck.get("epoch", 0)) + 1
        start_step = int(ck.get("step", 0))
        best_metric = float(ck.get("best_metric", best_metric))
        print(f"resumed from {ckpt_path} (epoch {start_epoch - 1}, "
              f"step {start_step})", flush=True)
        # best.pt may hold a stronger model than last.pt: keep its metric so a
        # weaker resumed run cannot overwrite the recorded best.
        if os.path.exists(best_path):
            bck = torch.load(best_path, map_location=device, weights_only=False)
            bm = bck.get("metric_name")
            if bm is not None and ("val_" + bm) in bck:
                v = float(bck["val_" + bm])
                best_metric = (min(best_metric, v) if lower_is_better
                               else max(best_metric, v))

    steps_per_epoch = max(1, len(train_loader))
    step_sched = args.iters > 0
    if step_sched:
        # Step budget: this run spends exactly --iters optimizer steps and the
        # learning rate decays across that whole budget.
        total_steps = args.iters
        eval_every = args.eval_every if args.eval_every > 0 else max(1, total_steps // 10)
    else:
        # Epoch mode: --epochs counts the total number of rounds, so a resumed
        # run keeps training until it reaches that round number.
        total_steps = max(0, args.epochs - (start_epoch - 1)) * steps_per_epoch
        eval_every = args.eval_every if args.eval_every > 0 else steps_per_epoch

    # A resumed optimizer carries the *previous* run's learning rate, which is
    # 0.0 once its cosine schedule has annealed to the end. CosineAnnealingLR
    # updates the rate multiplicatively from the current value, so a restored
    # lr of 0 would pin this run at 0 and the model would never train. Re-seed
    # the rate from --lr before building the new schedule.
    for group in optimizer.param_groups:
        group["lr"] = args.lr
        group["initial_lr"] = args.lr

    # The schedule is rebuilt from this run's budget instead of being restored
    # from the checkpoint: the saved state carries the previous run's T_max and
    # last_epoch, which no longer match the budget being spent now.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps) if step_sched else max(1, args.epochs))
    if not step_sched and start_epoch > 1:
        # Continue the per-epoch cosine from the round we resume at.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_epoch - 1):
                scheduler.step()
    print(f"plan: {total_steps} steps ({total_steps / steps_per_epoch:.1f} epochs), "
          f"eval every {eval_every} steps, lr={optimizer.param_groups[0]['lr']:.2e} "
          f"({'per-step' if step_sched else 'per-epoch'} schedule)", flush=True)

    def loader_stream():
        """Cycle the training loader forever so a step budget can be spent."""
        while True:
            for batch in train_loader:
                yield batch

    data_iter = loader_stream()
    step = 0
    epoch = start_epoch
    running, seen = 0.0, 0
    t0 = time.time()
    model.train()
    while step < total_steps:
        x, y = next(data_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        if task_type == "reg":
            # model(x) is (B, 1) but the target is (B,); without this the
            # loss broadcasts to (B, B) pairwise errors and the gradients
            # are meaningless.
            out = out.reshape(-1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if step_sched:
            scheduler.step()
        running += loss.item() * x.size(0)
        seen += x.size(0)
        step += 1

        at_epoch_end = (not step_sched) and (step % steps_per_epoch == 0)
        if at_epoch_end:
            scheduler.step()
        if step % eval_every == 0 or step == total_steps:
            train_loss = running / max(seen, 1)
            val_loss, val_metric, metric_name, val_mae = evaluate(
                model, val_loader, device, task_type, target_norm
            )
            history.append({
                "epoch": epoch,
                "step": step + start_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                f"val_{metric_name}": val_metric,
                "lr": scheduler.get_last_lr()[0],
                "secs": round(time.time() - t0, 1),
            })
            extra = f" val_MAE={val_mae:.4f}" if val_mae == val_mae else ""
            print(f"epoch {epoch:3d} step {step + start_step} "
                  f"(this run {step}/{total_steps}) "
                  f"train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_{metric_name}={val_metric:.4f}{extra} "
                  f"({time.time()-t0:.1f}s)")

            payload = {
                "epoch": epoch,
                "step": step + start_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "task": args.task,
                "task_type": task_type,
                "num_outputs": num_outputs,
                "is_3d": is_3d,
                "width": args.width,
                "metric_name": metric_name,
                "target_norm": target_norm,
                "val_" + metric_name: val_metric,
                "best_metric": min(best_metric, val_metric) if lower_is_better
                else max(best_metric, val_metric),
                "history": history,
            }
            torch.save(payload, ckpt_path)
            improved = (val_metric < best_metric) if lower_is_better \
                else (val_metric > best_metric)
            if improved:
                best_metric = val_metric
                torch.save(payload, best_path)
            epoch += 1
            running, seen = 0.0, 0
            t0 = time.time()
            model.train()

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_metric, metric_name, test_mae = evaluate(
        model, test_loader, device, task_type, target_norm
    )
    extra = f" test_MAE={test_mae:.4f}" if test_mae == test_mae else ""
    print(f"\nBEST (epoch {ckpt['epoch']}) val_{metric_name}={best_metric:.4f} -> "
          f"test_{metric_name}={test_metric:.4f}{extra} (loss {test_loss:.4f})")

    summary = {
        "task": args.task,
        "task_type": task_type,
        "num_outputs": num_outputs,
        "source": spec.get("source", ""),
        "params": n_params,
        "epochs": len(history),
        "total_steps": step + start_step,
        "metric_name": metric_name,
        "best_epoch": ckpt["epoch"],
        "best_val": best_metric,
        "test": test_metric,
        "test_mae": test_mae if test_mae == test_mae else None,
        "test_loss": test_loss,
        "best_checkpoint": best_path,
        "last_checkpoint": ckpt_path,
        "history": history,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"checkpoints written to {out_dir}")


if __name__ == "__main__":
    main()
