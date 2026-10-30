"""Train one multi-task model on the merged medical-imaging dataset.

The merged set (built by merge_datasets.py) holds every task in one tensor
with a shared image resolution. This script trains a single backbone with one
classification head per task, so the model can exploit the whole corpus at
once instead of 25 isolated models.

    python train_merged.py --epochs 10
    python train_merged.py --epochs 20 --tasks pneumoniamnist,organamnist
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = os.path.dirname(os.path.abspath(__file__))
MERGED = os.path.join(ROOT, "data", "merged", "merged.npz")
META = os.path.join(ROOT, "data", "merged", "merged_meta.json")
CKPT_DIR = os.path.join(ROOT, "checkpoints", "merged")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class MergedDataset(Dataset):
    """Holds one split of the merged corpus with per-task targets."""

    def __init__(self, images, task, cls, ml, reg, heads, augment=False,
                 target_dim=14, reg_stats=None):
        self.images = images
        self.task = task
        self.cls = cls
        self.ml = ml
        self.reg = reg
        self.heads = heads          # task id -> head type
        self.augment = augment
        self.target_dim = target_dim
        self.reg_stats = reg_stats or {}   # task id -> (mean, std)

    def __len__(self):
        return len(self.task)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255.0
        if self.augment and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])
        t = int(self.task[i])
        head = self.heads[t]
        # Every target is a fixed-width vector so the default collate can
        # stack a batch that mixes classification, multi-label and
        # regression tasks. Slot 0 carries the class id / regression value.
        y = torch.zeros(self.target_dim, dtype=torch.float32)
        if head == "cls":
            y[0] = float(self.cls[i])
        elif head == "multilabel":
            y[:] = torch.from_numpy(self.ml[i].astype(np.float32))
        else:
            # Regression targets are standardised with train-split statistics
            # so the MSE stays on the same scale as the other heads.
            mean, std = self.reg_stats.get(t, (0.0, 1.0))
            y[0] = (float(self.reg[i]) - mean) / std
        return x, t, head, y


def build_split(z, split, sel, remap):
    """Extract (and if needed subset + renumber) one split."""
    tarr = z[f"{split}_task"]
    present = np.unique(tarr)
    full = (len(sel) == len(present)
            and set(int(v) for v in present) == set(int(v) for v in sel))
    if full:
        return (z[f"{split}_images"], tarr.astype(np.int16),
                z[f"{split}_cls"], z[f"{split}_ml"], z[f"{split}_reg"])
    mask = np.isin(tarr, sel)
    idx = np.nonzero(mask)[0]
    return (z[f"{split}_images"][idx],
            np.array([remap[int(v)] for v in tarr[idx]], dtype=np.int16),
            z[f"{split}_cls"][idx], z[f"{split}_ml"][idx], z[f"{split}_reg"][idx])


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def build_trunk(width, in_ch=3):
    def block(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    layers = [
        nn.Conv2d(in_ch, width, 3, 1, 1, bias=False),
        nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        block(width, width * 2, 2),
        block(width * 2, width * 2),
        block(width * 2, width * 4, 2),
        block(width * 4, width * 4),
        block(width * 4, width * 8, 2),
        block(width * 8, width * 8),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ]
    return nn.Sequential(*layers), width * 8


class MultiTaskNet(nn.Module):
    """Shared CNN trunk plus one linear head per task."""

    def __init__(self, tasks, width=32):
        super().__init__()
        self.trunk, feat = build_trunk(width)
        self.drop = nn.Dropout(0.2)
        self.heads = nn.ModuleDict({
            str(i): nn.Linear(feat, t["num_outputs"]) for i, t in enumerate(tasks)
        })
        self.head_type = {i: t["head"] for i, t in enumerate(tasks)}

    def forward(self, x, task_ids):
        """Returns (task_id, logits) per unique task, with logits already
        restricted to the rows of that task."""
        feat = self.drop(self.trunk(x).flatten(1))
        out = []
        for t in task_ids.unique().tolist():
            mask = task_ids == int(t)
            out.append((int(t), self.heads[str(int(t))](feat[mask])))
        return out


# --------------------------------------------------------------------------
# loss / metrics
# --------------------------------------------------------------------------

def batch_loss(logits_by_task, task_ids, heads, ys):
    """Per-task loss, weighted by each task's share of the batch."""
    total = 0.0
    parts = []
    for t, logits in logits_by_task:
        mask = task_ids == t
        head = heads[int(t)]
        if head == "cls":
            l = F.cross_entropy(logits, ys[mask][:, 0].long(), reduction="mean")
        elif head == "multilabel":
            l = F.binary_cross_entropy_with_logits(logits, ys[mask], reduction="mean")
        else:
            l = F.mse_loss(logits.reshape(-1), ys[mask][:, 0], reduction="mean")
        total = total + mask.float().mean().item() * l
        parts.append((int(t), float(l.item()), int(mask.sum().item())))
    return total, parts


def auc_macro(y_true, y_score):
    """Macro-averaged AUC from rank statistics (no sklearn dependency)."""
    y_true = np.asarray(y_true).reshape(len(y_true), -1)
    y_score = np.asarray(y_score).reshape(len(y_score), -1)
    aucs = []
    for c in range(y_true.shape[1]):
        t, s = y_true[:, c], y_score[:, c]
        n_pos = float((t > 0.5).sum())
        n_neg = float((t <= 0.5).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(len(s), dtype=np.float64)
        ss = s[order]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and ss[j + 1] == ss[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        aucs.append((ranks[t > 0.5].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    return float(np.mean(aucs)) if aucs else float("nan")


@torch.no_grad()
def evaluate(model, loader, heads, reg_stats=None):
    """Per-task metrics over a split."""
    reg_stats = reg_stats or {}
    model.eval()
    preds, targets = {}, {}
    for x, task_ids, _hl, ys in loader:
        x = x.cuda(non_blocking=True)
        for t, logits in model(x, task_ids):
            preds.setdefault(t, []).append(logits.cpu().numpy())
            targets.setdefault(t, []).append(ys[task_ids == t].numpy())
    out = {}
    for t in preds:
        p = np.concatenate(preds[t])
        y = np.concatenate(targets[t])
        if heads[t] == "cls":
            out[t] = {"metric": "ACC",
                      "value": float((p.argmax(1) == y[:, 0].astype(np.int64)).mean())}
        elif heads[t] == "multilabel":
            out[t] = {"metric": "AUC", "value": auc_macro(y, p)}
        else:
            mean, std = reg_stats.get(t, (0.0, 1.0))
            pred = p.reshape(-1) * std + mean
            truth = y[:, 0] * std + mean
            out[t] = {"metric": "RMSE",
                      "value": float(np.sqrt(np.mean((pred - truth) ** 2)))}
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--balanced", action="store_true", default=True,
                    help="sample tasks uniformly instead of proportionally")
    ap.add_argument("--no-balanced", dest="balanced", action="store_false")
    ap.add_argument("--tasks", default="", help="comma-separated task subset")
    ap.add_argument("--iters", type=int, default=0,
                    help="total optimizer steps for this run; overrides --epochs")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="evaluate every N steps (0 = auto, ~10 points per run)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from checkpoints/merged/last.pt")
    ap.add_argument("--ckpt-dir", default="",
                    help="override the checkpoint directory (default checkpoints/merged)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(MERGED):
        raise SystemExit("merged dataset not found; run merge_datasets.py first")
    with open(META, encoding="utf-8") as f:
        meta = json.load(f)

    all_tasks = meta["tasks"]
    if args.tasks:
        keep = {t.strip() for t in args.tasks.split(",") if t.strip()}
        sel = [i for i, t in enumerate(all_tasks) if t["task"] in keep]
    else:
        sel = list(range(len(all_tasks)))
    remap = {old: new for new, old in enumerate(sel)}
    tasks = [all_tasks[i] for i in sel]
    heads = {i: t["head"] for i, t in enumerate(tasks)}
    print("training on %d/%d tasks" % (len(tasks), len(all_tasks)))

    z = np.load(MERGED, allow_pickle=True)
    parts = {}
    for sp in ("train", "val", "test"):
        parts[sp] = build_split(z, sp, sel, remap)

    # Standardise the regression targets with train-split statistics only.
    reg_stats = {}
    for i, t in enumerate(tasks):
        if t["head"] != "reg":
            continue
        vals = parts["train"][4][parts["train"][1] == i].astype(np.float64)
        mean = float(vals.mean())
        std = float(vals.std()) or 1.0
        reg_stats[i] = (mean, std)
        print("reg target %s: mean=%.2f std=%.2f" % (t["task"], mean, std))

    tr = MergedDataset(*parts["train"], heads, augment=True, reg_stats=reg_stats)
    va = MergedDataset(*parts["val"], heads, reg_stats=reg_stats)
    te = MergedDataset(*parts["test"], heads, reg_stats=reg_stats)
    print("merged dataset: train=%d val=%d test=%d, %dx%d"
          % (len(tr), len(va), len(te), meta["size"], meta["size"]))

    if args.balanced:
        counts = np.bincount(tr.task, minlength=len(tasks)).astype(np.float64)
        w = np.zeros(len(tr.task), dtype=np.float64)
        for i in range(len(tasks)):
            if counts[i] > 0:
                w[tr.task == i] = 1.0 / counts[i]
        sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=len(tr),
                                        replacement=True)
        train_loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=0, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(va, batch_size=256, num_workers=0)
    test_loader = DataLoader(te, batch_size=256, num_workers=0)

    device = torch.device(args.device)
    model = MultiTaskNet(tasks, width=args.width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("params %.2fM, device %s" % (n_params / 1e6, device))

    ckpt_dir = args.ckpt_dir or CKPT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")

    # ---------------------------------------------------------------- resume
    # A resumed run continues from the last checkpoint instead of starting
    # over, so extra budget is added on top of an existing model rather than
    # replacing it.
    start_round = 1
    start_step = 0
    history = []
    best = -1.0
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if "optimizer_state" in ck:
            opt.load_state_dict(ck["optimizer_state"])
        history = list(ck.get("history") or [])
        start_round = int(ck.get("round", ck.get("epoch", 0))) + 1
        start_step = int(ck.get("step", 0))
        best = float(ck.get("best", -1.0))
        print("resumed from %s (round %d, step %d, best=%.4f)"
              % (last_path, start_round - 1, start_step, best), flush=True)
        # best.pt may hold a stronger model than last.pt: keep its metric so a
        # weaker resumed run cannot overwrite the recorded best.
        if os.path.exists(best_path):
            bck = torch.load(best_path, map_location=device, weights_only=False)
            bvm = bck.get("val") or {}
            bmean = float(np.mean([v["value"] for v in bvm.values()
                                   if v["metric"] == "ACC"])) if bvm else -1.0
            best = max(best, bmean)
    elif args.resume:
        print("--resume requested but %s does not exist; starting fresh"
              % last_path, flush=True)

    steps_per_epoch = max(1, len(train_loader))
    step_sched = args.iters > 0
    if args.iters > 0:
        # Step budget: this run spends exactly --iters optimizer steps and
        # the learning rate decays across that whole budget.
        total_steps = args.iters
        eval_every = args.eval_every if args.eval_every > 0 else max(1, total_steps // 10)
    else:
        # Epoch mode: --epochs counts the total number of rounds, so a
        # resumed run keeps training until it reaches that round number.
        rounds_left = max(0, args.epochs - (start_round - 1))
        total_steps = rounds_left * steps_per_epoch
        eval_every = args.eval_every if args.eval_every > 0 else steps_per_epoch

    # A resumed optimizer carries the *previous* run's learning rate, which is
    # 0.0 once its cosine schedule has annealed to the end. CosineAnnealingLR
    # updates the rate multiplicatively from the current value, so a restored
    # lr of 0 would pin this run at 0 and the model would never train. Re-seed
    # the rate from --lr before building the new schedule.
    for group in opt.param_groups:
        group["lr"] = args.lr
        group["initial_lr"] = args.lr

    # The schedule is rebuilt from this run's budget instead of being restored
    # from the checkpoint: the saved state carries the previous run's T_max and
    # last_epoch, which no longer match the budget being spent now.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, total_steps) if step_sched else max(1, args.epochs))
    if not step_sched and start_round > 1:
        # Continue the per-epoch cosine from the round we resume at.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_round - 1):
                sched.step()
    print("plan: %d steps (%.1f epochs), eval every %d steps, lr=%.2e (%s schedule)"
          % (total_steps, total_steps / steps_per_epoch, eval_every,
             opt.param_groups[0]["lr"],
             "per-step" if step_sched else "per-epoch"))

    def loader_stream():
        """Cycle the training loader forever so a step budget can be spent."""
        while True:
            for batch in train_loader:
                yield batch

    data_iter = loader_stream()
    step = 0
    round_no = start_round
    run, nb = 0.0, 0
    t0 = time.time()
    model.train()
    while step < total_steps:
        x, task_ids, _hl, ys = next(data_iter)
        x = x.to(device, non_blocking=True)
        task_ids = task_ids.to(device)
        ys = ys.to(device)
        opt.zero_grad(set_to_none=True)
        loss, _ = batch_loss(model(x, task_ids), task_ids, heads, ys)
        loss.backward()
        opt.step()
        if step_sched:
            sched.step()
        run += float(loss.item())
        nb += 1
        step += 1

        at_epoch_end = (not step_sched) and (step % steps_per_epoch == 0)
        if at_epoch_end:
            sched.step()
        if step % eval_every == 0 or step == total_steps:
            vm = evaluate(model, val_loader, heads, reg_stats)
            mean_cls = float(np.mean([v["value"] for v in vm.values()
                                      if v["metric"] == "ACC"])) if vm else 0.0
            history.append({"epoch": round_no, "step": step + start_step,
                            "train_loss": run / max(nb, 1),
                            "val_mean_cls_acc": mean_cls,
                            "val_mean_ACC": mean_cls,
                            "secs": round(time.time() - t0, 1)})
            print("step %5d (this run %d/%d, round %d) loss=%.4f "
                  "val_mean_cls_ACC=%.4f (%.1fs)"
                  % (step + start_step, step, total_steps, round_no,
                     run / max(nb, 1), mean_cls, time.time() - t0), flush=True)
            payload = {"epoch": round_no, "round": round_no, "step": step + start_step,
                       "model_state": model.state_dict(),
                       "optimizer_state": opt.state_dict(),
                       "sched_state": sched.state_dict(),
                       "tasks": tasks, "history": history, "val": vm,
                       "best": max(best, mean_cls)}
            torch.save(payload, last_path)
            if mean_cls > best:
                best = mean_cls
                torch.save(payload, best_path)
            round_no += 1
            run, nb = 0.0, 0
            t0 = time.time()
            model.train()

    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    tm = evaluate(model, test_loader, heads, reg_stats)
    vm = ck["val"]

    rows = []
    for i, t in enumerate(tasks):
        v = vm.get(i, {})
        s = tm.get(i, {})
        rows.append({
            "task": t["task"], "head": t["head"],
            "metric": s.get("metric", v.get("metric")),
            "val": v.get("value"), "test": s.get("value"),
            "n_test": t["counts"]["test"],
        })
    cls_vals = [r["test"] for r in rows
                if r["metric"] == "ACC" and r["test"] is not None]
    summary = {
        "task": "merged",
        "task_type": "multitask",
        "metric_name": "mean_ACC",
        "best_val": best,
        "test": float(np.mean(cls_vals)) if cls_vals else None,
        "epochs": len(history),
        "total_steps": step + start_step,
        "params": n_params,
        "best_epoch": ck["epoch"],
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "balanced": args.balanced,
        "per_task": rows,
        "history": history,
    }
    with open(os.path.join(ckpt_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n%-30s %-7s %-6s %-9s %-9s" % ("task", "metric", "head", "val", "test"))
    for r in rows:
        print("%-30s %-7s %-6s %-9s %-9s" % (
            r["task"], r["metric"], r["head"],
            "%.4f" % r["val"] if r["val"] is not None else "-",
            "%.4f" % r["test"] if r["test"] is not None else "-"))
    print("\nmean ACC over %d classification tasks: %.4f"
          % (len(cls_vals), summary["test"]))
    print("checkpoints written to " + ckpt_dir)


if __name__ == "__main__":
    main()
