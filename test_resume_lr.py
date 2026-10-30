"""Verify that a resumed run does not inherit lr=0 from a finished schedule.

Reproduces the exact resume sequence of train_medmnist.py / train_merged.py:
load the optimizer state saved at the end of a completed cosine run (whose
learning rate has annealed to 0), then build the new schedule.
"""
import torch


def build_resume_sched(optimizer, saved_state, total_steps, step_sched=True,
                        lr=0.001):
    """Mirrors the fixed resume path in the trainers."""
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["initial_lr"] = lr
    t_max = max(1, total_steps) if step_sched else max(1, 30)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    if not step_sched:
        for _ in range(10):
            sched.step()
    return sched


def main():
    # A finished 30-epoch run: the cosine schedule has annealed lr to 0.
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    for _ in range(30):
        sched.step()
    saved = opt.state_dict()
    print("finished run  : optimizer lr =", saved["param_groups"][0]["lr"])

    # Old behaviour: restore the optimizer state, then build a fresh
    # per-step schedule over the new budget. The restored param_groups carry
    # lr=0.0 (and initial_lr=0.001), and CosineAnnealingLR recurses
    # multiplicatively from the *current* group lr, so the zero propagates.
    m2 = torch.nn.Linear(4, 2)
    o2 = torch.optim.AdamW(m2.parameters(), lr=0.001)
    o2.load_state_dict(saved)
    s2 = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=2000)
    lrs = []
    for _ in range(2000):
        s2.step()
        lrs.append(o2.param_groups[0]["lr"])
    print("old behaviour : lr after resume =", lrs[0],
          "| after 200 steps =", lrs[199],
          "| all zero:", all(v == 0.0 for v in lrs))

    # Fixed behaviour.
    m3 = torch.nn.Linear(4, 2)
    o3 = torch.optim.AdamW(m3.parameters(), lr=0.001)
    o3.load_state_dict(saved)
    s3 = build_resume_sched(o3, None, 2000)
    lrs = []
    for _ in range(2000):
        s3.step()
        lrs.append(o3.param_groups[0]["lr"])
    print("fixed behaviour: lr after resume =", lrs[0],
          "| after 200 steps =", "%.3e" % lrs[199],
          "| after 2000 steps =", "%.3e" % lrs[-1])

    assert lrs[0] > 0, "resumed lr must be > 0"
    assert lrs[-1] < lrs[0], "resumed lr must decay"
    print("PASS: resumed schedule starts at a non-zero lr and decays")


if __name__ == "__main__":
    main()
