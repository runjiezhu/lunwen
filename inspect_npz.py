import glob
import os

import numpy as np

for p in sorted(glob.glob("data/medmnist/*.npz")):
    t = os.path.basename(p)[:-4]
    try:
        d = np.load(p)
        tr = d["train_labels"]
        va = d["val_labels"]
        te = d["test_labels"]
        u = np.unique(tr)
        vals = u[:6].tolist() if len(u) <= 8 else u[:4].tolist() + ["..."]
        print(
            "%-18s imgs=%-22s lbl=%-8s %-8s nuniq=%-5d vals=%s  test_lbl=%s"
            % (
                t,
                str(d["train_images"].shape),
                str(tr.shape),
                str(tr.dtype),
                len(u),
                vals,
                str(te.shape),
            )
        )
    except Exception as e:
        print("%-18s ERR %s: %s" % (t, type(e).__name__, str(e)[:70]))
