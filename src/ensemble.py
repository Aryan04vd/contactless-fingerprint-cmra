# src/ensemble.py
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate import compute_eer, tar_at_far

DATA = Path(__file__).resolve().parent / "data"
RUNS = ["base_seed1", "base_seed2", "base_seed3"]

mm_val = pd.read_csv(DATA / "scores_val.csv")[["label", "score_mm"]]
mm_test = pd.read_csv(DATA / "scores_test.csv")[["label", "score_mm"]]
groups = pd.read_csv(DATA / "pairs_test.csv").finger_a.values     # resample by finger

dl = {}
for split, ref in [("val", mm_val), ("test", mm_test)]:
    mats = []
    for r in RUNS:
        d = pd.read_csv(DATA / "runs" / r / f"scores_{split}_dl.csv")
        assert (d.label.values == ref.label.values).all(), "pair order mismatch"
        mats.append(d.score_dl.values)
    dl[split] = np.mean(mats, axis=0)                    # average the 3 models

nrm = lambda v: (lambda x: (x - v.min()) / (v.max() - v.min() + 1e-9))
nd, nm = nrm(dl["val"]), nrm(mm_val.score_mm.values)
fuse = lambda a, b, w: w * nd(a) + (1 - w) * nm(b)
ws = np.round(np.arange(0, 1.01, 0.1), 1)
w = min(ws, key=lambda w: compute_eer(mm_val.label.values, fuse(dl["val"], mm_val.score_mm.values, w)))
print("fusion weight chosen on validation: w =", w)


def boot(y, s, n=200, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ids}
    out = []
    for _ in range(n):
        idx = np.concatenate([by[g] for g in rng.choice(ids, len(ids), replace=True)])
        out.append(compute_eer(y[idx], s[idx]))
    return np.percentile(out, [2.5, 97.5])


y = mm_test.label.values
rows = [("Minutiae only", mm_test.score_mm.values),
        ("Wavelet+Siamese (3-model avg)", dl["test"]),
        (f"Fusion (w={w})", fuse(dl["test"], mm_test.score_mm.values, w))]
print(f"\n{'method':32s} {'EER':>6s}  {'95% range':>14s}  {'TAR@FAR=1%':>10s}")
for name, s in rows:
    lo, hi = boot(y, s)
    print(f"{name:32s} {compute_eer(y, s):6.3f}  [{lo:.3f}, {hi:.3f}]  {tar_at_far(y, s):10.3f}")