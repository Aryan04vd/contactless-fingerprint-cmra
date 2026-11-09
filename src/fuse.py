# src/fuse.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from evaluate import compute_eer, tar_at_far

DATA = Path(__file__).resolve().parent / "data"
val = pd.read_csv(DATA / "scores_val.csv")
test = pd.read_csv(DATA / "scores_test.csv")


def norm(col):                       # min-max using VALIDATION range only
    lo, hi = val[col].min(), val[col].max()
    return lambda x: (x - lo) / (hi - lo + 1e-9)

nd, nm = norm("score_dl"), norm("score_mm")
fuse = lambda df, w: w * nd(df.score_dl) + (1 - w) * nm(df.score_mm)

print("weight on deep learning -> VAL EER")
res = {}
for w in np.round(np.arange(0, 1.01, 0.1), 1):
    res[w] = compute_eer(val.label, fuse(val, w))
    print(f"  w = {w:.1f}   EER {res[w]:.3f}")
w_best = min(res, key=res.get)
print(f"\nbest weight chosen on validation: w = {w_best}")

print("\n=== TEST RESULTS (final numbers for your report) ===")
rows = [("Minutiae only", test.score_mm), ("Wavelet+Siamese only", test.score_dl),
        (f"Fusion (w={w_best})", fuse(test, w_best))]
for name, s in rows:
    print(f"{name:24s} EER {compute_eer(test.label, s):.3f}   "
          f"TAR@FAR=1% {tar_at_far(test.label, s):.3f}")

plt.figure(figsize=(5, 5))
for name, s in rows:
    fpr, tpr, _ = roc_curve(test.label, s)
    plt.plot(fpr, tpr, label=name)
plt.plot([0, 1], [0, 1], "k--", lw=0.5)
plt.xlabel("False accept rate"); plt.ylabel("True accept rate")
plt.legend(); plt.title("ROC on test set"); plt.tight_layout()
plt.savefig(DATA.parent / "roc_test.png", dpi=120)
print("saved src/roc_test.png")