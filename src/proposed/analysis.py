"""Stratified, quality-bin and error analysis for the final test scores."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
RESULTS = SRC.parent / "results"
sys.path.insert(0, str(SRC))
from evaluate import compute_eer, tar_at_far  # noqa: E402

DATA = SRC / "data"
KEY = [
    "Minutiae",
    "VerifNet-style DL (3-model avg)",
    "Identity-aware scattering (ISE)",
    "Fixed fusion baselineDL+MM (w=0.50)",
    "Fixed fusion ISE+MM (w=0.85)",
    "Calibrated LR: ISE + MM + baselineDL",
    "Quality-aware LR: ISE + MM + baselineDL + quality",
    "Quality-aware LR (min+diff)",
]


def bootstrap_eer(y, s, groups, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ids}
    out = []
    for _ in range(n):
        ix = np.concatenate([by[g] for g in rng.choice(ids, len(ids), replace=True)])
        out.append(compute_eer(y[ix], s[ix]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def strats(df):
    meta = pd.read_csv(DATA / "metadata.csv")[["path", "session", "modality"]]
    m = meta.set_index(["path", "modality"])
    sess = {}
    for side, col in [("contactless", "contactless"), ("contact", "contact")]:
        for p in df[col]:
            sess[(p, side)] = m.loc[(p, side), "session"]
    df = df.copy()
    df["sess_a"] = [sess[(p, "contactless")] for p in df.contactless]
    df["sess_b"] = [sess[(p, "contact")] for p in df.contact]
    df["stratum"] = np.where(df.sess_a == df.sess_b, "same-session", "cross-session")
    return df


def main():
    df = strats(pd.read_csv(RESULTS / "test_scores.csv"))
    y = df.label.values
    print("test pairs by stratum:\n", df.groupby(["stratum", "label"]).size().to_string())

    rows = []
    for method in KEY:
        s = df[method].values
        row = dict(method=method, n=len(df),
                   eer=float(compute_eer(y, s)),
                   tar1=float(tar_at_far(y, s, far=0.01)),
                   tar01=float(tar_at_far(y, s, far=0.001)))
        for st in ["same-session", "cross-session"]:
            ix = (df.stratum == st).values
            row[f"{st}_n"] = int(ix.sum())
            # metrics are only defined when both genuine and impostor pairs exist
            if len(np.unique(y[ix])) == 2:
                row[f"{st}_eer"] = float(compute_eer(y[ix], s[ix]))
                row[f"{st}_tar1"] = float(tar_at_far(y[ix], s[ix], far=0.01))
            else:
                row[f"{st}_eer"] = np.nan
                row[f"{st}_tar1"] = np.nan
        rows.append(row)
    strat = pd.DataFrame(rows)
    strat.to_csv(RESULTS / "stratified_metrics.csv", index=False)
    print("\n", strat.round(4).to_string(index=False))

    # quality bins: use the minimum coherence of the pair, quartiles on test
    q = pd.read_csv(RESULTS / "quality.csv").set_index("path")
    df["coh_min"] = np.minimum(q.loc[df.contactless, "coherence"].values,
                               q.loc[df.contact, "coherence"].values)
    df["blur_min"] = np.minimum(q.loc[df.contactless, "blur"].values,
                                q.loc[df.contact, "blur"].values)
    df["qbin"] = pd.qcut(df.coh_min, 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    rows = []
    for b, g in df.groupby("qbin", observed=True):
        row = dict(quality_bin=str(b), n=len(g), coh_min_mean=float(g.coh_min.mean()))
        for method in ["Minutiae", "VerifNet-style DL (3-model avg)",
                       "Identity-aware scattering (ISE)", "Quality-aware LR (min+diff)"]:
            row[f"eer_{method}"] = float(compute_eer(g.label.values, g[method].values))
        rows.append(row)
    qb = pd.DataFrame(rows)
    qb.to_csv(RESULTS / "quality_bins.csv", index=False)
    print("\n", qb.round(4).to_string(index=False))

    # quality-bin figure
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = ["Minutiae", "VerifNet DL", "ISE", "Quality-aware fusion"]
    cols = ["Minutiae", "VerifNet-style DL (3-model avg)",
            "Identity-aware scattering (ISE)", "Quality-aware LR (min+diff)"]
    x = np.arange(len(qb))
    for c, lab in zip(cols, labels):
        ax.bar(x + 0.2 * (cols.index(c) - 1.5), qb[f"eer_{c}"], width=0.2, label=lab)
    ax.set_xticks(x, qb.quality_bin.astype(str))
    ax.set_ylabel("EER")
    ax.set_xlabel("Pair minimum orientation coherence (quartile)")
    ax.set_title("EER by image-quality quartile (test set)")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "quality_bins.png", dpi=160)
    plt.close()

    # score distributions for key methods
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))
    short = ["Minutiae", "ISE", "Quality-aware fusion"]
    for ax, m, lab in zip(axes, ["Minutiae", "Identity-aware scattering (ISE)",
                                 "Quality-aware LR (min+diff)"], short):
        ax.hist(df[df.label == 1][m], bins=50, alpha=0.6, density=True, label="genuine")
        ax.hist(df[df.label == 0][m], bins=50, alpha=0.6, density=True, label="impostor")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("match score", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(RESULTS / "score_distributions.png", dpi=160)
    plt.close()

    # failure analysis for the best method
    best = "Quality-aware LR (min+diff)"
    out = df[["contactless", "contact", "label", "finger_a", "finger_b", "stratum",
              "coh_min"] + KEY].copy()
    fa = out[(out.label == 0)].nlargest(10, best)
    fr = out[(out.label == 1)].nsmallest(10, best)
    pd.concat([fa.assign(error="false_accept"), fr.assign(error="false_reject")]).to_csv(
        RESULTS / "failure_cases.csv", index=False)
    print("\nfalse accepts:", len(fa), "false rejects:", len(fr))

    # summary json
    lo, hi = bootstrap_eer(y, df[best].values, df.finger_a.values)
    summary = dict(
        strata=[dict(stratum=str(k[0]), label=int(k[1]), n=int(v))
                for k, v in df.groupby(["stratum", "label"]).size().items()],
        best_method=best, best_eer=float(compute_eer(y, df[best].values)),
        ci=[lo, hi])
    (RESULTS / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                   encoding="utf-8")
    print("\nsaved stratified_metrics.csv, quality_bins.csv, failure_cases.csv, figures")


if __name__ == "__main__":
    main()
