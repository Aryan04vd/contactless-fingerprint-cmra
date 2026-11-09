"""Calibrated and quality-aware score fusion for HKPolyU contactless-to-contact."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from evaluate import compute_eer, tar_at_far  # noqa: E402

DATA = SRC / "data"
RESULTS = SRC.parent / "results"
RUN = RESULTS / "ise_seed7"
BASELINE_RUNS = ["base_seed1", "base_seed2", "base_seed3"]


def load_split(split: str):
    base = pd.read_csv(DATA / f"scores_{split}.csv")
    ise = pd.read_csv(RUN / f"scores_{split}_ise.csv")
    assert (base.contactless.values == ise.contactless.values).all()
    assert (base.contact.values == ise.contact.values).all()
    df = base.copy()
    df["score_ise"] = ise.score_ise.values

    mats = []
    for r in BASELINE_RUNS:
        d = pd.read_csv(DATA / "runs" / r / f"scores_{split}_dl.csv")
        assert (d.label.values == df.label.values).all()
        mats.append(d.score_dl.values)
    df["score_dl_ens"] = np.mean(mats, axis=0)

    q = pd.read_csv(RESULTS / "quality.csv").set_index("path")
    qcols = ["fg_frac", "contrast", "blur", "coherence", "grad", "entropy"]
    for side in ["contactless", "contact"]:
        for c in qcols:
            df[f"{side}_{c}"] = q.loc[df[side], c].values
    for c in qcols:
        a = df[f"contactless_{c}"].values
        b = df[f"contact_{c}"].values
        df[f"q_{c}_min"] = np.minimum(a, b)
        df[f"q_{c}_diff"] = np.abs(a - b)
    return df


def normalize(val, test, col):
    lo, hi = val[col].min(), val[col].max()
    f = lambda x: (x - lo) / (hi - lo + 1e-9)
    return f(val[col].values), f(test[col].values)


def fit_lr(X, y, groups, C_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)):
    best_c, best_eer = None, 1.0
    gkf = GroupKFold(n_splits=5)
    for C in C_grid:
        scores = np.zeros(len(y))
        for tr, te in gkf.split(X, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
            clf.fit(X[tr], y[tr])
            scores[te] = clf.predict_proba(X[te])[:, 1]
        eer = compute_eer(y, scores)
        if eer < best_eer:
            best_eer, best_c = eer, C
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced"))
    clf.fit(X, y)
    return clf, best_c, best_eer


def fixed_weight(val, test, a_col, b_col, grid=np.arange(0, 1.001, 0.05)):
    va, vb = normalize(val, test, a_col)
    tv, tt = normalize(val, test, b_col)
    best_w, best_eer = 0.5, 1.0
    for w in grid:
        eer = compute_eer(val.label.values, w * va + (1 - w) * vb)
        if eer < best_eer:
            best_eer, best_w = eer, float(w)
    return lambda df, w=best_w: w * normalize(val, test, a_col)[0 if df is val else 1] + (1 - w) * normalize(val, test, b_col)[0 if df is val else 1], best_w, best_eer


def bootstrap_eer(y, s, groups, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ids}
    out = []
    for _ in range(n):
        ix = np.concatenate([by[g] for g in rng.choice(ids, len(ids), replace=True)])
        out.append(compute_eer(y[ix], s[ix]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    val = load_split("val")
    test = load_split("test")
    yv, yt = val.label.values, test.label.values
    groups = test.finger_a.values

    methods = {}

    # single branches
    for name, col in [("Minutiae", "score_mm"), ("VerifNet-style DL (3-model avg)", "score_dl_ens"),
                      ("Identity-aware scattering (ISE)", "score_ise")]:
        methods[name] = (val[col].values, test[col].values, None)

    # fixed fusion of ISE + minutiae
    va, vt = normalize(val, test, "score_ise")
    ma, mt = normalize(val, test, "score_mm")
    best_w, best_eer = 0.5, 1.0
    for w in np.arange(0, 1.001, 0.05):
        eer = compute_eer(yv, w * va + (1 - w) * ma)
        if eer < best_eer:
            best_eer, best_w = eer, float(w)
    methods[f"Fixed fusion ISE+MM (w={best_w:.2f})"] = (
        best_w * va + (1 - best_w) * ma, best_w * vt + (1 - best_w) * mt, None)

    # fixed fusion of baseline DL + MM
    da, dt = normalize(val, test, "score_dl_ens")
    best_w2, best_eer2 = 0.5, 1.0
    for w in np.arange(0, 1.001, 0.05):
        eer = compute_eer(yv, w * da + (1 - w) * ma)
        if eer < best_eer2:
            best_eer2, best_w2 = eer, float(w)
    methods[f"Fixed fusion baselineDL+MM (w={best_w2:.2f})"] = (
        best_w2 * da + (1 - best_w2) * ma, best_w2 * dt + (1 - best_w2) * mt, None)

    # calibrated LR models
    quality_cols = [c for c in val.columns if c.startswith("q_") and c.endswith("_min")]
    diff_cols = [c for c in val.columns if c.startswith("q_") and c.endswith("_diff")]

    specs = {
        "Calibrated LR: ISE + MM": ["score_ise", "score_mm"],
        "Calibrated LR: ISE + MM + baselineDL": ["score_ise", "score_mm", "score_dl_ens"],
        "Quality-aware LR: ISE + MM + quality": ["score_ise", "score_mm"] + quality_cols,
        "Quality-aware LR: ISE + MM + baselineDL + quality": ["score_ise", "score_mm", "score_dl_ens"] + quality_cols,
        "Quality-aware LR (min+diff)": ["score_ise", "score_mm", "score_dl_ens"] + quality_cols + diff_cols,
    }
    lr_models = {}
    for name, cols in specs.items():
        Xv, Xt = val[cols].values.astype(float), test[cols].values.astype(float)
        clf, C, cv_eer = fit_lr(Xv, yv, val.finger_a.values)
        lr_models[name] = dict(model=clf, cols=list(cols), C=C, cv_eer=float(cv_eer),
                               val_eer=float(compute_eer(yv, clf.predict_proba(Xv)[:, 1])))
        methods[name] = (clf.predict_proba(Xv)[:, 1], clf.predict_proba(Xt)[:, 1], dict(C=C, cv_eer=float(cv_eer)))

    rows = []
    for name, (sv, st, extra) in methods.items():
        lo, hi = bootstrap_eer(yt, st, groups)
        row = dict(method=name,
                   val_eer=float(compute_eer(yv, sv)),
                   test_eer=float(compute_eer(yt, st)),
                   test_tar1=float(tar_at_far(yt, st, far=0.01)),
                   test_tar01=float(tar_at_far(yt, st, far=0.001)),
                   ci_lo=lo, ci_hi=hi)
        if extra:
            row.update(extra)
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values("test_eer")
    metrics["selected_on_val"] = metrics.method == min(lr_models, key=lambda k: lr_models[k]["val_eer"])
    metrics.to_csv(RESULTS / "fusion_metrics.csv", index=False)
    print(metrics.round(4).to_string(index=False))

    # select the fusion model on validation only, then freeze it for test / cross-session use
    primary_name = min(lr_models, key=lambda k: lr_models[k]["val_eer"])
    primary = lr_models[primary_name]
    joblib.dump(dict(name=primary_name, model=primary["model"], cols=primary["cols"],
                     C=primary["C"], cv_eer=primary["cv_eer"], val_eer=primary["val_eer"]),
                RESULTS / "quality_aware_fusion.joblib")
    print(f"selected on validation: {primary_name} (val EER {primary['val_eer']:.4f})")

    # save test scores
    out = test[["contactless", "contact", "label", "finger_a", "finger_b", "score_mm",
                "score_dl_ens", "score_ise"]].copy()
    for name, (_, st, _) in methods.items():
        out[name] = st
    out.to_csv(RESULTS / "test_scores.csv", index=False)

    # plots
    plt.figure(figsize=(6, 5))
    for name in ["Minutiae", "Identity-aware scattering (ISE)", primary_name]:
        if name not in methods:
            continue
        s = methods[name][1]
        fpr, tpr, _ = roc_curve(yt, s)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", lw=0.5)
    plt.xlabel("False accept rate")
    plt.ylabel("True accept rate")
    plt.legend(fontsize=7)
    plt.title("ROC on test set (HKPolyU contactless-to-contact)")
    plt.tight_layout()
    plt.savefig(RESULTS / "roc_test.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    for name in ["Minutiae", "Identity-aware scattering (ISE)", primary_name]:
        if name not in methods:
            continue
        s = methods[name][1]
        fpr, tpr, _ = roc_curve(yt, s)
        plt.plot(fpr, 1 - tpr, label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False accept rate")
    plt.ylabel("False reject rate")
    plt.legend(fontsize=7)
    plt.title("DET on test set")
    plt.tight_layout()
    plt.savefig(RESULTS / "det_test.png", dpi=160)
    plt.close()

    # score distribution for the validation-selected method
    best_name = primary_name
    s = methods[best_name][1]
    plt.figure(figsize=(6, 4))
    plt.hist(s[yt == 1], bins=50, alpha=0.6, label="genuine", density=True)
    plt.hist(s[yt == 0], bins=50, alpha=0.6, label="impostor", density=True)
    plt.xlabel("fused score")
    plt.ylabel("density")
    plt.legend()
    plt.title(best_name)
    plt.tight_layout()
    plt.savefig(RESULTS / "score_distribution.png", dpi=160)
    plt.close()

    summary = dict(best_method=best_name,
                   primary_selected_on="validation",
                   best=metrics[metrics.method == best_name].iloc[0].to_dict(),
                   fixed_w_ise_mm=best_w,
                   fixed_w_baseline_mm=best_w2)
    (RESULTS / "fusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nbest: {best_name}")
    print(f"saved {RESULTS / 'fusion_metrics.csv'}")


if __name__ == "__main__":
    main()
