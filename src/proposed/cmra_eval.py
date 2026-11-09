"""Evaluation for CMRA runs: ablation, bootstrap CIs and risk-coverage curves.

Reads every results/cmra_*/summary.json plus the original ISE run, rebuilds the
comparison table, and computes selective-verification (abstention) behaviour on
the held-out test identities.

The decision threshold is always chosen on the validation split and then applied
unchanged to the test split. Abstention ranks test pairs by confidence
|score - tau| and reports the error rate among the retained pairs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from evaluate import compute_eer, tar_at_far  # noqa: E402

DATA = SRC / "data"
RESULTS = SRC.parent / "results"

MODE_LABEL = {
    "arc": "CMRA arc-only (PK batches)",
    "proto": "CMRA + prototype",
    "rel": "CMRA + relational",
    "full": "CMRA full (prototype + relational)",
}


def eer_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(thr[i])


def bootstrap_eer(y, s, groups, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ids}
    out = []
    for _ in range(n):
        ix = np.concatenate([by[g] for g in rng.choice(ids, len(ids), replace=True)])
        out.append(compute_eer(y[ix], s[ix]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_bootstrap_delta(y, s_better, s_worse, groups, n=500, seed=0):
    """EER(worse) - EER(better) on identical identity resamples.

    Positive values mean the first argument has the lower EER.
    """
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ids}
    out = []
    for _ in range(n):
        ix = np.concatenate([by[g] for g in rng.choice(ids, len(ids), replace=True)])
        out.append(compute_eer(y[ix], s_worse[ix]) - compute_eer(y[ix], s_better[ix]))
    out = np.asarray(out)
    return (float(out.mean()), float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5)), float((out > 0).mean()))


def risk_coverage(y, s, tau, coverages):
    """Error rate at a fixed threshold among the most confident fraction."""
    decision = (s >= tau).astype(int)
    correct = (decision == y).astype(int)
    conf = np.abs(s - tau)
    order = np.argsort(-conf)
    out = {}
    for c in coverages:
        n_keep = max(1, int(round(c * len(y))))
        keep = order[:n_keep]
        out[c] = float(1.0 - correct[keep].mean())
    return out


def aurc(y, s, tau, steps=100):
    cov = np.linspace(0.05, 1.0, steps)
    rc = risk_coverage(y, s, tau, cov)
    risk = np.array([rc[c] for c in cov])
    return float(np.trapezoid(risk, cov) / (cov[-1] - cov[0]))


def load_cmra_runs():
    runs = []
    for d in sorted(RESULTS.glob("cmra_*")):
        sm = d / "summary.json"
        vf = d / "scores_val.csv"
        tf = d / "scores_test.csv"
        if not (sm.exists() and vf.exists() and tf.exists()):
            continue
        summary = json.loads(sm.read_text())
        runs.append(dict(
            name=d.name,
            mode=summary["config"]["mode"],
            label=MODE_LABEL.get(summary["config"]["mode"], summary["config"]["mode"]),
            seed=summary["config"]["seed"],
            val=pd.read_csv(vf),
            test=pd.read_csv(tf),
            summary=summary,
        ))
    return runs


def main():
    runs = load_cmra_runs()
    if not runs:
        print("no CMRA runs found")
        return
    print(f"found {len(runs)} CMRA runs: {[r['name'] for r in runs]}")

    ise_val = pd.read_csv(RESULTS / "ise_seed7" / "scores_val_ise.csv")
    ise_test = pd.read_csv(RESULTS / "ise_seed7" / "scores_test_ise.csv")

    methods = [dict(label="ISE (v1 baseline)", val=ise_val.score_ise.values,
                    test=ise_test.score_ise.values, val_df=ise_val, test_df=ise_test)]
    for r in runs:
        methods.append(dict(label=r["label"] + f" [{r['name']}]",
                            val=r["val"].score_cmra.values,
                            test=r["test"].score_cmra.values,
                            val_df=r["val"], test_df=r["test"]))

    # ---------- main metrics ----------
    rows = []
    groups = ise_test.finger_a.values
    for m in methods:
        yv = m["val_df"].label.values
        yt = m["test_df"].label.values
        sv, st = m["val"], m["test"]
        lo, hi = bootstrap_eer(yt, st, groups, n=500)
        rows.append(dict(
            method=m["label"],
            val_eer=compute_eer(yv, sv),
            test_eer=compute_eer(yt, st),
            test_tar1=tar_at_far(yt, st, 0.01),
            test_tar01=tar_at_far(yt, st, 0.001),
            ci_lo=lo, ci_hi=hi,
            tau=eer_threshold(yv, sv),
        ))
    table = pd.DataFrame(rows).sort_values("test_eer")
    table.to_csv(RESULTS / "cmra_metrics.csv", index=False)
    print("\n=== CMRA ablation (test, 10,728 pairs) ===")
    print(table.round(4).to_string(index=False))

    # ---------- risk-coverage ----------
    coverages = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
    rc_rows = []
    rc_curves = {}
    for m, row in zip(methods, table.set_index("method").reindex([x["label"] for x in methods]).to_dict("index").values()):
        tau = row["tau"]
        yt = m["test_df"].label.values
        st = m["test"]
        rc = risk_coverage(yt, st, tau, coverages)
        rc_curves[m["label"]] = rc
        rc_rows.append(dict(method=m["label"], aurc=aurc(yt, st, tau),
                            **{f"err@{int(c*100)}": rc[c] for c in coverages}))
    rc_table = pd.DataFrame(rc_rows).sort_values("aurc")
    rc_table.to_csv(RESULTS / "cmra_risk_coverage.csv", index=False)
    print("\n=== Risk-coverage (fixed validation threshold) ===")
    print(rc_table.round(4).to_string(index=False))

    # ---------- figures ----------
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for m in methods:
        yt = m["test_df"].label.values
        fpr, tpr, _ = roc_curve(yt, m["test"])
        ax.plot(fpr * 100, (1 - tpr) * 100, lw=1.6, label=m["label"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.1, 40); ax.set_ylim(0.1, 60)
    ax.set_xlabel("False accept rate (%)"); ax.set_ylabel("False reject rate (%)")
    ax.set_title("CMRA ablation: DET on held-out test identities")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(RESULTS / "cmra_det.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for label, rc in rc_curves.items():
        xs = sorted(rc, reverse=True)
        ax.plot([x * 100 for x in xs], [rc[x] * 100 for x in xs], marker="o", ms=3, lw=1.5, label=label)
    ax.set_xlabel("Coverage (% of comparisons answered)")
    ax.set_ylabel("Error rate at fixed threshold (%)")
    ax.set_title("Risk-controlled abstention: error vs coverage")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(RESULTS / "cmra_risk_coverage.png", dpi=170); plt.close(fig)

    # ---------- v1 pipeline context (test only) ----------
    v1 = pd.read_csv(RESULTS / "test_scores.csv")
    ctx_rows = []
    for label, col in [("Minutiae", "Minutiae"),
                       ("VerifNet-style DL (3-seed avg)", "VerifNet-style DL (3-model avg)"),
                       ("v1 fixed fusion DL+MM", "Fixed fusion baselineDL+MM (w=0.50)"),
                       ("v1 best fusion (quality-aware LR)", "Quality-aware LR (min+diff)")]:
        if col not in v1.columns:
            continue
        ctx_rows.append(dict(
            method=label,
            test_eer=compute_eer(v1.label.values, v1[col].values),
            test_tar1=tar_at_far(v1.label.values, v1[col].values, 0.01),
            test_tar01=tar_at_far(v1.label.values, v1[col].values, 0.001),
        ))
    ctx = pd.DataFrame(ctx_rows)
    ctx.to_csv(RESULTS / "cmra_v1_context.csv", index=False)
    print("\n=== v1 pipeline context (test only) ===")
    print(ctx.round(4).to_string(index=False))

    # ---------- paired comparison: best CMRA vs v1 selected fusion ----------
    v1col = "Quality-aware LR (min+diff)"
    paired = None
    if v1col in v1.columns:
        best_label = str(table.iloc[0]["method"])
        best = next(m for m in methods if m["label"] == best_label)
        cmp = v1[["contactless", "contact", "label", "finger_a", v1col]].merge(
            best["test_df"][["contactless", "contact", "score_cmra"]],
            on=["contactless", "contact"], how="inner")
        mean, lo, hi, prob = paired_bootstrap_delta(
            cmp.label.values.astype(float), cmp.score_cmra.values,
            cmp[v1col].values, cmp.finger_a.values)
        paired = dict(against=best_label, delta_pp=mean * 100, ci_lo=lo * 100,
                      ci_hi=hi * 100, prob_better=prob, n=int(len(cmp)))
        print(f"\nPaired bootstrap vs v1 best fusion ({len(cmp)} pairs): "
              f"EER reduction {mean*100:.2f} pp, 95% CI [{lo*100:.2f}, {hi*100:.2f}], "
              f"P(CMRA lower EER) = {prob:.3f}")

    summary = dict(
        table=table.to_dict("records"),
        risk_coverage=rc_table.to_dict("records"),
        v1_context=ctx.to_dict("records"),
        paired_vs_v1=paired,
        coverages=coverages,
    )
    (RESULTS / "cmra_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("\nsaved results/cmra_metrics.csv, cmra_risk_coverage.csv, cmra_det.png, cmra_risk_coverage.png")


if __name__ == "__main__":
    main()
