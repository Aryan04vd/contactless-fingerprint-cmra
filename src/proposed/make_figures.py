"""Generate pipeline, training-curve and comparison figures for the report/slides.

The figures in this file describe the final CMRA system, not the superseded v1
(ISE + quality-aware fusion) pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

SRC = Path(__file__).resolve().parents[1]
RESULTS = SRC.parent / "results"


def box(ax, x, y, w, h, text, color="#e8f0fe", edge="#4a6fa5", size=9.8):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       linewidth=1.3, edgecolor=edge, facecolor=color)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color="#1f2937", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-", color="#4b5563", lw=1.3):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color, linestyle=ls))


def pipeline():
    """CMRA architecture: shared embedding + two cross-modal consistency losses."""
    fig, ax = plt.subplots(figsize=(11.6, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------------------------------------------------------- inputs
    box(ax, 0.005, 0.705, 0.115, 0.135, "Contactless\nphoto", "#fef3c7", "#b45309")
    box(ax, 0.005, 0.480, 0.115, 0.135, "Contact\nprint", "#dcfce7", "#15803d")

    # ------------------------------------------------------------ front-end
    box(ax, 0.155, 0.480, 0.135, 0.360,
        "Preprocess\nflip · CLAHE\nridge scale 192²", "#e0f2fe", "#0369a1", 9.2)
    box(ax, 0.325, 0.480, 0.160, 0.360,
        "Fixed scattering\nJ = 2\n81 × 48 × 48", "#ede9fe", "#6d28d9", 9.6)
    box(ax, 0.520, 0.480, 0.170, 0.360,
        "Shared embedding\nCNN + ArcFace\n128-D, L2-norm", "#ddd6fe", "#5b21b6", 9.8)

    # --------------------------------------------------------------- scoring
    box(ax, 0.725, 0.555, 0.105, 0.210, "Cosine\nsimilarity", "#f3f4f6", "#374151", 9.6)
    box(ax, 0.865, 0.615, 0.125, 0.135, "Match score\n(verification)", "#dcfce7", "#166534", 9.2)
    box(ax, 0.865, 0.480, 0.125, 0.135, "Risk-controlled\nabstention\n|s − τ|", "#ffedd5", "#c2410c", 8.8)

    # ---------------------------------------------------------------- arrows
    arrow(ax, 0.120, 0.7725, 0.155, 0.745)
    arrow(ax, 0.120, 0.5475, 0.155, 0.575)
    arrow(ax, 0.290, 0.660, 0.325, 0.660)
    arrow(ax, 0.485, 0.660, 0.520, 0.660)
    arrow(ax, 0.690, 0.660, 0.725, 0.660)
    arrow(ax, 0.830, 0.678, 0.865, 0.682)
    arrow(ax, 0.830, 0.642, 0.865, 0.548)

    # ------------------------------------------------------- training losses
    ax.plot([0.005, 0.995], [0.405, 0.405], color="#d1d5db", lw=1.0)
    ax.text(0.005, 0.372,
            "Training objective:  identity-balanced PK batches  (P = 16, K = 2 per modality)",
            ha="left", va="center", fontsize=9.8, color="#374151", fontweight="bold")
    ax.text(0.995, 0.372, "training only — removed at inference", ha="right", va="center",
            fontsize=8.6, color="#6b7280", style="italic")

    box(ax, 0.155, 0.095, 0.190, 0.210,
        "ArcFace loss\ns = 30, m = 0.30\ncross-modal\nidentity classes", "#e0e7ff", "#3730a3", 8.8)
    box(ax, 0.375, 0.095, 0.225, 0.210,
        "Prototype consistency\n1 − cos(μCL, μCT)\naligns per-identity\nmodality centres", "#fce7f3", "#be185d", 8.8)
    box(ax, 0.630, 0.095, 0.240, 0.210,
        "Relational consistency\nKL(softmax(SCL/τ) || softmax(SCT/τ))\nmatch P × P cross-modal\nsimilarity structure",
        "#e0f2fe", "#0369a1", 8.3)

    # dashed routing from the shared embedding to the three losses
    ax.plot([0.605, 0.605], [0.480, 0.335], color="#6b7280", lw=1.1, ls="--")
    ax.plot([0.250, 0.750], [0.335, 0.335], color="#6b7280", lw=1.1, ls="--")
    for x in (0.250, 0.4875, 0.750):
        arrow(ax, x, 0.335, x, 0.305, ls="--", color="#6b7280")

    # ---------------------------------------------------------------- titles
    ax.text(0.5, 0.975,
            "Cross-Modal Relational Alignment (CMRA) for contactless-to-contact verification",
            ha="center", fontsize=14.5, fontweight="bold", color="#111827")
    ax.text(0.5, 0.925,
            "A single embedding replaces the VerifNet-style Siamese branch and score fusion",
            ha="center", fontsize=9.6, color="#4b5563", style="italic")

    plt.tight_layout()
    plt.savefig(RESULTS / "pipeline.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def training_curve():
    """Validation EER / TAR@1% for the CMRA full model (seed 8)."""
    s = json.loads((RESULTS / "cmra_full_s8" / "summary.json").read_text())
    h = pd.DataFrame(s["history"])
    fig, ax1 = plt.subplots(figsize=(7.4, 4.3))
    ax1.plot(h.epoch, h.val_eer * 100, "o-", ms=3, color="#5b21b6", label="validation EER")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation EER (%)", color="#5b21b6")
    ax1.tick_params(axis="y", colors="#5b21b6")
    ax2 = ax1.twinx()
    ax2.plot(h.epoch, h.val_tar1 * 100, "s--", ms=3, color="#059669", label="validation TAR@FAR=1%")
    ax2.set_ylabel("TAR@FAR=1% (%)", color="#059669")
    ax2.tick_params(axis="y", colors="#059669")
    best = h.loc[h.val_eer.idxmin()]
    ax1.axvline(best.epoch, color="gray", lw=0.8, ls=":")
    ax1.annotate(f"selected epoch {int(best.epoch)}\nEER {best.val_eer * 100:.2f}%",
                 xy=(best.epoch, best.val_eer * 100),
                 xytext=(best.epoch + 2, best.val_eer * 100 + 5),
                 fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"))
    lines = [l for l in ax1.get_lines() + ax2.get_lines() if not l.get_label().startswith("_")]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper center",
               bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=8, frameon=False)
    ax1.set_title("CMRA training (30 epochs, seed 8, CPU)")
    ax1.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / "training_curve.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


def comparison():
    """Bar chart of test EER for baselines, v1 fusion and the CMRA ablation."""
    cmra = pd.read_csv(RESULTS / "cmra_metrics.csv")
    v1 = pd.read_csv(RESULTS / "cmra_v1_context.csv")

    def cmra_row(key):
        row = cmra[cmra.method.str.contains(key, regex=False)].iloc[0]
        return row.test_eer * 100

    def v1_row(key):
        row = v1[v1.method.str.contains(key, regex=False)].iloc[0]
        return row.test_eer * 100

    systems = [
        ("Minutiae only", v1_row("Minutiae"), "baseline"),
        ("VerifNet-style DL", v1_row("VerifNet-style DL"), "baseline"),
        ("v1 fixed fusion DL+MM", v1_row("v1 fixed fusion"), "v1"),
        ("ISE (v1 single embedding)", cmra_row("ISE (v1 baseline)"), "v1"),
        ("CMRA ArcFace + PK only", cmra_row("arc-only"), "cmra"),
        ("v1 best fusion (quality-aware)", v1_row("v1 best fusion"), "v1"),
        ("CMRA + prototype", cmra_row("CMRA + prototype"), "cmra"),
        ("CMRA + relational", cmra_row("CMRA + relational"), "cmra"),
        ("CMRA full (seed 7)", cmra_row("cmra_full_s7"), "cmra"),
        ("CMRA full (seed 8)", cmra_row("cmra_full_s8"), "cmra"),
    ]
    systems.sort(key=lambda t: t[1], reverse=True)
    labels = [s[0] for s in systems]
    values = [s[1] for s in systems]
    palette = {"baseline": "#94a3b8", "v1": "#f59e0b", "cmra": "#16a34a"}
    colors = [palette[s[2]] for s in systems]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.barh(labels, values, color=colors, edgecolor="white")
    for y, v in enumerate(values):
        ax.text(v + 0.4, y, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Test EER (%) — lower is better")
    ax.set_title("Contactless-to-contact verification on HKPolyU CL2CB (10,728 pairs)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, 45)
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k]) for k in ("baseline", "v1", "cmra")]
    ax.legend(handles, ["Baseline", "v1 pipeline", "CMRA (proposed)"], loc="upper right",
              fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(RESULTS / "cmra_comparison.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    pipeline()
    training_curve()
    comparison()
    print("wrote pipeline.png, training_curve.png, cmra_comparison.png")
