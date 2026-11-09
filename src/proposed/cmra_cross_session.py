"""Cross-session evaluation for a trained CMRA run.

Reuses the exact cross-session pair manifest built by cross_session.py (48 test
identities present in both sessions) and scores it with a frozen CMRA model.
The same-session reference is restricted to the same 48 identities so the two
columns are like-for-like.

Usage:
    python src/proposed/cmra_cross_session.py --run cmra_full_s7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "proposed"))
from evaluate import compute_eer, tar_at_far  # noqa: E402
from cross_session import build_pairs  # noqa: E402
from cmra import ScatterNet, score_pairs, load_index  # noqa: E402

DATA = SRC / "data"
RESULTS = SRC.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="cmra_full_s7")
    args = ap.parse_args()

    meta = pd.read_csv(DATA / "metadata.csv")
    split = pd.read_csv(DATA / "finger_split.csv")
    meta = meta.merge(split, on="finger_id")
    both = meta.groupby("finger_id").session.nunique()
    test_f = sorted(set(split[split.split == "test"].finger_id) & set(both[both == 2].index))
    pairs = build_pairs(meta, test_f)
    print(f"identities {len(test_f)} | pairs {len(pairs)} | "
          f"genuine {int(pairs.label.sum())} | impostor {int((1-pairs.label).sum())}")

    feats = np.load(DATA / "scat.npy", mmap_mode="r")
    idx = load_index()
    model = ScatterNet(128)
    model.load_state_dict(torch.load(RESULTS / args.run / "model.pt", map_location="cpu"))
    pairs["score_cmra"], y = score_pairs(model, feats, pairs, idx, tta=True)
    assert (y == pairs.label.values).all()

    # reuse the v1 cross-session scores when available (same pair order)
    cache = RESULTS / "cross_session_scores.csv"
    if cache.exists():
        ref = pd.read_csv(cache)
        if len(ref) == len(pairs) and (ref.contactless.values == pairs.contactless.values).all():
            for c in ["score_mm", "score_ise", "fusion"]:
                if c in ref.columns:
                    pairs[c] = ref[c].values
            print("attached cached minutiae / ISE / fusion cross-session scores")

    same = pd.read_csv(RESULTS / args.run / "scores_test.csv")
    same = same[same.finger_a.isin(test_f) & same.finger_b.isin(test_f)]
    print(f"same-session reference pairs: {len(same)}")

    cols = [("CMRA (this run)", "score_cmra")]
    if "score_mm" in pairs:
        cols.append(("Minutiae", "score_mm"))
    if "score_ise" in pairs:
        cols.append(("ISE (v1)", "score_ise"))
    if "fusion" in pairs:
        cols.append(("v1 quality-aware fusion", "fusion"))

    rows = []
    for name, c in cols:
        same_c = {"score_cmra": "score_cmra", "score_ise": "score_ise",
                  "score_mm": "score_mm", "fusion": "fusion"}.get(c)
        row = dict(method=name,
                   n_genuine=int(pairs.label.sum()),
                   n_impostor=int((1 - pairs.label).sum()))
        row["cross_eer"] = float(compute_eer(pairs.label.values, pairs[c].values))
        row["cross_tar1"] = float(tar_at_far(pairs.label.values, pairs[c].values, 0.01))
        row["cross_tar01"] = float(tar_at_far(pairs.label.values, pairs[c].values, 0.001))
        if same_c in same.columns:
            row["same_n"] = len(same)
            row["same_eer"] = float(compute_eer(same.label.values, same[same_c].values))
            row["same_tar1"] = float(tar_at_far(same.label.values, same[same_c].values, 0.01))
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "cmra_cross_session.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
