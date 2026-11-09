"""Supplementary cross-session evaluation on test identities present in both sessions.

The primary manifests pair contactless and contact images from the same session.
This script builds genuine cross-session pairs (contactless session A vs contact
session B, A != B) for the 48 test fingers that occur in both sessions, plus an
equal number of cross-session impostor pairs, and scores them with the frozen
baseline, ISE and validation-selected fusion models.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "proposed"))
from evaluate import compute_eer, tar_at_far  # noqa: E402
from minutiae import extract_minutiae, match_score  # noqa: E402
from siamese import Siamese  # noqa: E402
from ise import ScatterNet, score_pairs  # noqa: E402

DATA = SRC / "data"
RESULTS = SRC.parent / "results"
RUNS = ["base_seed1", "base_seed2", "base_seed3"]


def build_pairs(meta: pd.DataFrame, fingers: list[int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    by = {}
    for f in fingers:
        d = meta[meta.finger_id == f]
        by[f] = {
            "first_session": (
                sorted(d[(d.modality == "contactless") & (d.session == "first_session")].path),
                sorted(d[(d.modality == "contact") & (d.session == "first_session")].path),
            ),
            "second_session": (
                sorted(d[(d.modality == "contactless") & (d.session == "second_session")].path),
                sorted(d[(d.modality == "contact") & (d.session == "second_session")].path),
            ),
        }

    rows = []
    for f in fingers:
        for sa, sb in [("first_session", "second_session"), ("second_session", "first_session")]:
            cl, _ = by[f][sa]
            _, ct = by[f][sb]
            for a in cl:
                for b in ct:
                    rows.append(dict(contactless=a, contact=b, label=1, finger_a=f, finger_b=f,
                                     sess_a=sa, sess_b=sb))
    n_gen = len(rows)
    seen = {(r["contactless"], r["contact"]) for r in rows}
    while len(rows) < 2 * n_gen:
        fa, fb = rng.choice(fingers, 2, replace=False)
        sa, sb = rng.choice(["first_session", "second_session"], 2)
        a = by[fa][sa][0][rng.integers(6)]
        b = by[fb][sb][1][rng.integers(6)]
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append(dict(contactless=a, contact=b, label=0, finger_a=int(fa), finger_b=int(fb),
                         sess_a=sa, sess_b=sb))
    return pd.DataFrame(rows)


def score_minutiae(pairs: pd.DataFrame) -> np.ndarray:
    cache_path = DATA / "minutiae_cache.pkl"
    cache = pickle.load(open(cache_path, "rb")) if cache_path.exists() else {}
    needed = {(p, "contactless" if "contactless" in p else "contact")
              for p in list(pairs.contactless) + list(pairs.contact)}
    todo = [x for x in needed if x[0] not in cache]
    print(f"extracting minutiae from {len(todo)} new images")
    if todo:
        res = Parallel(n_jobs=-1)(delayed(extract_minutiae)(p, m) for p, m in todo)
        cache.update({p: r for (p, _), r in zip(todo, res)})
        pickle.dump(cache, open(cache_path, "wb"))
    return np.array(Parallel(n_jobs=-1, batch_size=32)(
        delayed(match_score)(cache[a], cache[b])
        for a, b in zip(pairs.contactless, pairs.contact)))


@torch.no_grad()
def score_baseline(pairs: pd.DataFrame) -> np.ndarray:
    feats = np.load(DATA / "scat.npy", mmap_mode="r")
    idx = dict(pd.read_csv(DATA / "images_index.csv").values)
    a = np.array([idx[p] for p in pairs.contactless])
    b = np.array([idx[p] for p in pairs.contact])
    scores = []
    for run in RUNS:
        model = Siamese(shared=True)
        model.load_state_dict(torch.load(DATA / "runs" / run / "best_model.pt", map_location="cpu"))
        model.eval()
        out = []
        for i in range(0, len(a), 256):
            xa = torch.from_numpy(np.asarray(feats[a[i:i + 256]])).float()
            xb = torch.from_numpy(np.asarray(feats[b[i:i + 256]])).float()
            ea, eb = model(xa, xb)
            out.append((-F.pairwise_distance(ea, eb)).numpy())
        scores.append(np.concatenate(out))
    return np.mean(scores, axis=0)


def main():
    meta = pd.read_csv(DATA / "metadata.csv")
    split = pd.read_csv(DATA / "finger_split.csv")
    meta = meta.merge(split, on="finger_id")
    both = meta.groupby("finger_id").session.nunique()
    test_f = sorted(set(split[split.split == "test"].finger_id) & set(both[both == 2].index))
    print(f"test fingers in both sessions: {len(test_f)}")

    pairs = build_pairs(meta, test_f)
    print("pairs:", len(pairs), pairs.label.value_counts().to_dict())
    cache_path = RESULTS / "cross_session_scores.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if len(cached) == len(pairs) and {"score_mm", "score_dl_ens", "score_ise"} <= set(cached.columns):
            print("reusing cached cross-session scores")
            pairs = cached
        else:
            cached = None
    else:
        cached = None
    if cached is None or "score_mm" not in pairs.columns:
        pairs["score_mm"] = score_minutiae(pairs)
        print("minutiae done")
        pairs["score_dl_ens"] = score_baseline(pairs)
        print("baseline done")

        feats = np.load(DATA / "scat.npy", mmap_mode="r")
        idx = dict(pd.read_csv(DATA / "images_index.csv").values)
        model = ScatterNet(128)
        model.load_state_dict(torch.load(RESULTS / "ise_seed7" / "model.pt", map_location="cpu"))
        pairs["score_ise"], _ = score_pairs(model, feats, pairs, idx, tta=True)
        print("ISE done")

    q = pd.read_csv(RESULTS / "quality.csv").set_index("path")
    qcols = ["fg_frac", "contrast", "blur", "coherence", "grad", "entropy"]
    for c in qcols:
        a = q.loc[pairs.contactless, c].values
        b = q.loc[pairs.contact, c].values
        pairs[f"q_{c}_min"] = np.minimum(a, b)
        pairs[f"q_{c}_diff"] = np.abs(a - b)

    bundle = joblib.load(RESULTS / "quality_aware_fusion.joblib")
    pairs["fusion"] = bundle["model"].predict_proba(pairs[bundle["cols"]].values)[:, 1]
    print("fusion model:", bundle["name"])

    # fixed ISE+MM fusion, with the weight and score ranges fixed on validation
    val = pd.read_csv(DATA / "scores_val.csv")
    vise = pd.read_csv(RESULTS / "ise_seed7" / "scores_val_ise.csv")
    w = 0.85

    def norm(x, lo, hi):
        return (x - lo) / (hi - lo + 1e-9)

    for df in [pairs]:
        df["fixed_ise_mm"] = (
            w * norm(df.score_ise.values, vise.score_ise.min(), vise.score_ise.max())
            + (1 - w) * norm(df.score_mm.values, val.score_mm.min(), val.score_mm.max()))
    pairs.to_csv(RESULTS / "cross_session_scores.csv", index=False)

    # same-session reference restricted to the same 48 identities
    same = pd.read_csv(RESULTS / "test_scores.csv")
    same = same[same.finger_a.isin(test_f) & same.finger_b.isin(test_f)]
    same["fusion"] = same[bundle["name"]]
    methods = [("Minutiae", "score_mm", "score_mm"),
               ("VerifNet-style DL (3-model avg)", "score_dl_ens", "VerifNet-style DL (3-model avg)"),
               ("Identity-aware scattering (ISE)", "score_ise", "Identity-aware scattering (ISE)"),
               ("Fixed fusion ISE+MM (w=0.85)", "fixed_ise_mm", "Fixed fusion ISE+MM (w=0.85)"),
               ("Validation-selected fusion", "fusion", "fusion")]
    rows = []
    for name, cross_col, same_col in methods:
        row = dict(method=name, n_genuine=int(pairs.label.sum()), n_impostor=int((1 - pairs.label).sum()))
        row["cross_eer"] = float(compute_eer(pairs.label.values, pairs[cross_col].values))
        row["cross_tar1"] = float(tar_at_far(pairs.label.values, pairs[cross_col].values, far=0.01))
        row["cross_tar01"] = float(tar_at_far(pairs.label.values, pairs[cross_col].values, far=0.001))
        row["same_n"] = len(same)
        row["same_eer"] = float(compute_eer(same.label.values, same[same_col].values))
        row["same_tar1"] = float(tar_at_far(same.label.values, same[same_col].values, far=0.01))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "cross_session_metrics.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
