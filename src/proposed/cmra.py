"""Cross-Modal Relational Alignment (CMRA) for contactless-to-contact matching.

Replaces VerifNet's pairwise-contrastive branch with a single embedding trained so
that the two modalities agree on the *relational* layout of identities:

    L = L_arcface + lam_proto * L_proto + lam_rel * L_rel

where, for a batch of P identities with K contactless and K contact images each,

    mu_i^cl = mean_k f(x_i,k^cl)          (L2-normalised prototype)
    mu_i^ct = mean_k f(x_i,k^ct)

    L_proto = mean_i (1 - cos(mu_i^cl, mu_i^ct))
    L_rel   = mean_{i,j} ( <mu_i^cl,mu_j^cl> - <mu_i^ct,mu_j^ct> )^2

The relational term never forces the absolute coordinates of the two modalities to
coincide; it only forces the similarity (Gram) structure over identities to agree.

Modes (for the ablation):
    arc   : ArcFace only, identity-balanced PK batches
    proto : ArcFace + prototype consistency
    rel   : ArcFace + relational consistency
    full  : ArcFace + both

Usage:
    python src/proposed/cmra.py --mode full --seed 7 --out cmra_full_s7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from evaluate import compute_eer, tar_at_far  # noqa: E402

DATA = SRC / "data"
RESULTS = SRC.parent / "results"
RESULTS.mkdir(exist_ok=True)


class ScatterNet(nn.Module):
    """Same 0.13M-parameter body used by ISE (unchanged, so the comparison is fair)."""

    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(81),
            nn.Conv2d(81, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, emb_dim)

    def forward(self, x):
        x = self.body(x).flatten(1)
        return F.normalize(self.head(x), dim=1)


class ArcFace(nn.Module):
    def __init__(self, emb_dim: int, n_classes: int, scale: float = 30.0, margin: float = 0.30):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, emb_dim))
        self.scale = scale
        self.margin = margin

    def forward(self, emb, labels):
        cos = F.linear(emb, F.normalize(self.weight, dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        onehot = F.one_hot(labels, cos.size(1)).float()
        logits = torch.cos(theta + self.margin * onehot) * self.scale
        return F.cross_entropy(logits, labels, label_smoothing=0.05)


def load_index() -> dict:
    idx = pd.read_csv(DATA / "images_index.csv")
    return dict(zip(idx.path, idx.idx))


def pair_tensors(pairs: pd.DataFrame, idx: dict):
    a = np.array([idx[p] for p in pairs.contactless], dtype=np.int64)
    b = np.array([idx[p] for p in pairs.contact], dtype=np.int64)
    y = pairs.label.values.astype(np.float32)
    return a, b, y


@torch.no_grad()
def embed_batch(model, feats, ix, tta: bool = True):
    model.eval()
    views = [(0, 0)]
    if tta:
        views += [(2, 0), (-2, 0), (0, 2), (0, -2)]
    out = []
    for dx, dy in views:
        x = torch.from_numpy(np.asarray(feats[ix])).float()
        if dx or dy:
            x = torch.roll(x, (dy, dx), dims=(2, 3))
        out.append(model(x))
    return F.normalize(torch.stack(out).mean(0), dim=1)


@torch.no_grad()
def score_pairs(model, feats, pairs, idx: dict, tta: bool = True, batch: int = 256):
    a, b, y = pair_tensors(pairs, idx)
    scores = []
    for i in range(0, len(a), batch):
        ea = embed_batch(model, feats, a[i:i + batch], tta=tta)
        eb = embed_batch(model, feats, b[i:i + batch], tta=tta)
        scores.append((ea * eb).sum(1).cpu().numpy())
    return np.concatenate(scores), y


def build_pools(meta: pd.DataFrame, train_ids: set):
    """id -> (contactless indices, contact indices) restricted to training identities."""
    cl: dict[int, list[int]] = {i: [] for i in train_ids}
    ct: dict[int, list[int]] = {i: [] for i in train_ids}
    for r in meta.itertuples():
        if r.finger_id not in train_ids:
            continue
        (cl if r.modality == "contactless" else ct)[r.finger_id].append(r.idx)
    missing = [i for i in train_ids if not cl[i] or not ct[i]]
    if missing:
        raise RuntimeError(f"identities without both modalities: {missing[:5]}")
    return cl, ct


def make_batch(rng: np.random.Generator, ids_pool, cl, ct, p: int, k: int):
    """Return (feature indices, global class labels, order guaranteed cl then ct)."""
    chosen = rng.choice(ids_pool, size=p, replace=False)
    x, y = [], []
    # all contactless first, then all contact: cmra_losses relies on this layout
    for j, fid in enumerate(chosen):
        for _ in range(k):
            x.append(cl[fid][rng.integers(len(cl[fid]))])
            y.append(j)
    for j, fid in enumerate(chosen):
        for _ in range(k):
            x.append(ct[fid][rng.integers(len(ct[fid]))])
            y.append(j)
    return np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64), chosen


def cmra_losses(emb, p: int, k: int, mode: str, lam_proto: float, lam_rel: float,
                temp: float = 0.1):
    """emb is ordered [P*K contactless | P*K contact]."""
    cl = F.normalize(emb[: p * k].view(p, k, -1).mean(1), dim=1)
    ct = F.normalize(emb[p * k:].view(p, k, -1).mean(1), dim=1)

    l_proto = torch.zeros((), dtype=emb.dtype)
    l_rel = torch.zeros((), dtype=emb.dtype)
    if mode in ("proto", "full"):
        l_proto = (1.0 - (cl * ct).sum(1)).mean()
    if mode in ("rel", "full"):
        # Each row is the distribution over identities "who is similar to i".
        # Matching the two distributions matches the relational layout without
        # forcing absolute coordinates to coincide.
        s_cl = cl @ cl.t()
        s_ct = ct @ ct.t()
        p_cl = F.softmax(s_cl / temp, dim=1)
        p_ct = F.softmax(s_ct / temp, dim=1)
        l_rel = 0.5 * (
            F.kl_div(p_cl.clamp_min(1e-8).log(), p_ct, reduction="batchmean")
            + F.kl_div(p_ct.clamp_min(1e-8).log(), p_cl, reduction="batchmean")
        )
    total = lam_proto * l_proto + lam_rel * l_rel
    return total, l_proto.detach(), l_rel.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["arc", "proto", "rel", "full"], default="full")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=64, help="batches per epoch")
    ap.add_argument("--p", type=int, default=16, help="identities per batch")
    ap.add_argument("--k", type=int, default=2, help="images per identity per modality")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--lam-proto", type=float, default=0.5)
    ap.add_argument("--lam-rel", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="cmra_full_s7")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(16)
    rng = np.random.default_rng(args.seed)

    feats = np.load(DATA / "scat.npy", mmap_mode="r")
    idx = load_index()
    meta = pd.read_csv(DATA / "metadata.csv")
    meta["idx"] = meta.path.map(idx)
    split = pd.read_csv(DATA / "finger_split.csv")
    train_ids = sorted(split[split.split == "train"].finger_id)
    cl, ct = build_pools(meta, set(train_ids))
    cmap = {fid: i for i, fid in enumerate(train_ids)}
    n_classes = len(train_ids)
    print(f"train identities {n_classes} | mode {args.mode} | P={args.p} K={args.k} "
          f"| steps/epoch {args.steps}")

    val = pd.read_csv(DATA / "pairs_val.csv")
    test = pd.read_csv(DATA / "pairs_test.csv")

    model = ScatterNet(args.emb)
    arc = ArcFace(args.emb, n_classes)
    params = list(model.parameters()) + list(arc.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_eer, best_state, history = 1.0, None, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_total = run_proto = run_rel = 0.0
        for _ in range(args.steps):
            x_ix, local_y, chosen = make_batch(rng, train_ids, cl, ct, args.p, args.k)
            x = torch.from_numpy(np.asarray(feats[x_ix])).float()
            if np.random.rand() < 0.8:
                dx, dy = np.random.randint(-2, 3, 2)
                x = torch.roll(x, (int(dy), int(dx)), dims=(2, 3))

            emb = model(x)
            global_y = torch.tensor([cmap[int(chosen[j])] for j in local_y], dtype=torch.long)
            loss_arc = arc(emb, global_y)
            loss_extra, lp, lr_ = cmra_losses(
                emb, args.p, args.k, args.mode, args.lam_proto, args.lam_rel, args.temp)
            loss = loss_arc + loss_extra

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            run_total += loss.item()
            run_proto += float(lp)
            run_rel += float(lr_)
        sched.step()

        sv, yv = score_pairs(model, feats, val, idx, tta=False)
        eer = compute_eer(yv, sv)
        tar = tar_at_far(yv, sv)
        history.append(dict(epoch=epoch,
                            loss=run_total / args.steps,
                            l_proto=run_proto / args.steps,
                            l_rel=run_rel / args.steps,
                            val_eer=float(eer), val_tar1=float(tar)))
        tag = ""
        if eer < best_eer:
            best_eer = eer
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            tag = "  <- best"
        print(f"epoch {epoch:2d} | loss {run_total/args.steps:.4f} "
              f"| proto {run_proto/args.steps:.4f} rel {run_rel/args.steps:.4f} "
              f"| val EER {eer:.4f} | TAR@1% {tar:.4f} | {time.time()-t0:.0f}s{tag}", flush=True)

    model.load_state_dict(best_state)
    run_dir = RESULTS / args.out
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")

    summary = dict(config=vars(args), best_val_eer=float(best_eer), history=history)
    for split_name, pairs in [("val", val), ("test", test)]:
        s, y = score_pairs(model, feats, pairs, idx, tta=True)
        out = pairs[["contactless", "contact", "label", "finger_a", "finger_b"]].copy()
        out["score_cmra"] = s
        out.to_csv(run_dir / f"scores_{split_name}.csv", index=False)
        summary[split_name] = dict(
            eer=float(compute_eer(y, s)),
            tar1=float(tar_at_far(y, s)),
            tar01=float(tar_at_far(y, s, far=0.001)),
        )
        print(f"[{split_name}] CMRA({args.mode}) EER {summary[split_name]['eer']:.4f} "
              f"TAR@1% {summary[split_name]['tar1']:.4f} "
              f"TAR@0.1% {summary[split_name]['tar01']:.4f}", flush=True)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
