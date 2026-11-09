"""Identity-aware scattering embedding (ISE) for contactless-to-contact matching.

This replaces the plain contrastive Siamese branch in the VerifNet-style baseline
with an ArcFace-trained embedding on the same cached scattering features.
"""
from __future__ import annotations

import argparse
import json
import math
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


def load_index():
    idx = pd.read_csv(DATA / "images_index.csv")
    return dict(zip(idx.path, idx.idx))


def pair_tensors(pairs: pd.DataFrame, idx: dict):
    a = np.array([idx[p] for p in pairs.contactless], dtype=np.int64)
    b = np.array([idx[p] for p in pairs.contact], dtype=np.int64)
    y = pairs.label.values.astype(np.float32)
    return a, b, y


def get_feats(feats, ix, shift: int = 0):
    x = torch.from_numpy(np.asarray(feats[ix])).float()
    if shift:
        dx, dy = np.random.randint(-shift, shift + 1, 2)
        x = torch.roll(x, (int(dy), int(dx)), dims=(2, 3))
    return x


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
def score_pairs(model, feats, pairs, idx, tta: bool = True, batch: int = 256):
    a, b, y = pair_tensors(pairs, idx)
    scores = []
    for i in range(0, len(a), batch):
        ea = embed_batch(model, feats, a[i:i + batch], tta=tta)
        eb = embed_batch(model, feats, b[i:i + batch], tta=tta)
        scores.append((ea * eb).sum(1).cpu().numpy())
    return np.concatenate(scores), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--out", type=str, default="ise_seed7")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(16)

    feats = np.load(DATA / "scat.npy", mmap_mode="r")
    idx = load_index()
    meta = pd.read_csv(DATA / "metadata.csv")
    path2id = dict(zip(meta.path, meta.finger_id))

    train = pd.read_csv(DATA / "pairs_train.csv")
    tr_paths = sorted(set(train.contactless) | set(train.contact))
    tr_idx = np.array([idx[p] for p in tr_paths], dtype=np.int64)
    tr_y0 = np.array([path2id[p] for p in tr_paths])
    classes = sorted(set(tr_y0))
    cmap = {c: i for i, c in enumerate(classes)}
    tr_y = np.array([cmap[y] for y in tr_y0], dtype=np.int64)
    n_classes = len(classes)
    print(f"train images {len(tr_idx)} | identities {n_classes} | feature shape {feats.shape}")

    val = pd.read_csv(DATA / "pairs_val.csv")
    test = pd.read_csv(DATA / "pairs_test.csv")

    model = ScatterNet(args.emb)
    arc = ArcFace(args.emb, n_classes)
    opt = torch.optim.AdamW(list(model.parameters()) + list(arc.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_eer, best_state, history = 1.0, None, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = np.random.permutation(len(tr_idx))
        total = 0.0
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            x = get_feats(feats, tr_idx[b], shift=2)
            y = torch.from_numpy(tr_y[b]).long()
            emb = model(x)
            loss = arc(emb, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(arc.parameters()), 1.0)
            opt.step()
            total += loss.item() * len(b)
        sched.step()

        sv, yv = score_pairs(model, feats, val, idx, tta=False)
        eer = compute_eer(yv, sv)
        tar = tar_at_far(yv, sv)
        history.append(dict(epoch=epoch, loss=total / len(order), val_eer=float(eer), val_tar1=float(tar)))
        flag = ""
        if eer < best_eer:
            best_eer = eer
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = "  <- best"
        print(f"epoch {epoch:2d} | loss {total/len(order):.4f} | val EER {eer:.4f} "
              f"| TAR@1% {tar:.4f} | {time.time()-t0:.0f}s{flag}")

    model.load_state_dict(best_state)
    run_dir = RESULTS / args.out
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")

    summary = dict(config=vars(args), best_val_eer=float(best_eer), history=history)
    for split, pairs in [("val", val), ("test", test)]:
        s, y = score_pairs(model, feats, pairs, idx, tta=True)
        out = pairs[["contactless", "contact", "label", "finger_a", "finger_b"]].copy()
        out["score_ise"] = s
        out.to_csv(run_dir / f"scores_{split}_ise.csv", index=False)
        summary[split] = dict(
            eer=float(compute_eer(y, s)),
            tar1=float(tar_at_far(y, s)),
            tar01=float(tar_at_far(y, s, far=0.001)),
        )
        print(f"[{split}] ISE EER {summary[split]['eer']:.4f} "
              f"TAR@1% {summary[split]['tar1']:.4f} TAR@0.1% {summary[split]['tar01']:.4f}")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
