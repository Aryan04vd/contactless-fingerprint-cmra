# src/train.py
# usage:   python train.py NAME SEED AUG
#   e.g.   python train.py base 1 0      (no rotation augmentation)
#          python train.py rot 1 1       (with rotation augmentation)
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from siamese import Siamese, contrastive_loss
from evaluate import compute_eer, tar_at_far

NAME = sys.argv[1] if len(sys.argv) > 1 else "base"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
AUG = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "runs" / f"{NAME}_seed{SEED}"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- settings ----------------
SHARED = True
EPOCHS = 40
BATCH = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8
PAIRS_PER_EPOCH = 6000
SHIFT = 2                  # random shift in feature cells (1 cell = 4 pixels)
P_ROT = 0.7                # chance a contactless training image is swapped for a rotated copy
# ------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

feats = torch.from_numpy(np.load(DATA / "scat.npy"))             # float16 (N, 81, 48, 48)
index = dict(pd.read_csv(DATA / "images_index.csv").values)
if AUG:
    aug = torch.from_numpy(np.load(DATA / "aug_scat.npy"))
    aug_map = torch.from_numpy(np.load(DATA / "aug_map.npy"))


def load_pairs(split):
    p = pd.read_csv(DATA / f"pairs_{split}.csv")
    return (torch.tensor(p.contactless.map(index).values),
            torch.tensor(p.contact.map(index).values),
            torch.tensor(p.label.values, dtype=torch.float32))


def shift_roll(x, shift):
    if shift:
        dx, dy = np.random.randint(-shift, shift + 1, 2)
        x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
    return x


def get(idx, shift=0):
    return shift_roll(feats[idx].float(), shift).to(dev)


def get_cl_train(idx, shift=0):
    """Contactless training input; with AUG, sometimes swap in a rotated copy."""
    x = feats[idx].float()
    if AUG:
        for n, i in enumerate(idx.tolist()):
            if np.random.rand() < P_ROT:
                r = int(aug_map[i, np.random.randint(aug_map.shape[1])])
                if r >= 0:
                    x[n] = aug[r].float()
    return shift_roll(x, shift).to(dev)


@torch.no_grad()
def score_split(model, split):
    model.eval()
    a, b, y = load_pairs(split)
    out = []
    for i in range(0, len(a), 256):
        ea, eb = model(get(a[i:i + 256]), get(b[i:i + 256]))
        out.append((-F.pairwise_distance(ea, eb)).cpu())          # higher = more similar
    return torch.cat(out).numpy(), y.numpy()


model = Siamese(shared=SHARED).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
a_tr, b_tr, y_tr = load_pairs("train")
print(f"run {OUT.name} | aug={AUG} | device {dev} | train pairs {len(a_tr)}")

best_eer, bad = 1.0, 0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    order = torch.randperm(len(a_tr))[:PAIRS_PER_EPOCH]
    total = 0.0
    for i in range(0, len(order), BATCH):
        ix = order[i:i + BATCH]
        ea, eb = model(get_cl_train(a_tr[ix], SHIFT), get(b_tr[ix], SHIFT))
        loss = contrastive_loss(ea, eb, y_tr[ix].to(dev))
        if not torch.isfinite(loss):
            print("non-finite loss, stopping"); raise SystemExit
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * len(ix)
    s, y = score_split(model, "val")
    eer = compute_eer(y, s)
    flag = ""
    if eer < best_eer:
        best_eer, bad, flag = eer, 0, "  <- best, saved"
        torch.save(model.state_dict(), OUT / "best_model.pt")
    else:
        bad += 1
    print(f"epoch {epoch:2d} | train loss {total / len(order):.4f} | val EER {eer:.3f} "
          f"| {time.time() - t0:.0f}s{flag}")
    if bad >= PATIENCE:
        print("early stop"); break

model.load_state_dict(torch.load(OUT / "best_model.pt", map_location=dev))
for split in ["val", "test"]:
    s, y = score_split(model, split)
    pd.DataFrame({"label": y.astype(int), "score_dl": s}).to_csv(
        OUT / f"scores_{split}_dl.csv", index=False)
    print(f"[{split}] wavelet+Siamese EER = {compute_eer(y, s):.3f}   "
          f"TAR@FAR=1% = {tar_at_far(y, s):.3f}")