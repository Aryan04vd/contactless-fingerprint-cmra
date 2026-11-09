# src/cache_aug.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from kymatio.torch import Scattering2D

DATA = Path(__file__).resolve().parent / "data"
COPIES = 3
rng = np.random.default_rng(0)

imgs = np.load(DATA / "images.npy")                           # (5952, 192, 192)
index = dict(pd.read_csv(DATA / "images_index.csv").values)
train = pd.read_csv(DATA / "pairs_train.csv")
cl_idx = sorted({index[p] for p in train.contactless})        # TRAINING contactless images only
print(len(cl_idx), "training contactless images x", COPIES, "rotated copies")

aug_map = -np.ones((len(imgs), COPIES), dtype=np.int64)       # image index -> row in aug_scat
rows, items = 0, []
for i in cl_idx:
    for k in range(COPIES):
        angle = rng.uniform(10, 45) * rng.choice([-1, 1])
        aug_map[i, k] = rows; rows += 1
        items.append((i, angle))

scat = Scattering2D(J=2, shape=(192, 192))
out = np.empty((rows, 81, 48, 48), dtype=np.float16)
with torch.no_grad():
    for s in tqdm(range(0, rows, 32)):
        batch = []
        for i, a in items[s:s + 32]:
            M = cv2.getRotationMatrix2D((96, 96), a, 1.0)
            batch.append(cv2.warpAffine(imgs[i].astype(np.float32), M, (192, 192),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        x = torch.from_numpy(np.stack(batch)).unsqueeze(1)
        f = scat(x).reshape(len(batch), 81, 48, 48)
        f[:, 1:] = torch.log(f[:, 1:].clamp(min=0) + 1e-2)    # same recipe as cache_scat.py
        out[s:s + len(batch)] = f.numpy().astype(np.float16)

assert np.isfinite(out.astype(np.float32)).all()
np.save(DATA / "aug_scat.npy", out)
np.save(DATA / "aug_map.npy", aug_map)
print("saved", out.shape, f"({out.nbytes / 1e9:.1f} GB)")