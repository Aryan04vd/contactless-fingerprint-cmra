# src/cache_scat.py
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from kymatio.torch import Scattering2D

DATA = Path(__file__).resolve().parent / "data"

imgs = np.load(DATA / "images.npy")                       # (5952, 192, 192)
N = len(imgs)
scat = Scattering2D(J=2, shape=(192, 192))                # fixed wavelets, nothing to learn
out = np.empty((N, 81, 48, 48), dtype=np.float16)

with torch.no_grad():
    for i in tqdm(range(0, N, 32)):
        x = torch.from_numpy(imgs[i:i + 32]).float().unsqueeze(1)
        s = scat(x).reshape(x.size(0), 81, 48, 48)
        s[:, 1:] = torch.log(s[:, 1:].clamp(min=0) + 1e-2)   # magnitudes only; clamp tiny negatives
        # channel 0 (low-pass image) can be negative, so it stays as it is
        out[i:i + len(x)] = s.numpy().astype(np.float16)

assert np.isfinite(out.astype(np.float32)).all(), "still has non-finite values!"
np.save(DATA / "scat.npy", out)
print("saved", out.shape, f"({out.nbytes / 1e9:.1f} GB)")