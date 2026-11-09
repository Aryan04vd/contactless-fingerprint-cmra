"""Compute simple image-quality features for cached preprocessed fingerprints."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
DATA = SRC / "data"
RESULTS = SRC.parent / "results"
RESULTS.mkdir(exist_ok=True)


def image_quality(img: np.ndarray) -> dict:
    f = img.astype(np.float32)
    fg = f < 0.5
    if fg.sum() < 100:
        fg = np.ones_like(fg, dtype=bool)
    vals = f[fg]
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)
    # structure-tensor coherence
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), 4)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), 4)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), 4)
    tmp = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2)
    l1 = 0.5 * (gxx + gyy + tmp)
    l2 = 0.5 * (gxx + gyy - tmp)
    coh = (l1 - l2) / (l1 + l2 + 1e-6)
    hist, _ = np.histogram(vals, bins=32, range=(-3, 3), density=False)
    p = hist / max(hist.sum(), 1)
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return dict(
        fg_frac=float(fg.mean()),
        contrast=float(vals.std()),
        blur=float(cv2.Laplacian(f, cv2.CV_32F).var()),
        coherence=float(coh[fg].mean()),
        grad=float(gmag[fg].mean()),
        entropy=ent,
    )


def main():
    imgs = np.load(DATA / "images.npy", mmap_mode="r")
    idx = pd.read_csv(DATA / "images_index.csv")
    rows = []
    for r in idx.itertuples():
        q = image_quality(np.asarray(imgs[int(r.idx)]))
        q["path"] = r.path
        rows.append(q)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "quality.csv", index=False)
    print(out.describe().round(4).to_string())
    print(f"saved {RESULTS / 'quality.csv'} ({len(out)} images)")


if __name__ == "__main__":
    main()
