# src/preprocess.py
import cv2
import numpy as np

SIZE = 192                 # default canvas (deep learning branch)
TARGET_PERIOD = 4.0        # default ridge-to-ridge distance after resizing (px)
MEDIAN_PERIOD = {"contactless": 8.7, "contact": 7.7}   # measured on your data
INVERT_CONTACTLESS = True


def ridge_period(gray):
    """Diagnostic only: estimate ridge spacing (px) with an FFT."""
    h, w = gray.shape
    s = min(h, w, 192)
    y0, x0 = (h - s) // 2, (w - s) // 2
    patch = gray[y0:y0 + s, x0:x0 + s].astype(np.float32)
    patch -= patch.mean()
    patch *= np.outer(np.hanning(s), np.hanning(s))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(patch)))
    yy, xx = np.indices(mag.shape)
    r = np.hypot(yy - s // 2, xx - s // 2).astype(int)
    radial = np.bincount(r.ravel(), mag.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    radial = radial * np.arange(len(radial))
    lo, hi = max(s // 30, 3), s // 4
    return s / (lo + int(np.argmax(radial[lo:hi])))

FLIP_CONTACTLESS = True        # we will set this after the test below
ROTATE_CONTACTLESS = True       # turn contactless 90 degrees clockwise (found by diagnose_rotation)

def preprocess(path, modality, size=SIZE, target_period=TARGET_PERIOD, flip=None):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if flip is None:
        flip = FLIP_CONTACTLESS

    if modality == "contactless":
        if flip:
            img = img[:, ::-1].copy()                        # mirror left-right
        if ROTATE_CONTACTLESS:
            img = np.rot90(img, k=-1).copy()                 # 90 degrees clockwise
        f = img.astype(np.float32)
        f = f - cv2.GaussianBlur(f, (0, 0), 15)              # remove uneven lighting
        img = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)
        if INVERT_CONTACTLESS:
            img = 255 - img                                  # dark ridges like contact

    # one FIXED scale per modality
    scale = target_period / MEDIAN_PERIOD[modality]
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)

    # centre on a white square canvas (pad if smaller, crop if larger)
    canvas = np.full((size, size), 255, np.uint8)
    h, w = img.shape
    ys, xs = max((size - h) // 2, 0), max((size - w) // 2, 0)
    yc, xc = max((h - size) // 2, 0), max((w - size) // 2, 0)
    hh, ww = min(h, size), min(w, size)
    canvas[ys:ys + hh, xs:xs + ww] = img[yc:yc + hh, xc:xc + ww]

    out = canvas.astype(np.float32)
    return (out - out.mean()) / (out.std() + 1e-6)