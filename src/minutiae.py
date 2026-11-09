# src/minutiae.py
import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from preprocess import preprocess

SIZE_M, PERIOD_M = 400, 8.0      # minutiae branch works at larger scale


def orientation_field(gray, sigma=8):
    """Ridge direction at every pixel, in radians (0 to pi)."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy) + np.pi / 2
    return np.mod(theta, np.pi)


def extract_minutiae(path, modality, flip=None):
    """Rows: [x, y, orientation, type]; type 1 = ending, 3 = bifurcation."""
    img = preprocess(path, modality, size=SIZE_M, target_period=PERIOD_M, flip=flip)
    gray = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.5)

    # mask = where the image is darker than the white background, minus a wide border
    f = gray.astype(np.float32)
    local_mean = cv2.GaussianBlur(f, (0, 0), 8)
    mask = (local_mean < 215).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((41, 41), np.uint8),       # ~20 px = 2-3 ridges
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)   # <- the fix

    binary = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 3) > 0
    binary = remove_small_objects(binary, max_size=29)
    skel = skeletonize(binary).astype(np.uint8)
    theta = orientation_field(smooth)

    pts = []
    for y, x in zip(*np.nonzero((skel > 0) & (mask > 0))):
        p = [skel[y-1, x], skel[y-1, x+1], skel[y, x+1], skel[y+1, x+1],
             skel[y+1, x], skel[y+1, x-1], skel[y, x-1], skel[y-1, x-1]]
        cn = sum(abs(int(p[i]) - int(p[(i + 1) % 8])) for i in range(8)) // 2
        if cn in (1, 3):
            pts.append((x, y, theta[y, x], cn))
    pts = np.array(pts, dtype=np.float32).reshape(-1, 4)

    if len(pts) > 1:                                  # drop minutiae crowded together
        D = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
        np.fill_diagonal(D, 1e9)
        pts = pts[(D > 8).all(axis=1)]
    return pts


def match_score(A, B, d_thr=8, a_thr=np.deg2rad(20), n_anchors=25,
                overlap=False, min_overlap=15):
    """A = contactless minutiae, B = contact minutiae.
    With overlap=True, only contact minutiae that land inside the contactless
    footprint count against the score (partial overlap is handled fairly)."""
    if len(A) < 4 or len(B) < 4:
        return 0.0
    x0, x1 = A[:, 0].min() - 10, A[:, 0].max() + 10       # contactless footprint
    y0, y1 = A[:, 1].min() - 10, A[:, 1].max() + 10
    ia = np.linspace(0, len(A) - 1, min(len(A), n_anchors)).astype(int)
    jb = np.linspace(0, len(B) - 1, min(len(B), n_anchors)).astype(int)
    best = 0.0
    for i in ia:
        for j in jb:
            for flip in (0.0, np.pi):
                dth = A[i, 2] - B[j, 2] + flip
                c, s = np.cos(dth), np.sin(dth)
                dx, dy = B[:, 0] - B[j, 0], B[:, 1] - B[j, 1]
                bx = c * dx - s * dy + A[i, 0]
                by = s * dx + c * dy + A[i, 1]
                bt = np.mod(B[:, 2] + dth, np.pi)
                D = np.hypot(A[:, None, 0] - bx[None], A[:, None, 1] - by[None])
                da = np.abs(A[:, None, 2] - bt[None])
                da = np.minimum(da, np.pi - da)
                ok = (D < d_thr) & (da < a_thr)
                if overlap:
                    inside = (bx >= x0) & (bx <= x1) & (by >= y0) & (by <= y1)
                    nb = int(inside.sum())
                    if nb < min_overlap:
                        continue
                    ok = ok[:, inside]
                else:
                    nb = len(B)
                n = min(ok.any(1).sum(), ok.any(0).sum())
                best = max(best, n * n / (len(A) * nb))
    return best