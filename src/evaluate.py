# src/evaluate.py
import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2

def tar_at_far(labels, scores, far=0.01):
    fpr, tpr, _ = roc_curve(labels, scores)
    return tpr[np.searchsorted(fpr, far, side="right") - 1]