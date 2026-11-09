# src/siamese.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Branch(nn.Module):
    """Small CNN: scattering features (81 x 48 x 48) -> 16-number code.
    Global pooling at the end makes it independent of where the pattern sits."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(81),
            nn.Conv2d(81, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2),   # 48 -> 24
            nn.Conv2d(24, 48, 5, padding=2), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2),   # 24 -> 12
            nn.Conv2d(48, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 16))

    def forward(self, f):
        x = self.conv(f)                                          # (B, 64, 12, 12)
        x = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)   # global avg + max -> 128
        return F.normalize(self.head(x), dim=1)                   # unit-length embedding


class Siamese(nn.Module):
    def __init__(self, shared=True):
        super().__init__()
        self.b1 = Branch()
        self.b2 = self.b1 if shared else Branch()                 # shared=True is the paper's setup

    def forward(self, fa, fb):                                    # fa = contactless, fb = contact
        return self.b1(fa), self.b2(fb)


def contrastive_loss(ea, eb, y, margin=1.0):
    """y = 1 for the same finger, 0 for different fingers (standard form)."""
    d = F.pairwise_distance(ea, eb)
    return (y * d.pow(2) + (1 - y) * F.relu(margin - d).pow(2)).mean()