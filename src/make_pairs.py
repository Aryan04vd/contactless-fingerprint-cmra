# src/make_pairs.py
from pathlib import Path
import pandas as pd, random

random.seed(42)
DATA = Path(__file__).resolve().parent / "data"
df = pd.read_csv(DATA / "metadata.csv")

# 1) split FINGERS (not images): 60% train, 10% val, 30% test
fingers = sorted(df.finger_id.unique())
random.shuffle(fingers)
n = len(fingers)
splits = {
    "train": fingers[: int(0.6 * n)],
    "val":   fingers[int(0.6 * n): int(0.7 * n)],
    "test":  fingers[int(0.7 * n):],
}
pd.DataFrame([(f, s) for s, fl in splits.items() for f in fl],
             columns=["finger_id", "split"]).to_csv(DATA / "finger_split.csv", index=False)

def make_pairs(finger_list, impostor_ratio):
    sub = df[df.finger_id.isin(finger_list)]
    cl = sub[sub.modality == "contactless"]
    ct = sub[sub.modality == "contact"]
    pairs = []
    for fid in finger_list:
        for session in ["first_session", "second_session"]:
            a = cl[(cl.finger_id == fid) & (cl.session == session)]
            b = ct[(ct.finger_id == fid) & (ct.session == session)]
            for _, ra in a.iterrows():
                for _, rb in b.iterrows():                      # genuine pairs
                    pairs.append((ra.path, rb.path, 1, fid, fid))
                for _ in range(impostor_ratio * len(b)):        # impostor pairs
                    other = random.choice([f for f in finger_list if f != fid])
                    pool = ct[(ct.finger_id == other) & (ct.session == session)]
                    if pool.empty:
                        pool = ct[ct.finger_id == other]
                    rb = pool.sample(1).iloc[0]
                    pairs.append((ra.path, rb.path, 0, fid, other))
    return pd.DataFrame(pairs, columns=["contactless", "contact", "label",
                                        "finger_a", "finger_b"])

for name, fl in splits.items():
    p = make_pairs(fl, impostor_ratio=1)
    p.to_csv(DATA / f"pairs_{name}.csv", index=False)
    print(f"{name:5s}: {len(fl):3d} fingers, {len(p):6d} pairs, "
          f"{p.label.sum()} genuine, {(p.label == 0).sum()} impostor")

# safety check: no finger shared between splits
assert not set(splits["train"]) & set(splits["test"])
assert not set(splits["train"]) & set(splits["val"])
assert not set(splits["val"]) & set(splits["test"])
print("OK: no finger appears in more than one split")