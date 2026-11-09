# src/cache_images.py
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from preprocess import preprocess

DATA = Path(__file__).resolve().parent / "data"


def main():
    df = pd.read_csv(DATA / "metadata.csv")
    used = set()
    for s in ["train", "val", "test"]:
        p = pd.read_csv(DATA / f"pairs_{s}.csv")
        used |= set(p.contactless) | set(p.contact)
    sub = df[df.path.isin(used)].reset_index(drop=True)
    print(len(sub), "images to cache")

    imgs = Parallel(n_jobs=-1)(
        delayed(preprocess)(r.path, r.modality) for r in tqdm(list(sub.itertuples())))
    arr = np.stack(imgs).astype(np.float16)            # (N, 192, 192)
    np.save(DATA / "images.npy", arr)
    pd.DataFrame({"path": sub.path, "idx": range(len(sub))}).to_csv(
        DATA / "images_index.csv", index=False)
    print("saved", arr.shape, f"({arr.nbytes / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()