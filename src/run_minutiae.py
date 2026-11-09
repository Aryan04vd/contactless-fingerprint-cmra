# src/run_minutiae.py
from pathlib import Path
import pickle
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from minutiae import extract_minutiae, match_score
from evaluate import compute_eer, tar_at_far

DATA = Path(__file__).resolve().parent / "data"
CACHE = DATA / "minutiae_cache.pkl"


def main():
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    for split in ["val", "test"]:
        pairs = pd.read_csv(DATA / f"pairs_{split}.csv")

        items = {(p, "contactless") for p in pairs.contactless} | \
                {(p, "contact") for p in pairs.contact}
        todo = [it for it in items if it[0] not in cache]
        print(f"[{split}] extracting minutiae from {len(todo)} images...")
        results = Parallel(n_jobs=-1)(delayed(extract_minutiae)(p, m) for p, m in tqdm(todo))
        cache.update({it[0]: r for it, r in zip(todo, results)})
        pickle.dump(cache, open(CACHE, "wb"))

        print(f"[{split}] matching {len(pairs)} pairs...")
        scores = Parallel(n_jobs=-1, batch_size=32)(
            delayed(match_score)(cache[a], cache[b])
            for a, b in tqdm(list(zip(pairs.contactless, pairs.contact))))
        pairs["score_mm"] = scores
        pairs.to_csv(DATA / f"scores_{split}.csv", index=False)

        print(f"[{split}] minutiae-only EER = {compute_eer(pairs.label, pairs.score_mm):.3f}   "
              f"TAR@FAR=1% = {tar_at_far(pairs.label, pairs.score_mm):.3f}")


if __name__ == "__main__":      # required on Windows for parallel code
    main()