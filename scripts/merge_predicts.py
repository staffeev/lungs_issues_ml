import sys
import pandas as pd
import numpy as np
from tqdm import tqdm


if __name__== '__main__':
    result = sys.argv[1]
    files = [pd.read_csv(f"outputs/{fname}.csv") for fname in sys.argv[1:]]
    base = files[0]
    features = pd.concat([i["target_feature"] for i in files], axis=1).to_numpy()
    features = np.round(np.median(features, axis=1))
    base["target_feature"] = pd.Series(features, dtype=int)
    base.to_csv("outputs/merged.csv",  index=False)
