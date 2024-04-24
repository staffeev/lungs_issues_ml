import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import json


config_file = 'merge_config.json'


def read_config():
    with open(config_file, 'r') as f:
        config = json.load(f)
        return config['output'], config['files']


def main():
    output, files = read_config()
    tables = list(map(pd.read_csv, files))

    features = pd.concat([i["target_feature"] for i in files], axis=1).mode(axis=1).max()

    df = tables[0]    
    df["target_feature"] = pd.Series(features, dtype=int)

    df.to_csv("outputs/merged.csv",  index=False)


if __name__== '__main__':
    main()
