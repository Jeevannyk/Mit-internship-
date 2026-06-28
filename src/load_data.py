import glob
import pandas as pd

def load_dataset():
    files = glob.glob("data/*.csv")

    dfs = [pd.read_csv(file) for file in files]

    df = pd.concat(dfs, ignore_index=True)

    return df