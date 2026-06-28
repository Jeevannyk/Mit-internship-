import pandas as pd
import numpy as np
from tqdm import tqdm

COLS_TO_DROP = [
    "Flow ID", "Src IP", "Dst IP", "Timestamp",
    "Fwd Seg Size Min", "Init Fwd Win Byts", "Fwd Act Data Pkts",
    "Label",
]


def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    df = df[df.iloc[:, 0] != df.columns[0]]  # remove duplicate header rows

    feature_cols = list(df.columns)
    for col in tqdm(feature_cols, desc="Converting columns", leave=False):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    float32_max = np.finfo(np.float32).max
    df[feature_cols] = df[feature_cols].clip(-float32_max, float32_max)

    return df
