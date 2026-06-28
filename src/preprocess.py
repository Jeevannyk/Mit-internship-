import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from tqdm import tqdm


def prepare_data(df):
    # Drop columns that are identifiers / not useful as features.
    # errors="ignore" => no crash if a column isn't present in this dataset.
    cols_to_drop = [
        # Non-numeric identifiers (must drop or sklearn crashes on strings)
        "Flow ID",
        "Src IP",
        "Dst IP",
        "Timestamp",
        # Low-importance / near-zero variance columns per feature_importance.csv
        "Fwd Seg Size Min",
        "Init Fwd Win Byts",
        "Fwd Act Data Pkts",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    feature_cols = [c for c in df.columns if c != "Label"]

    # Force each column to numeric (stray header rows leave object dtype).
    # tqdm shows one bar per column so you see progress instead of a blank screen.
    converted = {}
    for col in tqdm(feature_cols, desc="Converting columns to numeric"):
        converted[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_cols] = pd.DataFrame(converted, index=df.index)

    # Replace inf and NaN.
    print("Cleaning inf/NaN...")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # Clip values too large for float32 (sklearn converts internally).
    float32_max = np.finfo(np.float32).max
    print("Clipping float32 overflow...")
    df[feature_cols] = df[feature_cols].clip(-float32_max, float32_max)

    X = df.drop("Label", axis=1)
    y_str = df["Label"]

    # Encode string labels (Benign, SQL Injection, DoS-Hulk ...) to integers.
    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)

    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
    )

    # Oversample minority attack classes up to 5000 samples each.
    benign_label = le.transform(["Benign"])[0]
    counts = pd.Series(y_train_enc).value_counts()
    strategy = {
        cls: 3000
        for cls in counts.index
        if cls != benign_label and counts[cls] < 3000
    }

    if strategy:
        print(f"Oversampling: {[le.classes_[c] for c in strategy]}")
        ros = RandomOverSampler(random_state=42, sampling_strategy=strategy)
        X_train, y_train_enc = ros.fit_resample(X_train, y_train_enc)
        print(f"After oversampling — total rows: {len(y_train_enc)}")

    return X_train, X_test, y_train_enc, y_test_enc, le
