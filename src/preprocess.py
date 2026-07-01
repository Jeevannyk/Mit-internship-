import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from tqdm import tqdm


def prepare_data(df, augment_minority=True):
    # augment_minority=True keeps the old RandomOverSampler behaviour (used by
    # main.py). The augmentation experiment passes False to get a clean split,
    # then applies its own None/SMOTE/CTGAN strategy via src/augment.py.

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

    # Force each column to numeric IN PLACE. The old version built a separate
    # dict + DataFrame copy, tripling memory and OOM-ing on 12 GB machines.
    for col in tqdm(feature_cols, desc="Converting columns to numeric"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace inf/NaN on the feature columns only (Label is a string).
    print("Cleaning inf/NaN...")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Clip to float32 range AND downcast to float32 — halves memory vs float64,
    # which is what lets dedup + training fit on a 12 GB (Colab) machine.
    float32_max = np.finfo(np.float32).max
    print("Clipping + downcasting to float32...")
    df[feature_cols] = df[feature_cols].clip(-float32_max, float32_max).astype("float32")

    # Drop exact-duplicate flows. CIC-IDS2018 has many identical rows; with a
    # random split copies leak into BOTH train and test, faking ~100% accuracy.
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows ({len(df)} remain)")

    X = df.drop("Label", axis=1)
    y_str = df["Label"]

    # Save the exact training feature list/order so inference can reindex to it
    # (tree models use positional columns — wrong order = silent bad predictions).
    os.makedirs("models", exist_ok=True)
    joblib.dump(list(X.columns), "models/feature_names.pkl")

    # Encode string labels (Benign, SQL Injection, DoS-Hulk ...) to integers.
    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)

    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
    )

    # Oversample minority attack classes up to 3000 samples each (optional).
    if augment_minority:
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
