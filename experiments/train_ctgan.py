"""Train CTGAN on all 9 day files and save the model + synthetic dataset.

Run this ONCE before run_matrix.py so CTGAN doesn't retrain every experiment.

Outputs:
    models/ctgan.pkl              -- trained CTGAN model
    models/ctgan_label_encoder.pkl -- label encoder (class index <-> string)
    results/ctgan_synthetic.csv   -- synthetic minority rows (ready to concat)

Usage:
    venv\Scripts\python.exe experiments/train_ctgan.py
    venv\Scripts\python.exe experiments/train_ctgan.py --epochs 200 --fit-cap 5000
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from ctgan import CTGAN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import prepare_data

ALL_FILES = [f"data/{d}.csv" for d in [
    "02-14-2018", "02-15-2018", "02-16-2018", "02-21-2018", "02-22-2018",
    "02-23-2018", "02-28-2018", "03-01-2018", "03-02-2018",
]]


def main(epochs, fit_cap, target, seed):
    print(f"Loading {len(ALL_FILES)} files...")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in ALL_FILES],
                   ignore_index=True)
    df = df[df["Protocol"] != "Protocol"]

    X_train, X_test, y_train, y_test, le = prepare_data(df, augment_minority=False)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Save label encoder so downstream scripts can decode class indices.
    joblib.dump(le, "models/ctgan_label_encoder.pkl")

    feature_cols = list(X_train.columns)
    df_train = X_train.copy()
    df_train["Label"] = y_train

    # Cap per class so CTGAN training stays feasible.
    capped = (
        df_train.sample(frac=1, random_state=seed)
        .groupby("Label", group_keys=False)
        .head(fit_cap)
    )
    print(f"Training CTGAN on {len(capped)} rows ({epochs} epochs)...")

    ctgan = CTGAN(epochs=epochs, verbose=True)
    ctgan.fit(capped, discrete_columns=["Label"])

    joblib.dump(ctgan, "models/ctgan.pkl")
    print("Saved -> models/ctgan.pkl")

    # Generate synthetic minority rows.
    benign = le.transform(["Benign"])[0]
    counts = pd.Series(y_train).value_counts()
    plan = {
        cls: target
        for cls, cnt in counts.items()
        if cls != benign and cnt < target
    }

    synthetic = []
    for cls, tgt in plan.items():
        need = tgt - int(counts.get(cls, 0))
        if need <= 0:
            continue
        sampled = ctgan.sample(need * 2, condition_column="Label",
                               condition_value=cls)
        sampled = sampled[sampled["Label"] == cls].head(need)
        synthetic.append(sampled)
        print(f"  +{len(sampled)} synthetic rows for class '{le.classes_[cls]}'")

    if not synthetic:
        print("No synthetic rows needed — all classes already at target.")
        return

    syn_df = pd.concat(synthetic, ignore_index=True)
    # Decode numeric label back to string for readability.
    syn_df["Label_str"] = le.inverse_transform(syn_df["Label"].astype(int))
    syn_df.to_csv("results/ctgan_synthetic.csv", index=False)
    print(f"Saved {len(syn_df)} synthetic rows -> results/ctgan_synthetic.csv")
    print(syn_df["Label_str"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--fit-cap", type=int, default=3000,
                    help="max rows per class for CTGAN training")
    ap.add_argument("--target", type=int, default=3000,
                    help="target rows per minority class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.epochs, args.fit_cap, args.target, args.seed)
