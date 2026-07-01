"""Group 1 experiment runner.

Runs the full comparison matrix:
    augmentation {none, smote, ctgan}  x  model {rf, xgb}  x  features {all, shap}

For every cell it records: macro-F1, accuracy, per-class F1 for the rare attacks,
inference latency, and model size. Results -> results/experiment_results.csv.

Usage:
    # quick smoke test on 2 files
    python experiments/run_matrix.py --files data/02-14-2018.csv data/02-22-2018.csv
    # full run on all 9 days
    python experiments/run_matrix.py --all
"""
import argparse
import os
import pickle
import sys
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier

# Make `from src...` work when run as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import prepare_data           # noqa: E402
from src.augment import augment                    # noqa: E402
from src.feature_select import select_features_shap  # noqa: E402

ALL_FILES = [f"data/{d}.csv" for d in [
    "02-14-2018", "02-15-2018", "02-16-2018", "02-21-2018", "02-22-2018",
    "02-23-2018", "02-28-2018", "03-01-2018", "03-02-2018",
]]
# Rare attacks we report per-class F1 for (exact label strings in the dataset).
RARE = ["SQL Injection", "Brute Force -Web", "Brute Force -XSS"]


def build_model(name, seed):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed,
            class_weight="balanced_subsample",
        )
    return XGBClassifier(
        n_estimators=200, n_jobs=-1, random_state=seed,
        tree_method="hist", eval_metric="mlogloss",
    )


def score(model, X_test, y_test, le):
    """Return a metrics dict for a trained model."""
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    latency_ms = 1000 * (time.perf_counter() - t0) / len(X_test)

    rep = classification_report(
        y_test, y_pred, target_names=le.classes_,
        output_dict=True, zero_division=0,
    )
    row = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "latency_ms_per_flow": round(latency_ms, 4),
        "model_size_mb": round(len(pickle.dumps(model)) / 1e6, 2),
    }
    for cls in RARE:
        row[f"f1_{cls}"] = round(rep.get(cls, {}).get("f1-score", 0.0), 4)
    return row


def run(files, seed, top_k):
    print(f"Loading {len(files)} file(s)...")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                   ignore_index=True)
    df = df[df["Protocol"] != "Protocol"]

    # Clean + dedup + split, WITHOUT augmentation (we apply our own below).
    X_train, X_test, y_train, y_test, le = prepare_data(df, augment_minority=False)

    results = []
    for strategy in ["none", "smote", "ctgan"]:
        print(f"\n=== Augmentation: {strategy} ===")
        X_aug, y_aug = augment(X_train, y_train, le, strategy=strategy, seed=seed)

        for model_name in ["rf", "xgb"]:
            # --- all features ---
            model = build_model(model_name, seed)
            model.fit(X_aug, y_aug)
            row = {"strategy": strategy, "model": model_name,
                   "features": "all", "n_features": X_aug.shape[1], **score(model, X_test, y_test, le)}
            results.append(row)
            print(f"  {model_name}/all   macro_f1={row['macro_f1']}")

            # --- SHAP-selected features (reuse the model above for ranking) ---
            selected, _ = select_features_shap(model, X_aug, top_k=top_k, seed=seed)
            model_s = build_model(model_name, seed)
            model_s.fit(X_aug[selected], y_aug)
            row_s = {"strategy": strategy, "model": model_name,
                     "features": "shap", "n_features": len(selected),
                     **score(model_s, X_test[selected], y_test, le)}
            results.append(row_s)
            print(f"  {model_name}/shap  macro_f1={row_s['macro_f1']}")

    out = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    path = "results/experiment_results.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved {len(out)} rows -> {path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", help="specific CSV files to use")
    ap.add_argument("--all", action="store_true", help="use all 9 day files")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=30, help="SHAP features to keep")
    args = ap.parse_args()

    files = ALL_FILES if args.all else (args.files or ALL_FILES[:2])
    run(files, args.seed, args.top_k)
