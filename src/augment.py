"""Augmentation strategies for the Group 1 experiment.

All strategies are applied to the TRAINING split ONLY — never to the test set —
so no synthetic information can leak into evaluation. Pick one of:
    "none"  -> baseline, return training data unchanged
    "smote" -> classic interpolation oversampling (imbalanced-learn)
    "ctgan" -> conditional GAN that learns the data and invents new minority rows

Each minority class (non-Benign, currently below `target`) is brought up to
`target` samples. Majority classes are left untouched.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def _minority_targets(y_train, le, target):
    """Return {class_index: target} for non-Benign classes below `target`."""
    benign = le.transform(["Benign"])[0]
    counts = pd.Series(y_train).value_counts()
    return {
        cls: target
        for cls, cnt in counts.items()
        if cls != benign and cnt < target
    }


def augment(X_train, y_train, le, strategy="none", target=3000, seed=42,
            ctgan_epochs=100, ctgan_fit_cap=3000):
    """Return (X_aug, y_aug) for the chosen strategy. Training data only."""
    y_train = np.asarray(y_train)
    strategy = strategy.lower()

    if strategy == "none":
        return X_train, y_train

    plan = _minority_targets(y_train, le, target)
    if not plan:
        return X_train, y_train

    if strategy == "smote":
        # k_neighbors must be < the smallest class count, or SMOTE errors.
        smallest = pd.Series(y_train).value_counts().min()
        k = max(1, min(5, smallest - 1))
        sm = SMOTE(sampling_strategy=plan, random_state=seed, k_neighbors=k)
        X_aug, y_aug = sm.fit_resample(X_train, y_train)
        print(f"SMOTE: {len(y_train)} -> {len(y_aug)} rows (k_neighbors={k})")
        return X_aug, y_aug

    if strategy == "ctgan":
        return _augment_ctgan(X_train, y_train, le, plan, seed,
                              ctgan_epochs, ctgan_fit_cap)

    raise ValueError(f"Unknown strategy: {strategy!r} (use none/smote/ctgan)")


def _augment_ctgan(X_train, y_train, le, plan, seed, epochs, fit_cap):
    """Train one conditional CTGAN, then sample synthetic minority rows.

    CTGAN is trained on a per-class capped subsample (fit_cap rows/class) so
    training stays feasible on millions of rows, then we condition-sample each
    minority label up to its target.
    """
    from ctgan import CTGAN

    feature_cols = list(X_train.columns)
    df = X_train.copy()
    df["Label"] = y_train

    # Cap each class for CTGAN training (a GAN cannot train on 6M rows in time).
    # Shuffle once, then keep up to fit_cap rows per class (keeps the Label col).
    capped = (
        df.sample(frac=1, random_state=seed)
        .groupby("Label", group_keys=False)
        .head(fit_cap)
    )
    print(f"Training CTGAN on {len(capped)} rows ({epochs} epochs)...")

    ctgan = CTGAN(epochs=epochs, verbose=True)
    ctgan.fit(capped, discrete_columns=["Label"])

    counts = pd.Series(y_train).value_counts()
    synthetic = []
    for cls, tgt in plan.items():
        need = tgt - int(counts.get(cls, 0))
        if need <= 0:
            continue
        # Oversample then keep only rows CTGAN actually labelled as this class.
        sampled = ctgan.sample(need * 2, condition_column="Label",
                               condition_value=cls)
        sampled = sampled[sampled["Label"] == cls].head(need)
        synthetic.append(sampled)
        print(f"  CTGAN +{len(sampled)} synthetic for {le.classes_[cls]}")

    if not synthetic:
        return X_train, y_train

    syn = pd.concat(synthetic, ignore_index=True)
    X_syn = syn[feature_cols]
    y_syn = syn["Label"].to_numpy()

    X_aug = pd.concat([X_train, X_syn], ignore_index=True)
    y_aug = np.concatenate([y_train, y_syn])
    print(f"CTGAN: {len(y_train)} -> {len(y_aug)} rows")
    return X_aug, y_aug
