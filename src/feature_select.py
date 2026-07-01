"""SHAP-based feature selection.

Trains nothing itself — give it an already-trained tree model (RF/XGBoost) and
the training features. It ranks features by how much they actually drive the
model's decisions (mean absolute SHAP value), then returns the top-k.

We use SHAP instead of PCA so the selected features stay human-readable — an
analyst can see *which* traffic features flag an attack (interpretability).
"""
import numpy as np
import pandas as pd
import shap


def select_features_shap(model, X, top_k=30, sample=1000, seed=42):
    """Return (selected_feature_names, full_ranking_series).

    SHAP is computed on a random `sample` of rows (full data is too slow).
    """
    Xs = X.sample(min(len(X), sample), random_state=seed)
    explainer = shap.TreeExplainer(model)
    sv = np.array(explainer.shap_values(Xs))

    # SHAP output shape varies by version / model. Collapse to one importance
    # per feature = mean absolute SHAP across samples (and classes if multiclass).
    if sv.ndim == 3:
        if sv.shape[0] == Xs.shape[0]:        # (samples, features, classes)
            importance = np.abs(sv).mean(axis=(0, 2))
        else:                                  # (classes, samples, features)
            importance = np.abs(sv).mean(axis=(0, 1))
    else:                                      # (samples, features)
        importance = np.abs(sv).mean(axis=0)

    ranking = pd.Series(importance, index=X.columns).sort_values(ascending=False)
    selected = ranking.head(top_k).index.tolist()
    print(f"SHAP selected top {len(selected)} of {X.shape[1]} features")
    return selected, ranking
