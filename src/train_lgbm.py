import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

def train_lgbm(X_train, y_train, le):
    num_classes = len(le.classes_)
    print(f"Training multiclass LightGBM — {num_classes} classes: {list(le.classes_)}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    counts = pd.Series(y_tr).value_counts()
    class_weight = {
        cls: min(len(y_tr) / (num_classes * cnt), 5.0)
        for cls, cnt in counts.items()
    }
    sample_weights = np.array([class_weight[c] for c in y_tr])
    print("Class weights:", {le.classes_[k]: round(v, 2) for k, v in class_weight.items()})

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        objective="multiclass",
        num_class=num_classes,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=True),
            lgb.log_evaluation(period=20),
        ],
    )

    joblib.dump(model, "models/lightgbm.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    print("LightGBM model + label encoder saved.")

    return model
