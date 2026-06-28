import joblib
from sklearn.ensemble import RandomForestClassifier

def train_rf(X_train, y_train, le):
    num_classes = len(le.classes_)
    print(f"Training multiclass Random Forest — {num_classes} classes: {list(le.classes_)}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,          # unlimited depth — better separation of complex classes
        class_weight="balanced_subsample",
        max_samples=0.6,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/random_forest.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    print("Model + label encoder saved.")

    return model
