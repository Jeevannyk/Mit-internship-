import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def evaluate(model, X_test, y_test, le, results_path="results/xgb_metrics.csv"):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    pd.DataFrame([{"accuracy": acc}]).to_csv(results_path, index=False)
    print(f"Metrics saved to {results_path}")
