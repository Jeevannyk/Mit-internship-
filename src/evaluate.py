import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate(model, X_test, y_test, le, results_dir="results"):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}  (the number that matters for imbalanced IDS)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Per-class precision/recall/F1 — recall on rare attacks is what we care about.
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv(f"{results_dir}/metrics.csv")

    # Confusion matrix — rows = true class, cols = predicted.
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(
        f"{results_dir}/confusion_matrix.csv"
    )
    print(f"Metrics + confusion matrix saved to {results_dir}/")
