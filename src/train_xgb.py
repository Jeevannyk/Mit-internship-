from xgboost import XGBClassifier
import joblib

def train_xgb(X_train, y_train):

    model = XGBClassifier(
        random_state=42,
        enable_categorical=True
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/xgboost.pkl")

    return model