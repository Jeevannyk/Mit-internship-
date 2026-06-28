import pandas as pd
import numpy as np

from src.preprocess import prepare_data
from src.train_rf import train_rf
from src.train_lgbm import train_lgbm
from src.evaluate import evaluate
from src.feature_importance import show_feature_importance

# Start with ONE file to keep things fast and light on memory.
# Once this works, you can switch to load_dataset() to use all CSVs.
print("Loading data...")
train_files = [
    "data/02-14-2018.csv",
    "data/02-15-2018.csv",
    "data/02-16-2018.csv",
    "data/02-21-2018.csv",
    "data/02-22-2018.csv",
    "data/02-23-2018.csv",
    "data/02-28-2018.csv",
    "data/03-01-2018.csv",
    "data/03-02-2018.csv",
]

dfs = []
for i, f in enumerate(train_files, 1):
    print(f"  [{i}/{len(train_files)}] Loading {f}...")
    dfs.append(pd.read_csv(f, low_memory=False))
df = pd.concat(dfs, ignore_index=True)

# Remove duplicated header rows accidentally read as data
df = df[df["Protocol"] != "Protocol"]

print("Preparing data...")
X_train, X_test, y_train, y_test, le = prepare_data(df)

print("Training LightGBM...")
print("Train inf values:", np.isinf(X_train.select_dtypes(include=np.number)).sum().sum())
print("Test inf values:", np.isinf(X_test.select_dtypes(include=np.number)).sum().sum())
model = train_lgbm(X_train, y_train, le)

print("Evaluating...")
evaluate(model, X_test, y_test, le)

print("Feature Importance...")
show_feature_importance(model, X_train)