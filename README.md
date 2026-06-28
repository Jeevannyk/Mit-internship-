# IDS - Intrusion Detection System (ML Project)

Trains machine learning models to classify network traffic as **Benign** or **Attack** using the CSE-CIC-IDS2018 dataset.

---

## Project Structure

```
.
├── data/                   # Raw CSV files from the CIC-IDS2018 dataset (one per day)
├── models/                 # Saved trained model files (.pkl)
├── results/                # Evaluation metrics saved as CSV files
├── src/                    # All source code modules
├── notebooks/              # Jupyter notebook for exploration and analysis
├── main.py                 # Entry point — run this to train and evaluate
└── requirements.txt        # Python dependencies
```

---

## File Descriptions

### Root

| File | What it does |
|---|---|
| `main.py` | Loads all CSVs, preprocesses data, trains XGBoost, evaluates and saves metrics |
| `requirements.txt` | Lists all Python packages needed to run the project |

### `src/`

| File | What it does |
|---|---|
| `load_data.py` | Reads and concatenates all CSV files from the `data/` folder into one DataFrame |
| `preprocess.py` | Cleans the data — drops identifier columns, fixes inf/NaN values, clips float32 overflow, encodes Label as 0/1, returns train/test split |
| `train_rf.py` | Trains an XGBoost classifier and saves it to `models/xgboost.pkl` |
| `train_xgb.py` | Trains an XGBoost classifier and saves it to `models/xgboost.pkl` |
| `train_lgbm.py` | Trains a LightGBM classifier and saves it to `models/lightgbm.pkl` |
| `evaluate.py` | Runs predictions on test set, prints accuracy/precision/recall/F1, saves metrics CSV to `results/` |
| `feature_analysis.py` | Prints the shape of train/test splits — used to inspect data dimensions |
| `utils.py` | Shared utility functions (currently empty) |

### `data/`

Daily network traffic captures from the CIC-IDS2018 dataset. Each CSV contains labeled flow records.

| File | Attack type |
|---|---|
| `02-14-2018.csv` | FTP-BruteForce, SSH-BruteForce |
| `02-15-2018.csv` | DoS-GoldenEye, DoS-Slowloris |
| `02-16-2018.csv` | DoS-SlowHTTPTest, DoS-Hulk |
| `02-21-2018.csv` | BruteForce-Web, BruteForce-XSS |
| `02-22-2018.csv` | SQL Injection |
| `02-23-2018.csv` | Infiltration |
| `02-28-2018.csv` | BotNet |
| `03-01-2018.csv` | BotNet |
| `03-02-2018.csv` | BotNet |

### `models/`

| File | What it contains |
|---|---|
| `random_forest.pkl` | Old trained Random Forest model (replaced by XGBoost) |
| `xgboost.pkl` | Trained XGBoost model |
| `lightgbm.pkl` | Trained LightGBM model |

### `results/`

| File | What it contains |
|---|---|
| `xgb_metrics.csv` | Accuracy, precision, recall, F1 for XGBoost |
| `xgb_metrics.csv` | Accuracy, precision, recall, F1 for XGBoost |
| `feature_importance.csv` | Feature importance scores from the trained model |

---

## How to Run

> **Note:** The `data/`, `models/`, and `results/` folders are **not included** in this repo (too large for GitHub). Download the [CSE-CIC-IDS2018 dataset](https://www.unb.ca/cic/datasets/ids-2018.html), place the daily CSVs in a `data/` folder, then run the steps below to train the models and generate results yourself.

```bash
# Install dependencies
pip install -r requirements.txt

# Train and evaluate
python main.py
```
