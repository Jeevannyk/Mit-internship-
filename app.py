import io
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.inference_preprocess import preprocess_for_inference

app = FastAPI(title="IDS Prediction API")
app.mount("/static", StaticFiles(directory="static"), name="static")

MODELS = {}

def load_model(name: str):
    if name not in MODELS:
        path_map = {
            "xgboost": "models/xgboost.pkl",
            "random_forest": "models/random_forest.pkl",
            "lightgbm": "models/lightgbm.pkl",
        }
        if name not in path_map:
            raise HTTPException(status_code=400, detail=f"Unknown model: {name}")
        try:
            MODELS[name] = joblib.load(path_map[name])
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=f"Model '{name}' not trained yet. Run main.py first.")
    return MODELS[name]

le = joblib.load("models/label_encoder.pkl")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Query(default="random_forest"),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty")

    X = preprocess_for_inference(df)

    if X.empty:
        raise HTTPException(status_code=400, detail="No valid rows after preprocessing")

    model = load_model(model_name)

    pred_indices = model.predict(X).tolist()
    pred_labels = le.inverse_transform(pred_indices)
    proba_matrix = model.predict_proba(X)

    results = [
        {
            "row": i,
            "prediction": label,
            "is_attack": label != "Benign",
            "confidence": round(float(proba_matrix[i][pred_indices[i]]), 4),
        }
        for i, label in enumerate(pred_labels)
    ]

    attack_counts = {}
    for r in results:
        if r["is_attack"]:
            attack_counts[r["prediction"]] = attack_counts.get(r["prediction"], 0) + 1

    summary = {
        "total_rows": len(results),
        "attacks_detected": sum(1 for r in results if r["is_attack"]),
        "benign": sum(1 for r in results if not r["is_attack"]),
        "attack_breakdown": attack_counts,
        "model_used": model_name,
    }

    return JSONResponse({"summary": summary, "predictions": results})
