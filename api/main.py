from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
from pathlib import Path
import logging
from datetime import date

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths (models folder at project root)
project_root = Path(__file__).parent.parent
models_folder = project_root / "models"

# Load models safely
models = {}
model_files = {
    "xgboost_model": "best_model_xgboost.joblib"
}

for name, file in model_files.items():
    path = models_folder / file
    if path.exists():
        models[name] = joblib.load(path)
        logger.info(f"Loaded {name} from {path}")
    else:
        logger.error(f"Model file not found at {path}")

# Load preprocessor if exists
preprocessor_path = models_folder / "preprocessor.joblib"
preprocessor = None
if preprocessor_path.exists():
    preprocessor = joblib.load(preprocessor_path)
    logger.info("Loaded preprocessor")
else:
    logger.warning("Preprocessor not found!")

# Request schema
class PolicyData(BaseModel):
    Age: int
    Gender: str
    VehicleType: str
    Province: str
    CustomValueEstimate: float
    TotalPremium: float
    ZipCode: int
    VehicleIntroDate: date

# FastAPI app
app = FastAPI(
    title="AlphaCare Insurance Risk Prediction API",
    description="API to serve ML model for predicting insurance risk levels",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"message": "Welcome to AlphaCare Insurance Risk Prediction API 🚀"}

@app.post("/predict")
def predict_risk(data: PolicyData):
    if "xgboost_model" not in models:
        return {"error": "Model not loaded"}

    try:
        df = pd.DataFrame([data.dict()])

        # Apply preprocessor if available
        if preprocessor:
            df_transformed = preprocessor.transform(df)
            try:
                feature_names = preprocessor.get_feature_names_out()
                df = pd.DataFrame(df_transformed, columns=feature_names)
            except AttributeError:
                df = pd.DataFrame(df_transformed)

        model = models["xgboost_model"]

        # Align features
        expected_features = getattr(model, "n_features_in_", df.shape[1])
        if df.shape[1] < expected_features:
            for i in range(expected_features - df.shape[1]):
                df[f"extra_{i}"] = 0
        elif df.shape[1] > expected_features:
            df = df.iloc[:, :expected_features]

        # Predict
        prediction = model.predict(df)[0]

        return {"predicted_risk": float(prediction)}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
