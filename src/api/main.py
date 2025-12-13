from fastapi import FastAPI
import mlflow
import mlflow.sklearn
import pandas as pd

from src.api.pydantic_models import CreditRiskRequest, CreditRiskResponse

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predicts credit default probability using MLflow-registered model.",
    version="1.0",
)

# -------- Load model from MLflow Model Registry ---------
MODEL_NAME = "credit_risk_random_forest"  # use the best model you tracked
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")


@app.get("/")
def home():
    return {"message": "Credit Risk Model API is running."}


@app.post("/predict", response_model=CreditRiskResponse)
def predict_credit_risk(payload: CreditRiskRequest):
    data = payload.dict()

    df = pd.DataFrame([data])

    proba = model.predict(df)[0]

    return CreditRiskResponse(
        risk_probability=float(proba),
        model_version="latest"
    )
