from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_model.pkl")

class IrisFeatures(BaseModel):
    features: list[float]

app = FastAPI(title="Iris Classifier API")

@app.get("/")
def read_root():
    return {"message": "FastAPI Iris Classifier is running!"}

@app.post("/predict")
def predict(iris: IrisFeatures):
    data = np.array(iris.features).reshape(1, -1)
    prediction = model.predict(data)[0]
    return {"prediction": int(prediction)}

# To run: uvicorn app_fastapi:app --reload --host 0.0.0.0 --port 8000