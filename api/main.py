# Import libraries and packages
from pathlib import Path
from fastapi import FastAPI
import joblib

# Save FastAPI to a variable called "app"
app = FastAPI()

# Define the model's path and put it in a variable.
MODEL_PATH = Path(__file__).resolve().parent.parent / "startup_model_v1.joblib"

# load the model
model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}