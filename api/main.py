# Import libraries and packages
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Save FastAPI to a variable called "app"
app = FastAPI()

# Add CORS middleware that allows requests from my specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:5500",
                     "https://www.iwhayes.com",
                     "https://iwhayes.com"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

# Define a Pydantic model for the input data
class StartupInput(BaseModel):
    funding_rounds: int = Field(ge = 0, le = 8)
    founder_experience_years: int = Field(ge = 0, le = 24)
    team_size: int = Field(ge = 1, le = 300)

    market_size_billion: float = Field(ge = 1, le = 100)
    product_traction_users: int = Field(ge = 1_000, le = 900_000)
    burn_rate_million: float = Field(ge = 0.5, le = 75)
    revenue_million: float = Field(ge =0, le = 4_000_000)

    investor_type: Literal[
        "none",
        "angel",
        "tier1_vc",
        "tier2_vc"
    ]

    sector: Literal [
        "AI",
        "Climate",
        "Crypto",
        "Ecommerce",
        "Fintech",
        "Health",
        "SaaS"
    ]

    founder_background: Literal [
        "academic",
        "first_time",
        "ex_bigtech",
        "serial_founder"
    ]

# Define the model's path and put it in a variable.
MODEL_PATH = Path(__file__).resolve().parent.parent / "startup_model_v1.joblib"

# Load the model
model = joblib.load(MODEL_PATH)


#  Health check endpoint (verifying that the API is running)
@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction endpoint
@app.post("/predict")
def predict(input_data: StartupInput):
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data.model_dump()])

    # Make a prediction using the loaded model
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    # Map the prediction to a human-readable label
    probability_map = {
        label: float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }

    # Get the probability score and label
    confidence = float(max(probabilities))

    # Return the prediction and probabilities as a JSON response
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probability_map,
        "model_version": "1.0"
    }