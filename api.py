"""
Digit Recognition API — FastAPI Microservice
Project Elevate: Building_A_Handwritten_Digits_Classifier

Endpoints:
  POST /predict        — Classify a single 8x8 digit image (64 pixel values)
  POST /predict-batch  — Classify multiple images in one request
  GET  /health         — Health check
  GET  /model-info     — Model metadata and performance summary

Usage:
  uvicorn api:app --reload
"""

import time
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Digit Recognition API",
    description=(
        "Classifies handwritten digits (0-9) from 8x8 pixel images using a "
        "Support Vector Machine (RBF kernel). Trained on the scikit-learn Digits dataset."
    ),
    version="1.0.0",
)

# ── Model Training at Startup ─────────────────────────────────────────────────
RANDOM_STATE = 42
_data = load_digits()
_X, _y = _data.data, _data.target
_X_train, _X_test, _y_train, _y_test = train_test_split(
    _X, _y, test_size=0.2, random_state=RANDOM_STATE, stratify=_y
)
_scaler = StandardScaler()
_X_train_s = _scaler.fit_transform(_X_train)
_X_test_s = _scaler.transform(_X_test)

_model = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)
_model.fit(_X_train_s, _y_train)

_test_accuracy = float((_model.predict(_X_test_s) == _y_test).mean())
_n_train = len(_X_train)
_n_test = len(_X_test)


# ── Schemas ───────────────────────────────────────────────────────────────────
class DigitInput(BaseModel):
    pixels: List[float]

    @field_validator("pixels")
    @classmethod
    def validate_pixels(cls, v):
        if len(v) != 64:
            raise ValueError(f"Expected exactly 64 pixel values, got {len(v)}.")
        if any(p < 0 or p > 16 for p in v):
            raise ValueError("Pixel values must be in the range [0, 16].")
        return v


class BatchDigitInput(BaseModel):
    images: List[List[float]]

    @field_validator("images")
    @classmethod
    def validate_images(cls, v):
        if len(v) == 0:
            raise ValueError("At least one image is required.")
        if len(v) > 100:
            raise ValueError("Maximum batch size is 100 images.")
        for i, img in enumerate(v):
            if len(img) != 64:
                raise ValueError(f"Image at index {i} has {len(img)} pixels; expected 64.")
        return v


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    all_probabilities: dict
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: List[int]
    confidences: List[float]
    latency_ms: float
    count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    kernel: str
    regularization_C: float
    n_training_samples: int
    n_test_samples: int
    test_accuracy: float
    feature_description: str
    classes: List[int]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health():
    """Returns service health status."""
    return HealthResponse(status="ok", model="SVM-RBF", version="1.0.0")


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Utility"])
def model_info():
    """Returns metadata about the trained model and its performance."""
    return ModelInfoResponse(
        model_type="Support Vector Machine",
        kernel="RBF (Radial Basis Function)",
        regularization_C=10.0,
        n_training_samples=_n_train,
        n_test_samples=_n_test,
        test_accuracy=round(_test_accuracy, 4),
        feature_description="64 pixel intensities from an 8x8 grayscale image (values 0-16)",
        classes=list(range(10)),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(body: DigitInput):
    """
    Classify a single handwritten digit image.

    - **pixels**: A flat list of exactly 64 float values representing the 8x8 pixel
      intensities of the digit image. Values should be in the range [0, 16].
    """
    t0 = time.perf_counter()
    try:
        X = np.array(body.pixels).reshape(1, -1)
        X_scaled = _scaler.transform(X)
        pred = int(_model.predict(X_scaled)[0])
        proba = _model.predict_proba(X_scaled)[0]
        confidence = float(proba[pred])
        all_probs = {str(i): round(float(p), 4) for i, p in enumerate(proba)}
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return PredictionResponse(
            prediction=pred,
            confidence=round(confidence, 4),
            all_probabilities=all_probs,
            latency_ms=latency_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(body: BatchDigitInput):
    """
    Classify multiple handwritten digit images in a single request.

    - **images**: A list of up to 100 images, each a flat list of 64 pixel values.
    """
    t0 = time.perf_counter()
    try:
        X = np.array(body.images)
        X_scaled = _scaler.transform(X)
        preds = _model.predict(X_scaled).tolist()
        probas = _model.predict_proba(X_scaled)
        confidences = [round(float(probas[i, p]), 4) for i, p in enumerate(preds)]
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return BatchPredictionResponse(
            predictions=preds,
            confidences=confidences,
            latency_ms=latency_ms,
            count=len(preds),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
