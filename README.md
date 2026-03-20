# Handwritten Digit Recognition API (Project Elevate)

This repository transforms a basic tutorial notebook on K-Nearest Neighbors into a **production-ready machine learning pipeline and computer vision microservice**.

Using the scikit-learn Digits dataset, we elevated the project to compare **KNN**, **Support Vector Machines (SVM)**, and **Multi-Layer Perceptrons (MLP)**. The project now includes comprehensive metrics, t-SNE embeddings, misclassification analysis, and a deployable FastAPI microservice.

## Project Structure

* `run.py` — The core reproducible ML pipeline. Trains models, computes t-SNE embeddings, and outputs visualizations.
* `api.py` — A FastAPI microservice that wraps the trained SVM model for real-time digit classification.
* `docs/report.md` — A comprehensive paper-style report detailing methodology, model comparison, and error analysis.
* `docs/assets/` — Generated charts and visualizations supporting the report.
* `Building A Handwritten Digits Classifier.ipynb` — The original exploratory tutorial notebook.

## Key Findings

The Support Vector Machine (RBF Kernel) emerged as the superior model for production, achieving:
* **Test Accuracy:** 98.06%
* **CV Accuracy (5-fold):** 98.00%
* t-SNE embedding reveals distinct non-linear clusters, explaining why SVM and MLP outperform KNN.

Read the full analysis in [docs/report.md](docs/report.md).

## How to Run

### 1. Run the ML Pipeline
Generates all metrics, trains the models, and outputs visualizations to `docs/assets/`.
```bash
pip install -r requirements.txt
python run.py
```

### 2. Start the API Server
Starts the FastAPI microservice on `localhost:8000`.
```bash
uvicorn api:app --reload
```

### 3. Test the API
You can send a flat array of 64 pixel intensities (0-16 scale) to the `/predict` endpoint:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"pixels": [0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0]}'
```
*Expected Response:*
```json
{
  "prediction": 0,
  "confidence": 0.99,
  "latency_ms": 1.2
}
```
