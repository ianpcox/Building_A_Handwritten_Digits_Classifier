# Problem Formulation: Handwritten Digit Classification

## Research question

Can we classify images of handwritten digits (0–9) with high accuracy so that the system is suitable for deployment as a digit-recognition API (e.g. for form processing or document pipelines)?

## Success criteria

- **Primary metric:** Accuracy on a held-out test set. Target: ≥ 95% for 8×8 digits (e.g. sklearn digits dataset).
- **Secondary metrics:** Per-class F1, confusion matrix; baseline = simple classifier (e.g. logistic regression).
- **Optional:** Inference latency if productized; robustness to slight rotation or scaling (augmentation).

## Stakeholders and decisions

- **Product/forms:** Use in form-digit extraction or document pipelines.
- **Portfolio:** Demonstrates image classification, train/test split, baselines, and full metrics.
