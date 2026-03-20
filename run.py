"""
Single entry point: handwritten digit classification with baseline and MLP.
Reproducible pipeline for Project Elevate (Phases 1-4).
Usage: python run.py [--out RESULTS_JSON]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    parser = argparse.ArgumentParser(description="Digit classification pipeline")
    parser.add_argument("--out", default=None, help="Optional path to write metrics JSON")
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    data = load_digits()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Baseline: logistic regression
    lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average="weighted")
    print("\n--- Baseline: Logistic Regression ---")
    print(f"  Accuracy: {acc_lr:.4f}")
    print(f"  F1 (weighted): {f1_lr:.4f}")
    print("  Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_lr))

    # Main: MLP
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=RANDOM_STATE)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    f1_mlp = f1_score(y_test, y_pred_mlp, average="weighted")
    print("\n--- Main: MLP (128, 64) ---")
    print(f"  Accuracy: {acc_mlp:.4f}")
    print(f"  F1 (weighted): {f1_mlp:.4f}")
    print("  Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_mlp))
    print("\nPer-class report:")
    print(classification_report(y_test, y_pred_mlp))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "baseline_lr": {"accuracy": float(acc_lr), "f1_weighted": float(f1_lr)},
                "mlp": {"accuracy": float(acc_mlp), "f1_weighted": float(f1_mlp)},
            }, f, indent=2)
        print(f"\nMetrics written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
