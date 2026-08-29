"""
==========================================================
File : 21_evaluate_stm32_int8.py

Purpose:
Evaluate the classification performance of the deployed
STM32 INT8 model.

Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
==========================================================
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================================
# Load Ground Truth Labels
# ==========================================================

print("=" * 60)
print("STM32 INT8 Model Evaluation")
print("=" * 60)

print("\nLoading ground truth labels...")

y_test = pd.read_csv("y_test.csv").values.ravel()

print(f"Number of Test Samples : {len(y_test)}")

# ==========================================================
# Load STM32 Predictions
# ==========================================================

print("\nLoading STM32 predictions...")

predictions = pd.read_csv(
    "stm32_int8_predictions.csv"
)["Prediction"].values

# ==========================================================
# Compute Metrics
# ==========================================================

print("\nEvaluating model...\n")

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    predictions
)

# ==========================================================
# Display Results
# ==========================================================

print("=" * 60)
print("Classification Performance")
print("=" * 60)

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1-Score  : {f1*100:.2f}%")

print("\nConfusion Matrix\n")

print(cm)

print("\nEvaluation Completed.")