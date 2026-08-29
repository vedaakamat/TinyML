"""
------------------------------------------------------------
File        : 13_validate_onnx_int8.py
Description : Validate the INT8 ONNX model using the test dataset
Author      : TinyML Workshop
------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import onnxruntime as ort

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# File Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

X_TEST_PATH = os.path.join(BASE_DIR, "X_test.csv")
Y_TEST_PATH = os.path.join(BASE_DIR, "y_test.csv")

MODEL_PATH = os.path.join(BASE_DIR, "..", "ONNX", "mlp_model_int8.onnx")

# ============================================================
# Load Test Dataset
# ============================================================

print("\nLoading test dataset...")

X_test = pd.read_csv(X_TEST_PATH).values.astype(np.float32)
y_test = pd.read_csv(Y_TEST_PATH).values.ravel().astype(np.int64)

print(f"Test Samples    : {X_test.shape[0]}")
print(f"Input Features  : {X_test.shape[1]}")

# ============================================================
# Load INT8 ONNX Model
# ============================================================

print("\nLoading INT8 ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Model loaded successfully.")

# ============================================================
# Run Inference
# ============================================================

print("\nRunning inference...")

predictions = []

for sample in X_test:

    # Reshape to (1, 6)
    sample = sample.reshape(1, -1).astype(np.float32)

    output = session.run(
        [output_name],
        {input_name: sample}
    )

    pred = np.argmax(output[0], axis=1)[0]

    predictions.append(pred)

predictions = np.array(predictions)

# ============================================================
# Compute Performance Metrics
# ============================================================

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

conf_matrix = confusion_matrix(y_test, predictions)

# ============================================================
# Display Results
# ============================================================

print("\n" + "=" * 60)
print("          INT8 ONNX MODEL VALIDATION RESULTS")
print("=" * 60)

print(f"Accuracy      : {accuracy * 100:.2f}%")
print(f"Precision     : {precision * 100:.2f}%")
print(f"Recall        : {recall * 100:.2f}%")
print(f"F1-Score      : {f1 * 100:.2f}%")

print("\nConfusion Matrix")
print("-" * 60)
print(conf_matrix)

print("\nClassification Report")
print("-" * 60)
print(classification_report(
    y_test,
    predictions,
    digits=4,
    zero_division=0
))

print("=" * 60)
print("INT8 ONNX Model validation completed successfully.")
print("=" * 60)