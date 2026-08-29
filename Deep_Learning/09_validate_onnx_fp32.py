from pathlib import Path
import random
import numpy as np
import pandas as pd
import onnxruntime as ort

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# -------------------------------------------------------
# Random Seed
# -------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Previously created test dataset
X_TEST_PATH = Path(__file__).resolve().parent / "X_test.csv"
Y_TEST_PATH = Path(__file__).resolve().parent / "y_test.csv"

# ONNX model
ONNX_PATH = PROJECT_ROOT / "ONNX" / "mlp_model.onnx"


# -------------------------------------------------------
# Check Required Files
# -------------------------------------------------------

if not X_TEST_PATH.exists():
    raise FileNotFoundError(
        f"Test feature file not found: {X_TEST_PATH}"
    )

if not Y_TEST_PATH.exists():
    raise FileNotFoundError(
        f"Test label file not found: {Y_TEST_PATH}"
    )

if not ONNX_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {ONNX_PATH}"
    )


# -------------------------------------------------------
# Load Previously Created Test Dataset
# -------------------------------------------------------

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH)


# -------------------------------------------------------
# Convert to NumPy Arrays
# -------------------------------------------------------

X_test = X_test.values.astype(np.float32)

y_test = y_test.values.ravel()


# -------------------------------------------------------
# Display Test Dataset Information
# -------------------------------------------------------

print("=" * 60)
print("Previously Created Test Dataset Loaded")
print("=" * 60)

print(f"\nTesting Samples : {len(X_test)}")
print(f"Testing Shape   : {X_test.shape}")

print("\nTesting Label Distribution")
print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
    .to_string()
)


# -------------------------------------------------------
# Load ONNX Model
# -------------------------------------------------------

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"],
)


input_name = session.get_inputs()[0].name

output_name = session.get_outputs()[0].name


# -------------------------------------------------------
# Run Inference
# -------------------------------------------------------

predictions = []

for sample in X_test:

    sample = sample.reshape(1, 6).astype(np.float32)

    output = session.run(
        [output_name],
        {input_name: sample}
    )[0]

    predictions.append(
        np.argmax(output)
    )


# -------------------------------------------------------
# Performance Metrics
# -------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
)

cm = confusion_matrix(
    y_test,
    predictions,
)


# -------------------------------------------------------
# Display Results
# -------------------------------------------------------

print("\n" + "=" * 60)
print("ONNX FP32 Validation")
print("=" * 60)

print(f"\nAccuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

print("\nConfusion Matrix")

print(cm)

print("\nClassification Report")

print(
    classification_report(
        y_test,
        predictions,
    )
)