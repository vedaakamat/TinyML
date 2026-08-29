"""
====================================================================
TinyML Workshop
Part V : Classical Machine Learning for TinyML Deployment

Slide 11 : Validating the Exported ONNX Model
====================================================================
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort

from sklearn.model_selection import train_test_split


# ==============================================================
# Project Paths
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "Datasets" / "dataset_processed.csv"

MODELS_DIR = PROJECT_ROOT / "Models"
ONNX_DIR = PROJECT_ROOT / "ONNX"

SKLEARN_MODEL = MODELS_DIR / "random_forest_model.pkl"
ONNX_MODEL = ONNX_DIR / "random_forest_model.onnx"


# ==============================================================
# Load Dataset
# ==============================================================

print("=" * 70)
print("Loading Processed Dataset...")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.\n")

X = df.iloc[:, :-1].astype(np.float32)
y = df.iloc[:, -1]


# ==============================================================
# Create Same Train-Test Split Used During Training
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
    shuffle=True
)


# ==============================================================
# Load Random Forest Model
# ==============================================================

print("=" * 70)
print("Loading Random Forest Model...")
print("=" * 70)

rf_model = joblib.load(SKLEARN_MODEL)

print("Random Forest model loaded successfully.\n")


# ==============================================================
# Load ONNX Model
# ==============================================================

print("=" * 70)
print("Loading ONNX Model...")
print("=" * 70)

session = ort.InferenceSession(
    str(ONNX_MODEL),
    providers=["CPUExecutionProvider"]
)

print("ONNX model loaded successfully.\n")


# ==============================================================
# Select Sample Test Data
# ==============================================================

sample = X_test.iloc[:10]
true_labels = y_test.iloc[:10].values


# ==============================================================
# Scikit-learn Prediction
# ==============================================================

sklearn_predictions = rf_model.predict(sample)


# ==============================================================
# ONNX Runtime Prediction
# ==============================================================

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

onnx_predictions = session.run(
    [output_name],
    {input_name: sample.values}
)[0]

onnx_predictions = np.array(onnx_predictions)


# ==============================================================
# Compare Results
# ==============================================================

print("=" * 70)
print("Prediction Comparison")
print("=" * 70)

print()

print(f"{'Sample':<8}{'True':<8}{'Scikit-learn':<18}{'ONNX Runtime'}")
print("-" * 60)

for i in range(len(sample)):
    print(
        f"{i+1:<8}"
        f"{true_labels[i]:<8}"
        f"{sklearn_predictions[i]:<18}"
        f"{onnx_predictions[i]}"
    )

print()

# ==============================================================
# Validation
# ==============================================================

if np.array_equal(sklearn_predictions, onnx_predictions):

    print("=" * 70)
    print("Validation Successful")
    print("=" * 70)

    print("Scikit-learn and ONNX Runtime produce identical predictions.")
    print("The exported ONNX model is ready for embedded deployment.")

else:

    print("=" * 70)
    print("Validation Failed")
    print("=" * 70)

    print("Prediction mismatch detected between the two models.")